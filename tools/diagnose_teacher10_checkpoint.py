"""Fixed-patch Teacher10 error/gradient audit; no optimizer or checkpoint writes.

Run manually in the server environment. Does not compute benchmark CD/P2M.
"""

import argparse
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import copy
import csv
from datetime import datetime, timezone
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
NODES = (0, 2, 5, 7, 10)
CONDITIONS = tuple((points, noise) for points in (10000, 50000) for noise in (0.01, 0.02, 0.03))


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(seed, *values):
    text = ':'.join(map(str, (seed,) + values))
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], 'little')


def label_for(points, noise):
    return f'{points // 1000}k_{round(noise * 100)}pct'


def inspect_run(train_root, torch):
    manifest = json.loads((train_root / 'manifest.json').read_text(encoding='utf-8'))
    rows = [json.loads(line) for line in (train_root / 'train.jsonl').read_text(encoding='utf-8-sig').splitlines()
            if line.strip()]
    selected = [r for r in rows if r.get('val_rollout_score') is not None]
    if not selected:
        raise ValueError(f'No validation records in {train_root}')
    best = min(selected, key=lambda r: r['val_rollout_score'])
    path = train_root / 'ckpt-best.pth'
    options = dict(map_location='cpu')
    if 'weights_only' in inspect.signature(torch.load).parameters:
        options['weights_only'] = False
    saved = torch.load(path, **options)
    if saved['epoch'] != best['epoch'] or saved['distillation'] != manifest['schedule']:
        raise ValueError(f'Checkpoint and best log disagree: {path}')
    if saved['distillation']['teacher_nodes'] != list(NODES):
        raise ValueError('This audit requires Teacher10 / Student4 nodes [0,2,5,7,10]')
    if saved.get('teacher_endpoint') != manifest['config'].get('teacher_endpoint'):
        raise ValueError(f'Checkpoint endpoint objective differs from manifest: {path}')
    if saved.get('training_input') != manifest['config']['training_input']:
        raise ValueError(f'Checkpoint local objective differs from manifest: {path}')
    del saved
    return manifest, dict(train_root=str(train_root), checkpoint=str(path), epoch=best['epoch'],
                          checkpoint_sha256=sha256(path), validation_score=best['val_rollout_score'],
                          objective=manifest['config'].get('teacher_endpoint'))


@contextmanager
def isolated_mode(student, mode, seed, torch, np):
    """Restore all running buffers, modes, RNGs and gradients after each probe."""
    modes = [(module, module.training) for module in student.modules()]
    buffers = {name: tensor.detach().clone() for name, tensor in student.named_buffers()}
    versions = {name: parameter._version for name, parameter in student.named_parameters()}
    py_state, numpy_state = random.getstate(), np.random.get_state()
    device = next(student.parameters()).device
    devices = [device.index] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            random.seed(seed)
            np.random.seed(seed)
            torch.random.default_generator.manual_seed(seed)
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            student.train(mode == 'train')
            if mode == 'bn_batch_stats':
                for module in student.modules():
                    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                        module.train()
            student.zero_grad(set_to_none=True)
            yield
    finally:
        with torch.no_grad():
            for name, tensor in student.named_buffers():
                tensor.copy_(buffers[name])
        for module, training in modes:
            module.training = training
        student.zero_grad(set_to_none=True)
        random.setstate(py_state)
        np.random.set_state(numpy_state)
        if any(p._version != versions[n] for n, p in student.named_parameters()):
            raise RuntimeError('A diagnostic probe modified Student parameter values')


def gradient_snapshot(student, torch):
    gradients = {}
    group_squares = {}
    parameter_norms = []
    for name, parameter in student.named_parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad.detach().float().cpu().clone()
        norm = float(gradient.norm())
        if not math.isfinite(norm):
            raise FloatingPointError(f'Nonfinite gradient in {name}')
        gradients[name] = gradient
        group = name.split('.')[0]
        group_squares[group] = group_squares.get(group, 0.0) + norm * norm
        parameter_norms.append(dict(parameter=name, norm=norm))
    norm = math.sqrt(sum(group_squares.values()))
    return gradients, dict(norm=norm,
                           groups={key: math.sqrt(value) for key, value in group_squares.items()},
                           top_parameters=sorted(parameter_norms, key=lambda r: r['norm'], reverse=True)[:12])


def cosine_to_reference(gradients, reference, norm, reference_norm, torch):
    if norm == 0 or reference_norm == 0:
        return None
    dot = sum(float(torch.dot(g.reshape(-1), reference[name].reshape(-1)))
              for name, g in gradients.items() if name in reference)
    return max(-1.0, min(1.0, dot / (norm * reference_norm)))


def make_bank(config, points, noise, args, teacher, runner, torch):
    from datasets.ScoreDenoiseDataset import ScoreDenoise
    dataset_config = copy.deepcopy(config.dataset._base_)
    dataset_config.TEST_RESOLUTION = f'{points}_poisson'
    dataset_config.TEST_NOISE = noise
    dataset_config.TEST_NOISY_DIR = f'PUNet_{points}_poisson_{noise:.2f}'
    dataset_config.pop('TEST_NOISY_PATH', None)
    dataset_config.NUM_WORKERS = 0
    _, loader = ScoreDenoise(argparse.Namespace(distributed=False), dataset_config).test_dataloader()
    names = list(loader.dataset.names)
    if len(names) != 20 or names != sorted(names):
        raise ValueError('Expected the sorted 20-cloud PUNet test dataset')
    selected = names[:args.max_shapes]
    patches, samples = [], []
    for index, name in enumerate(selected):
        sample = loader.dataset[index]
        noisy = sample['pcl_noisy'].float()
        if noisy.shape != (points, 3) or not torch.isfinite(noisy).all():
            raise ValueError(f'{name}: invalid normalized cloud')
        generator = torch.Generator(device='cpu').manual_seed(stable_seed(args.seed, points, name))
        seeds = torch.randint(points, (args.patches_per_shape,), generator=generator)
        for patch_number, seed in enumerate(seeds.tolist()):
            indices = (noisy - noisy[seed]).square().sum(-1).topk(
                int(config.inference_patch_size), largest=False).indices
            patch = noisy[indices].clone()
            patches.append(patch)
            samples.append(dict(name=name, patch=patch_number, seed_index=seed,
                                knn_indices_sha256=hashlib.sha256(indices.numpy().tobytes()).hexdigest(),
                                noisy_patch_sha256=hashlib.sha256(patch.numpy().tobytes()).hexdigest()))
    noisy_patches = torch.stack(patches)
    sigmas = torch.full((len(patches),), noise, dtype=noisy_patches.dtype)
    states = runner.capture_teacher(teacher, noisy_patches, sigmas,
                                    patch_batch=1, teacher_nodes=NODES)
    return dict(nodes=states, sigmas=sigmas, samples=samples, selected_names=selected,
                dataset_shapes=len(names), resolution=points, noise=noise,
                label=label_for(points, noise))


def trajectory_errors(student, teacher, bank, runner, torch):
    rows = []
    device = next(student.parameters()).device
    with torch.no_grad():
        for patch, sample in enumerate(bank['samples']):
            states = bank['nodes'][:, patch:patch + 1].to(device)
            sigma0 = bank['sigmas'][patch:patch + 1].to(device)
            free = states[0]
            for stage, (start, stop) in enumerate(zip(NODES[:-1], NODES[1:])):
                sigma = sigma0 * runner.TEACHER_DECAY ** start
                tf = runner.forward_student_interval(student, states[stage], sigma, start, stop)
                # Stage 1 is the same function on the same input; compute it once.
                prediction = (tf if stage == 0 else runner.forward_student_interval(
                    student, free, sigma, start, stop))
                actual_target = (states[1] if stage == 0 else runner.teacher_interval_target(
                    teacher, free, sigma0, start, stop, patch_batch=1))
                def mse(left, right):
                    return float((left - right).square().sum(-1).mean())
                teacher_forced = mse(tf, states[stage + 1])
                canonical = mse(prediction, states[stage + 1])
                rows.append(dict(**sample, stage=stage + 1, start_step=start, target_step=stop,
                                 teacher_forced_mse=teacher_forced,
                                 free_to_canonical_mse=canonical,
                                 free_minus_teacher_forced_mse=canonical - teacher_forced,
                                 free_to_input_matched_target_mse=mse(prediction, actual_target),
                                 matched_target_to_canonical_mse=mse(actual_target, states[stage + 1]),
                                 teacher_move_mse=mse(states[stage], states[stage + 1])))
                free = prediction
    return rows


def endpoint_backward(student, nodes, sigmas, mode, runner, torch):
    """Match the training endpoint loss; last_only disconnects the first 3 steps."""
    from torch.utils.checkpoint import checkpoint
    device = next(student.parameters()).device
    total = 0.0
    state_gradient_squares = [0.0] * 4
    for index in range(nodes.shape[1]):
        state = nodes[0, index:index + 1].detach().to(device)
        target = nodes[-1, index:index + 1].detach().to(device)
        sigma0 = sigmas[index:index + 1].to(device)
        retained = []
        for stage, (start, stop) in enumerate(zip(NODES[:-1], NODES[1:])):
            sigma = sigma0 * runner.TEACHER_DECAY ** start
            def step(points, interval_sigma=sigma, first=start, last=stop):
                return runner.forward_student_interval(student, points, interval_sigma, first, last)
            if mode == 'last_only' and stage < 3:
                with torch.no_grad():
                    state = step(state).detach()
            else:
                state = checkpoint(step, state, use_reentrant=False)
                state.retain_grad()
                retained.append((stage, state))
        loss = (state - target).square().sum(-1).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite endpoint probe loss')
        (loss / nodes.shape[1]).backward()
        for stage, value in retained:
            if value.grad is not None:
                state_gradient_squares[stage] += float(value.grad.detach().norm()) ** 2
        total += float(loss.detach()) / nodes.shape[1]
    return total, [math.sqrt(value) for value in state_gradient_squares]


def gradient_probes(student, teacher, bank, args, config, runner, torch, np):
    count = min(int(config.total_bs), len(bank['samples']))
    if count != int(config.total_bs):
        raise ValueError('Need enough diagnostic patches for one original optimizer batch')
    nodes, sigmas = bank['nodes'][:, :count], bank['sigmas'][:count]
    generator = torch.Generator(device='cpu').manual_seed(stable_seed(args.seed, 'mask', bank['label']))
    mask = runner.sample_teacher_input_mask(count, args.teacher_probability, generator)
    seed = stable_seed(args.seed, 'grad', bank['label'])
    cases = [('local_train', 'train'), ('local_eval', 'eval'),
             ('endpoint_eval_full', 'eval'), ('endpoint_eval_last_only', 'eval'),
             ('endpoint_bn_batch_stats_full', 'bn_batch_stats'), ('endpoint_train_full', 'train')]
    rows = []
    reference = None
    reference_norm = None
    for case, module_mode in cases:
        print(f'[gradient] {bank["label"]}: {case}', flush=True)
        with isolated_mode(student, module_mode, seed, torch, np):
            if case.startswith('local_'):
                losses = runner.backward_student_input_stages(
                    student, nodes, sigmas, mask, teacher_nodes=NODES,
                    student_target='teacher_from_student_input', teacher=teacher, teacher_patch_batch=1)
                loss = sum(losses) / 4
                state_norms = None
            else:
                loss, state_norms = endpoint_backward(
                    student, nodes, sigmas, 'last_only' if case.endswith('last_only') else 'full', runner, torch)
            gradients, metrics = gradient_snapshot(student, torch)
            if not gradients or not math.isfinite(loss):
                raise RuntimeError(f'{case}: missing gradients or nonfinite loss')
            if case == 'local_train':
                reference = gradients
                reference_norm = metrics['norm']
            row = dict(case=case, module_mode=module_mode, loss=loss, **metrics,
                       norm_relative_to_local_train=metrics['norm'] / reference_norm if reference_norm else None,
                       cosine_to_local_train=cosine_to_reference(
                           gradients, reference, metrics['norm'], reference_norm, torch),
                       student_output_gradient_norms=state_norms,
                       state_and_rng_restored=True, patches=count,
                       teacher_probability=args.teacher_probability,
                       teacher_counts=mask.sum(1).tolist())
            rows.append(row)
            if case != 'local_train':
                del gradients
    return rows


def summarize_stages(rows):
    groups = {}
    metrics = ('teacher_forced_mse', 'free_to_canonical_mse', 'free_minus_teacher_forced_mse',
               'free_to_input_matched_target_mse', 'matched_target_to_canonical_mse', 'teacher_move_mse')
    for row in rows:
        key = (row['checkpoint_label'], row['condition'], row['stage'])
        groups.setdefault(key, []).append(row)
    result = []
    for (checkpoint, condition, stage), samples in groups.items():
        result.append(dict(checkpoint_label=checkpoint, condition=condition, stage=stage,
                           patches=len(samples), **{key: sum(r[key] for r in samples) / len(samples)
                                                    for key in metrics}))
    return result


def run(args, output, manifest):
    import numpy as np
    import torch
    from easydict import EasyDict
    from tools import builder, runner_distill as runner
    from utils.gpu_memory import apply_gpu_memory_limit

    primary_manifest, primary = inspect_run(args.train_root, torch)
    configurations = [('endpoint', primary_manifest, primary)]
    if args.reference_train_root is not None:
        control_manifest, control = inspect_run(args.reference_train_root, torch)
        before = copy.deepcopy(primary_manifest['config'])
        before.pop('teacher_endpoint', None)
        if before != control_manifest['config'] or primary_manifest['checkpoint'] != control_manifest['checkpoint']:
            raise ValueError('The control must use the same Teacher10/Mixed settings without teacher_endpoint')
        configurations.insert(0, ('control', control_manifest, control))
    config = EasyDict(primary_manifest['config'])
    if runner.configured_teacher_nodes(config) != NODES or config.dataset._base_.DATASET != 'PUNet':
        raise ValueError('Expected fixed Teacher10 / Student4 on PUNet')
    if not config.get('teacher_endpoint'):
        raise ValueError('--train_root must identify the endpoint-alignment experiment')
    for source, expected in primary_manifest.get('source_sha256', {}).items():
        raw = (ROOT / source).read_bytes()
        if expected not in (hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()):
            raise ValueError(f'Current source differs from the trained endpoint implementation: {source}')
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(int(config.cpu_threads))
    manifest['gpu_memory_limit'] = apply_gpu_memory_limit(config, device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    teacher_path = (args.teacher_ckpt or Path(primary_manifest['checkpoint'])).expanduser().resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(teacher_path)
    protected_paths = [teacher_path] + [Path(item['checkpoint']) for _, _, item in configurations]
    protected_stats = {str(path): (path.stat().st_size, path.stat().st_mtime_ns) for path in protected_paths}
    manifest.update(checkpoints={name: item for name, _, item in configurations},
                    teacher_checkpoint=str(teacher_path), teacher_sha256=sha256(teacher_path),
                    effective_config=config,
                    diagnostic_subset=True, benchmark_metrics=False,
                    protocol=dict(nodes=list(NODES), teacher_eta=0.3, teacher_decay=0.95,
                                  patch_size=int(config.inference_patch_size),
                                  patch_selection='sorted_first_shapes_then_fixed_random_seed_and_noisy_KNN',
                                  stage_metric='corresponding_point_squared_l2_in_normalized_patch_coordinates',
                                  gradient_teacher_probability=args.teacher_probability,
                                  gradient_batch_size=int(config.total_bs),
                                  gradient_endpoint_patch_batch=1,
                                  gradient_conditions=args.gradient_conditions,
                                  exact_training_batch_replay=False,
                                  no_optimizer=True, no_parameter_update=True,
                                  bn_buffers_restored=True, three_percent_is_diagnostic_only=True))
    write_json(output / 'run_manifest.json', manifest)
    teacher = runner.freeze_teacher(builder.model_builder(config.model).to(device))
    builder.load_model(teacher, str(teacher_path))
    runner.freeze_teacher(teacher)
    banks = []
    for points, noise in CONDITIONS:
        print(f'[fixed patches] {label_for(points, noise)}', flush=True)
        banks.append(make_bank(config, points, noise, args, teacher, runner, torch))
    if any(b['selected_names'] != banks[0]['selected_names'] for b in banks):
        raise ValueError('Selected diagnostic cloud names differ across conditions')
    manifest['samples'] = {b['label']: b['samples'] for b in banks}
    write_json(output / 'run_manifest.json', manifest)
    student = builder.model_builder(config.model).to(device)
    all_rows, probes = [], []
    for checkpoint_label, _, checkpoint_info in configurations:
        runner.load_student_checkpoint(student, checkpoint_info['checkpoint'], builder, expected_nodes=NODES)
        student.requires_grad_(True).eval()
        versions = {name: p._version for name, p in student.named_parameters()}
        initial_buffers = {name: b.detach().clone() for name, b in student.named_buffers()}
        for bank in banks:
            print(f'[trajectory] {checkpoint_label}: {bank["label"]}', flush=True)
            with isolated_mode(student, 'eval', stable_seed(args.seed, 'trajectory', bank['label']), torch, np):
                rows = trajectory_errors(student, teacher, bank, runner, torch)
            for row in rows:
                row.update(checkpoint_label=checkpoint_label, condition=bank['label'])
            all_rows.extend(rows)
            if bank['label'] in args.gradient_conditions:
                results = gradient_probes(student, teacher, bank, args, config, runner, torch, np)
                probes.append(dict(checkpoint_label=checkpoint_label, condition=bank['label'], cases=results))
            write_json(output / 'gradient_probes.json', probes)
            write_json(output / 'stage_summary.json', summarize_stages(all_rows))
        if any(p._version != versions[n] or p.grad is not None for n, p in student.named_parameters()):
            raise RuntimeError('Student integrity check failed')
        if any(not torch.equal(b, initial_buffers[n]) for n, b in student.named_buffers()):
            raise RuntimeError('Student running statistics changed')
    if teacher.training or any(p.requires_grad or p.grad is not None for p in teacher.parameters()):
        raise RuntimeError('Teacher integrity check failed')
    for path in protected_paths:
        if (path.stat().st_size, path.stat().st_mtime_ns) != protected_stats[str(path)]:
            raise RuntimeError(f'A checkpoint file changed during the audit: {path}')
    with (output / 'stage_errors.csv').open('x', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    expected_rows = len(configurations) * len(banks) * args.max_shapes * args.patches_per_shape * 4
    if len(all_rows) != expected_rows or len(probes) != len(configurations) * len(args.gradient_conditions):
        raise RuntimeError('Incomplete diagnostic enumeration')
    manifest.update(status='completed', stage_rows=len(all_rows), gradient_probe_batches=len(probes),
                    integrity=dict(parameter_values_unchanged=True, buffers_restored=True,
                                   checkpoint_files_unchanged=True, teacher_frozen=True,
                                   no_optimizer_created=True),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device))


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train_root', type=Path, required=True)
    parser.add_argument('--reference_train_root', type=Path)
    parser.add_argument('--teacher_ckpt', type=Path)
    parser.add_argument('--output_dir', type=Path)
    parser.add_argument('--max_shapes', type=int, default=4)
    parser.add_argument('--patches_per_shape', type=int, default=2)
    parser.add_argument('--gradient_conditions', nargs='*', default=['10k_1pct', '50k_3pct'])
    parser.add_argument('--teacher_probability', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if (not 1 <= args.max_shapes <= 20 or args.patches_per_shape < 1 or args.device < 0
            or not 0 <= args.seed < 2 ** 32 or not 0 <= args.teacher_probability <= 1
            or not math.isfinite(args.teacher_probability)):
        raise ValueError('Invalid diagnostic sampling, probability, seed or device')
    if len(set(args.gradient_conditions)) != len(args.gradient_conditions) or any(
            key not in [label_for(*condition) for condition in CONDITIONS] for key in args.gradient_conditions):
        raise ValueError('Unknown or duplicate gradient condition')
    for key in ('train_root', 'reference_train_root'):
        value = getattr(args, key)
        if value is not None:
            value = value.expanduser().resolve()
            for name in ('manifest.json', 'train.jsonl', 'ckpt-best.pth'):
                if not (value / name).is_file():
                    raise FileNotFoundError(value / name)
            setattr(args, key, value)
    output = (args.output_dir.expanduser().resolve() if args.output_dir else
              args.train_root.parent / f'teacher10_diagnostic_{datetime.now():%Y%m%d_%H%M%S}_{os.getpid()}')
    archive_path = output.with_suffix('.zip')
    if archive_path.exists():
        raise FileExistsError(archive_path)
    output.mkdir(parents=True, exist_ok=False)
    os.chdir(ROOT)
    manifest = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                    arguments={key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                    output=str(output), diagnostic_source_sha256=sha256(Path(__file__)))
    write_json(output / 'run_manifest.json', manifest)
    started = time.perf_counter()
    with (output / 'run.log').open('x', encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args, output, manifest)
                manifest['elapsed_seconds'] = time.perf_counter() - started
                write_json(output / 'run_manifest.json', manifest)
            except BaseException as error:
                manifest.update(status='failed', error=f'{type(error).__name__}: {error}',
                                elapsed_seconds=time.perf_counter() - started)
                write_json(output / 'run_manifest.json', manifest)
                traceback.print_exc()
                raise
    with zipfile.ZipFile(archive_path, 'x', zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output.iterdir()):
            if path.is_file():
                archive.write(path, path.name)
    print(f'ZIP={archive_path}', flush=True)


if __name__ == '__main__':
    from utils.gpu_memory import exit_on_cuda_oom
    with exit_on_cuda_oom():
        main()
