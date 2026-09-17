"""Two real CUDA mini epochs through runner.train; never start formal training."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time
import traceback
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args, result):
    import numpy as np
    import torch
    import yaml
    from datasets.ScoreDenoiseDataset import ScoreDenoise
    from tools import builder, runner_distill as runner
    from utils.config import cfg_from_yaml_file

    started = time.monotonic()
    output = Path(args.output_dir)
    config = cfg_from_yaml_file(args.config)
    assert config.curriculum_mode == 'dynamic_pcd'
    assert config.model.depth == 24 and config.model.trans_dim == 1024
    assert config.total_bs == config.student_patch_batch == 8 and config.teacher_patch_batch == 1
    assert config.dynamic_pcd.update_every_epochs == 1
    config.dynamic_pcd.calibration_patches = args.calibration_patches
    config.validation_patches_per_cloud = 1
    config.rollout_val_interval = 1
    config.epochs = 2
    # Data limits only: original paired training Dataset and full one-cloud val pipeline.
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    torch.set_num_threads(int(config.cpu_threads))
    torch.cuda.set_per_process_memory_fraction(float(config.gpu_mem_fraction), device)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    teacher_path = Path(args.teacher_ckpt).resolve()
    checkpoint_stat = (teacher_path.stat().st_size, teacher_path.stat().st_mtime_ns)
    models, teacher_versions, teacher_buffers = {}, {}, {}
    gradient_checks, stage_calls, search_checks, checkpoint_checks = [], [], [], []

    def emit(key, value):
        result[key] = value
        print(json.dumps({key: value}, allow_nan=False), flush=True)

    emit('protocol', dict(gpu=torch.cuda.get_device_name(device), torch=str(torch.__version__),
                          mini_epochs=2, training_batches_per_epoch=1, train_batch=8,
                          calibration_patches=args.calibration_patches, full_validation_clouds=1,
                          initial_nodes=list(runner.configured_teacher_nodes(config)), formal_training=False))
    emit('source_sha256', {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for folder in ('models', 'tools', 'datasets', 'utils', 'cfgs')
                           for p in sorted((ROOT / folder).rglob('*')) if p.suffix in ('.py', '.yaml')})
    # JSON round-trip produces ordinary dicts for the human-readable effective YAML.
    (output / 'effective_config.yaml').write_text(yaml.safe_dump(json.loads(json.dumps(config))), encoding='utf-8')

    load_model = builder.load_model
    def load_teacher(model, path):
        load_model(model, path)
        models['teacher'] = model
        teacher_versions.update({name: p._version for name, p in model.named_parameters()})
        teacher_buffers.update({name: b.detach().cpu().clone() for name, b in model.named_buffers()})

    enable_condition = runner.enable_student_condition
    def enable_student(model):
        enable_condition(model)
        models['student'] = model
        for name, value in models['teacher'].state_dict().items():
            assert torch.equal(value, model.state_dict()[name]), name
        return model

    def norm(parameters):
        grads = [p.grad for p in parameters if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        value = float(torch.stack([g.float().norm() for g in grads]).norm())
        assert math.isfinite(value) and value > 0
        return value

    check_gradients = runner.check_gradients
    def check(teacher, student):
        answer = check_gradients(teacher, student)
        gradient_checks.append(dict(condition=norm(student.step_condition.parameters()),
                                    encoder=norm(student.encoder.parameters()),
                                    transformer=norm(student.blocks.parameters()),
                                    decoder=norm(student.generator_blocks.parameters())))
        return answer

    backward = runner.backward_stages
    def checked_backward(student, nodes, sigma0, **kwargs):
        used = tuple(kwargs['teacher_nodes'])
        calls, encodings = [], []
        def hook(model, inputs, call_kwargs):
            k = len(calls)
            t, u = used[k:k+2]
            assert torch.is_grad_enabled() and model.training
            assert (call_kwargs['start_step'], call_kwargs['target_step']) == (t, u)
            torch.testing.assert_close(inputs[0].cpu(), nodes[k], rtol=0, atol=0)
            torch.testing.assert_close(call_kwargs['noise_std'], sigma0.to(device) * .95 ** t, rtol=0, atol=0)
            calls.append([t, u])
        a = student.register_forward_pre_hook(hook, with_kwargs=True)
        b = student.step_condition.mlp[0].register_forward_pre_hook(
            lambda _, inputs: encodings.append(inputs[0].detach().cpu().clone()))
        try:
            losses = backward(student, nodes, sigma0, **kwargs)
        finally:
            a.remove(); b.remove()
        assert len(calls) == 4 and all(math.isfinite(value) for value in losses)
        for encoded, (t, u) in zip(encodings, calls):
            torch.testing.assert_close(encoded, torch.tensor([t, u, u-t]).float().expand_as(encoded) / 16, rtol=0, atol=0)
        stage_calls.append(dict(nodes=list(used), calls=calls, losses=losses))
        return losses

    update = runner.update_dynamic_curriculum
    def checked_update(student, bank, cfg):
        versions = {name: p._version for name, p in student.named_parameters()}
        buffers = {name: b.detach().cpu().clone() for name, b in student.named_buffers()}
        modes = [m.training for m in student.modules()]
        grad_versions = [p.grad._version if p.grad is not None else None for p in student.parameters()]
        calls = []
        def hook(model, inputs, kwargs):
            assert not torch.is_grad_enabled() and not model.training
            assert kwargs['target_step'] > kwargs['start_step']
            calls.append(1)
        handle = student.register_forward_pre_hook(hook, with_kwargs=True)
        try:
            selected = update(student, bank, cfg)
        finally:
            handle.remove()
        assert selected['paths_evaluated'] == 455 and selected['unique_intervals'] == 130
        assert selected['student_forward_calls'] == len(calls)
        assert all(p._version == versions[name] for name, p in student.named_parameters())
        assert all(torch.equal(b.cpu(), buffers[name]) for name, b in student.named_buffers())
        assert modes == [m.training for m in student.modules()]
        assert grad_versions == [p.grad._version if p.grad is not None else None for p in student.parameters()]
        search_checks.append(selected)
        emit('latest_search', selected)
        return selected

    save = runner.save_epoch_checkpoints
    def checked_save(*call_args, **kwargs):
        answer = save(*call_args, **kwargs)
        state = kwargs['curriculum_state']
        last = torch.load(output / 'train' / 'ckpt-last.pth', map_location='cpu')
        assert last['distillation'] == runner.schedule(state['nodes_used_this_epoch'])
        assert last['current_teacher_nodes'] == state['nodes_used_this_epoch']
        restored, history, metadata = runner.restore_dynamic_curriculum(last, config)
        assert list(restored) == state['next_teacher_nodes']
        assert history == state['curriculum_history'] and metadata == state['calibration_metadata']
        checkpoint_checks.append(dict(epoch=last['epoch'], used=last['nodes_used_this_epoch'],
                                      next=last['next_teacher_nodes'], best_epoch=last['best_epoch']))
        return answer

    val_loader = ScoreDenoise.val_dataloader
    def one_validation_cloud(module):
        _, loader = val_loader(module)
        subset = torch.utils.data.Subset(loader.dataset, [0])
        return subset, torch.utils.data.DataLoader(subset, batch_size=1, shuffle=False, num_workers=0,
                                                   collate_fn=loader.collate_fn)

    train_output = output / 'train'
    train_output.mkdir()
    with patch.object(builder, 'load_model', side_effect=load_teacher), \
            patch.object(runner, 'enable_student_condition', side_effect=enable_student), \
            patch.object(runner, 'check_gradients', side_effect=check), \
            patch.object(runner, 'backward_stages', side_effect=checked_backward), \
            patch.object(runner, 'update_dynamic_curriculum', side_effect=checked_update), \
            patch.object(runner, 'save_epoch_checkpoints', side_effect=checked_save), \
            patch.object(ScoreDenoise, 'val_dataloader', one_validation_cloud):
        runner.train(argparse.Namespace(epochs=2, max_shapes=0, max_patch_batches=1, resume=None),
                     config, builder, device, teacher_path, train_output)
    torch.cuda.synchronize(device)
    rows = [json.loads(line) for line in (train_output / 'train.jsonl').read_text().splitlines()]
    assert len(rows) == len(gradient_checks) == len(stage_calls) == len(search_checks) == 2
    assert rows[0]['teacher_nodes_used'] == list(runner.configured_teacher_nodes(config))
    assert rows[1]['teacher_nodes_used'] == rows[0]['next_teacher_nodes']
    teacher, student = models['teacher'], models['student']
    assert all(not m.training for m in teacher.modules())
    assert all(not p.requires_grad and p.grad is None and p._version == teacher_versions[name]
               for name, p in teacher.named_parameters())
    assert all(torch.equal(b.cpu(), teacher_buffers[name]) for name, b in teacher.named_buffers())
    assert all(torch.isfinite(p).all() for p in student.parameters())
    assert (teacher_path.stat().st_size, teacher_path.stat().st_mtime_ns) == checkpoint_stat
    best_path = train_output / 'ckpt-best.pth'
    best = torch.load(best_path, map_location='cpu')
    best_nodes = best['nodes_used_this_epoch']
    assert best_nodes == rows[best['epoch'] - 1]['teacher_nodes_used']
    del best
    runner.load_student_checkpoint(student, best_path, builder)
    assert list(student.distillation_teacher_nodes) == best_nodes
    emit('epochs', rows)
    emit('gradient_norms', gradient_checks)
    emit('training_intervals', stage_calls)
    emit('checkpoint_checks', checkpoint_checks)
    emit('best_checkpoint_nodes', best_nodes)
    emit('integrity', dict(teacher_frozen_eval_unchanged=True, teacher_checkpoint_unchanged=True,
                           search_no_grad_parameters_buffers_gradients_unchanged=True,
                           current_nodes_condition_sigma_targets_checked=True,
                           next_nodes_used_by_epoch2=True, strict_best_reload=True,
                           resume_curriculum_state_checked=True, all_student_parameters_finite=True))
    emit('peak_gpu_allocated_gib', torch.cuda.max_memory_allocated(device) / 1024 ** 3)
    emit('elapsed_seconds', time.monotonic() - started)
    result['passed'] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/distill_16to4_dynamic.yaml')
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--calibration_patches', type=int, default=8, help='Smoke-only override; formal YAML uses 64')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    if args.calibration_patches < 1:
        parser.error('calibration_patches must be positive')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a new empty smoke output directory')
    result = dict(passed=False)
    try:
        run(args, result)
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (output / 'smoke_result.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
