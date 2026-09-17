"""Read-only initial-Student diagnostic on the formal 64-patch curriculum bank."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ('models', 'tools', 'datasets', 'utils', 'cfgs')
            for p in sorted((ROOT / folder).rglob('*')) if p.suffix in ('.py', '.yaml')}


def summarize_paths(interval_means, target, balance):
    from tools.shadow_global_search import TEACHER_PATHS, search_global_teacher_nodes, score_teacher_path
    from tools import runner_distill as runner
    global_result = search_global_teacher_nodes(interval_means, target, balance)
    paths = [score_teacher_path(nodes, interval_means, target, balance) for nodes in TEACHER_PATHS]
    min_error = min(paths, key=lambda row: (row['mean_target_error'], row['nodes']))
    min_std = min(paths, key=lambda row: (row['pcd_std'], row['nodes']))
    joint = min(paths, key=lambda row: (row['path_score'], row['nodes']))
    assert joint['nodes'] == global_result['nodes'] and joint['path_score'] == global_result['path_score']
    within_band = [row['nodes'] for row in paths if all(.25 <= pcd <= .35 for pcd in row['stage_PCD'])]
    values = [row['PCD'] for row in interval_means.values()]
    return dict(
        comparisons=dict(fixed=score_teacher_path(runner.TEACHER_NODES, interval_means, target, balance),
                         nonuniform=score_teacher_path(runner.NONUNIFORM_NODES, interval_means, target, balance),
                         global_search=global_result),
        interval_pcd_statistics=dict(min=min(values), median=statistics.median(values), max=max(values),
                                     intervals_in_target_band=sum(.25 <= v <= .35 for v in values)),
        optimum_paths=dict(min_target_error=min_error, min_std=min_std, min_joint_score=joint),
        target_band_check=dict(bounds=[.25, .35], matching_path_count=len(within_band),
                               matching_paths=within_band, reachable=bool(within_band)),
        optimum_comparison=dict(
            all_three_same_nodes=min_error['nodes'] == min_std['nodes'] == joint['nodes'],
            min_std_equals_joint=min_std['nodes'] == joint['nodes'],
            internal_node_absolute_differences=[abs(a-b) for a, b in zip(min_std['nodes'][1:4], joint['nodes'][1:4])],
            joint_minus_min_std_std=joint['pcd_std'] - min_std['pcd_std'],
            min_std_minus_joint_target_error=min_std['mean_target_error'] - joint['mean_target_error'],
            min_std_minus_joint_score=min_std['path_score'] - joint['path_score']),
        paths=paths)


def report_markdown(result):
    if not result.get('passed'):
        return '# 64 patches calibration diagnostic\n\nFailed: ' + result.get('error', 'unknown') + '\n'
    protocol = result['protocol']
    lines = ['# 64 patches initial-Student calibration diagnostic', '',
             'Student 由 Teacher checkpoint 直接初始化，并使用正式 Step condition；未经过蒸馏。',
             '整个诊断仅 eval/no_grad，没有训练、backward 或 optimizer。', '',
             f"- Git commit: `{result['source']['git_commit']}`",
             f"- Branch: `{result['source']['git_branch']}`",
             f"- Working tree dirty: `{bool(result['source']['git_status'])}`（实际源码 SHA-256 保存在 result.json）",
             f"- Teacher checkpoint: `{result['checkpoint']['path']}`",
             f"- Checkpoint SHA-256: `{result['checkpoint']['sha256']}`",
             f"- Bank fingerprint: `{result['calibration_metadata']['data_sha256']}`",
             f"- Patches / seed / batch: {protocol['patches']} / {protocol['calibration_seed']} / {protocol['batch_size']}",
             f"- Intervals / paths / Student forwards: {protocol['intervals']} / {protocol['paths']} / {result['student_forward_calls']}",
             '- Aggregation: 每个 interval 先计算各 patch 的 PCD，再平均；路径 std 是四阶段均值的 population std。',
             '- Objective: mean(abs(PCD - 0.3)) + 1.0 * population_std(PCD).', '',
             '| Schedule | Nodes | 四阶段 mean PCD | std | range | mean target error | score |',
             '|---|---|---|---:|---:|---:|---:|']
    entries = list(result['comparisons'].items()) + [
        ('min_std', result['optimum_paths']['min_std']),
        ('min_target_error', result['optimum_paths']['min_target_error'])]
    for name, row in entries:
        lines.append('| {} | {} | {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |'.format(
            name, row['nodes'], ', '.join(f'{v:.6f}' for v in row['stage_PCD']),
            row['pcd_std'], row['pcd_max_minus_min'], row['mean_target_error'], row['path_score']))
    stats, band, comparison = result['interval_pcd_statistics'], result['target_band_check'], result['optimum_comparison']
    lines += ['', f"130 个 interval mean PCD：min={stats['min']:.6f}, median={stats['median']:.6f}, max={stats['max']:.6f}。",
              f"其中 {stats['intervals_in_target_band']} 个 interval 落在 [0.25,0.35]；455 条路径中有 {band['matching_path_count']} 条的四个 stage 全部落在该区间。",
              '可达性结论仅限该初始 Student、固定 calibration bank 和现有候选路径，不代表后续训练状态。', '',
              f"三种最优路径是否一致：{comparison['all_three_same_nodes']}。",
              f"最小 std 路径与联合目标路径是否一致：{comparison['min_std_equals_joint']}。",
              f"内部节点绝对差：{comparison['internal_node_absolute_differences']}；联合路径 std 高出 {comparison['joint_minus_min_std_std']:.6f}；最小 std 路径 target error 高出 {comparison['min_std_minus_joint_target_error']:.6f}。", '',
              f"完整运行时间（含加载、Teacher bank、搜索、参数完整性核对）：{result['elapsed_seconds']:.3f} 秒。",
              f"Peak GPU allocated / reserved：{result['peak_gpu_allocated_gib']:.3f} / {result['peak_gpu_reserved_gib']:.3f} GiB。", '',
              '完整性检查：', '```json', json.dumps(result['integrity'], indent=2, ensure_ascii=False), '```', '',
              '没有修改 target、正式训练 schedule 或算法，没有启动正式训练。', '']
    return '\n'.join(lines)


def run(args, result):
    started = time.monotonic()
    import numpy as np
    import torch
    from tools import builder, runner_distill as runner
    from tools.shadow_global_search import TEACHER_INTERVALS, TEACHER_PATHS, build_interval_pcd_cache
    from utils.config import cfg_from_yaml_file

    def emit(key, value):
        result[key] = value
        print(json.dumps({key: value}, ensure_ascii=False, allow_nan=False), flush=True)

    hashes_before = source_hashes()
    if args.source_manifest:
        provenance = json.loads(Path(args.source_manifest).read_text(encoding='utf-8'))
        assert provenance['source_sha256'] == hashes_before, 'Uploaded code differs from the local source manifest'
    else:
        def git(*arguments):
            return subprocess.check_output(['git', *arguments], cwd=ROOT, encoding='utf-8').strip()
        provenance = dict(git_commit=git('rev-parse', 'HEAD'), git_branch=git('branch', '--show-current'),
                          git_status=git('status', '--short'), source_sha256=hashes_before)
    emit('source', provenance)
    config = cfg_from_yaml_file(args.config)
    config_before = copy.deepcopy(config)
    options = runner.dynamic_pcd_options(config)
    assert config.curriculum_mode == 'dynamic_pcd'
    assert (options['calibration_patches'], options['calibration_seed'], options['interval_patch_batch']) == (64, 2025, 8)
    assert (options['target'], options['lambda_balance']) == (.3, 1.)
    assert len(TEACHER_INTERVALS) == 130 and len(TEACHER_PATHS) == 455
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.device)
    device = torch.device('cuda', args.device)
    torch.set_num_threads(int(config.cpu_threads))
    torch.cuda.set_per_process_memory_fraction(float(config.gpu_mem_fraction), device)
    torch.cuda.reset_peak_memory_stats(device)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    emit('protocol', dict(patches=64, calibration_seed=2025, batch_size=8, initialization_seed=args.seed,
                          intervals=130, paths=455, target=.3, lambda_balance=1., pcd_eps=runner.PCD_EPS,
                          initial_nodes=list(runner.configured_teacher_nodes(config)),
                          aggregation='mean_of_per_patch_ratios', formal_training=False,
                          gpu=torch.cuda.get_device_name(device), torch=str(torch.__version__),
                          cuda=torch.version.cuda, visible_devices=os.getenv('CUDA_VISIBLE_DEVICES')))
    checkpoint = Path(args.teacher_ckpt).resolve()
    checkpoint_stat = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
    digest = hashlib.sha256()
    with checkpoint.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    emit('checkpoint', dict(path=str(checkpoint), sha256=digest.hexdigest(), size=checkpoint_stat[0]))
    with torch.no_grad():
        # Same initialization sequence as runner.train(), stopping before optimizer creation.
        teacher = builder.model_builder(config.model).to(device)
        builder.load_model(teacher, str(checkpoint))
        student = copy.deepcopy(teacher).to(device)
        runner.enable_student_condition(student)
        student.requires_grad_(True)
        runner.freeze_teacher(teacher)
        student.eval()
        assert teacher.step_condition is None and student.step_condition is not None
        assert all(torch.equal(value, student.state_dict()[name]) for name, value in teacher.state_dict().items())
        snapshots = [{name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                     for model in (teacher, student)]
        counts = dict(teacher=0, student=0)
        def check_forward(model, inputs, kwargs):
            assert not torch.is_grad_enabled() and not model.training
            counts['teacher' if model is teacher else 'student'] += 1
        handles = [model.register_forward_pre_hook(check_forward, with_kwargs=True) for model in (teacher, student)]
        emit('phase', 'calibration_teacher_bank')
        phase_start = time.monotonic()
        loader = runner._train_loader(config)
        bank, metadata = runner.curriculum_calibration_bank(config, teacher, loader.dataset)
        count = sum(states.shape[1] for states, _ in bank)
        assert count == 64 and all(states.shape[0] == 17 for states, _ in bank)
        assert counts['teacher'] == 64 * 16
        emit('calibration_metadata', metadata)
        emit('bank_seconds', time.monotonic() - phase_start)
        phase_start = time.monotonic()
        totals, per_patch, forward_calls = {}, {}, 0
        for index, (states, sigmas) in enumerate(bank):
            cache = build_interval_pcd_cache(student, states, sigmas, options['interval_patch_batch'])
            assert set(cache['metrics']) == set(TEACHER_INTERVALS)
            for edge, row in cache['metrics'].items():
                totals.setdefault(edge, dict.fromkeys(('D_move', 'E_imit', 'PCD'), 0.))
                per_patch.setdefault(edge, [])
                for key, values in row.items():
                    assert not values.requires_grad and torch.isfinite(values).all()
                    # Same per-patch-ratio aggregation and reduction order as the training update.
                    totals[edge][key] += float(values.double().sum())
                per_patch[edge].extend(row['PCD'].tolist())
            forward_calls += cache['forward_calls']
            del cache
            print(json.dumps(dict(calibration_batches_done=index + 1, total_batches=len(bank),
                                  student_forward_calls=forward_calls)), flush=True)
        interval_means = {edge: {key: value / count for key, value in row.items()} for edge, row in totals.items()}
        result.update(summarize_paths(interval_means, options['target'], options['lambda_balance']))
        result['intervals'] = [dict(start=t, target=u, **interval_means[t, u], per_patch_PCD=per_patch[t, u])
                               for t, u in TEACHER_INTERVALS]
        emit('search_seconds', time.monotonic() - phase_start)
        emit('student_forward_calls', forward_calls)
        assert forward_calls == counts['student'] == 1040
        for handle in handles:
            handle.remove()
        for model, snapshot in zip((teacher, student), snapshots):
            assert all(not module.training for module in model.modules())
            assert all(p.grad is None for p in model.parameters())
            for name, value in model.state_dict().items():
                assert torch.isfinite(value).all(), name
                assert torch.equal(value.cpu(), snapshot[name]), name
        assert all(not p.requires_grad for p in teacher.parameters())
        assert checkpoint_stat == (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
        assert config == config_before and source_hashes() == hashes_before
        json.dumps(result, allow_nan=False)
        torch.cuda.synchronize(device)
    emit('integrity', dict(patches_64=True, intervals_130=True, paths_455=True,
                           student_parameters_and_buffers_bitwise_unchanged=True,
                           teacher_parameters_and_buffers_bitwise_unchanged=True, teacher_frozen=True,
                           all_forwards_eval_no_grad=True, all_parameter_gradients_none=True,
                           all_finite=True, checkpoint_file_unchanged=True,
                           formal_config_and_source_unchanged=True,
                           backward_calls=0, optimizer_steps=0, training_started=False))
    emit('peak_gpu_allocated_gib', torch.cuda.max_memory_allocated(device) / 1024 ** 3)
    emit('peak_gpu_reserved_gib', torch.cuda.max_memory_reserved(device) / 1024 ** 3)
    emit('elapsed_seconds', time.monotonic() - started)
    result['passed'] = True
    for key in ('comparisons', 'optimum_paths', 'interval_pcd_statistics', 'target_band_check', 'optimum_comparison'):
        print(json.dumps({key: result[key]}, allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='cfgs/PointGPT-L/distill_16to4_dynamic.yaml')
    parser.add_argument('--teacher_ckpt', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--source_manifest', help='Local git/source provenance when running an uploaded snapshot')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('Use a new empty diagnostic output directory')
    result = dict(passed=False)
    try:
        run(args, result)
    except Exception as exc:
        result.update(error=str(exc), traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (output / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
        (output / 'report.md').write_text(report_markdown(result), encoding='utf-8')
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    sys.exit(main())
