# -*- coding: utf-8 -*-
"""第 0 阶段：刻画当前多步点云去噪教师。

本工具复用正式 test 的模型、数据集和 ``patch_based_denoise``，但不执行原 test() 中的大量
PLY 可视化。主曲线把 1/2/4/8/15/30 视为同一 30 步教师（固定 step_size/decay）的前缀；
现有 ``abl_T1.yaml`` 的 step_size=1.0 是另一条 Tweedie 单步基线，不应混入本曲线。

示例（仓库根目录、原训练环境）：

    python tools/analyze_teacher_trajectory.py \
      --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
      --ckpt experiments/.../ckpt-best.pth \
      --run_name L_10k_1pct

单样本冒烟：

    python tools/analyze_teacher_trajectory.py \
      --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
      --ckpt experiments/.../ckpt-best.pth \
      --run_name smoke --max_shapes 1 --max_steps 2 --budgets 1 2 \
      --timing_shapes 1 --timing_repeats 1

输出位于 ``experiments/stage0_teacher_analysis/<run_name>/``。真实运行需要项目原有的
PyTorch/CUDA、pointnet2、Chamfer、PyTorch3D、数据集和 checkpoint；``--help`` 不加载它们。
"""

import argparse
import csv
import datetime as _datetime
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_parser():
    parser = argparse.ArgumentParser(description='阶段 0：教师步数曲线、完整轨迹与几何动力学审计')
    parser.add_argument('--config', default='cfgs/PointGPT-L/finetune_scoredenoise.yaml')
    parser.add_argument('--ckpt', '--ckpts', dest='ckpt', required=True, help='微调教师 checkpoint')
    parser.add_argument('--run_name', default=None, help='输出实验名；缺省使用时间戳')
    parser.add_argument('--output_dir', default=None,
                        help='缺省 experiments/stage0_teacher_analysis/<run_name>')
    parser.add_argument('--device', type=int, default=0, help='CUDA device index')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=None, help='覆盖数据配置 NUM_WORKERS')

    parser.add_argument('--max_steps', type=int, default=None, help='教师完整轨迹步数，缺省读 stage0 配置')
    parser.add_argument('--budgets', nargs='+', type=int, default=None,
                        help='需要汇总质量/时间的前缀步数，如 1 2 4 8 15 30')
    parser.add_argument('--max_shapes', type=int, default=None, help='0=全部；冒烟建议 1')
    parser.add_argument('--trajectory_names', nargs='*', default=None,
                        help='指定保存 NPZ/XYZ 的 shape 名；缺省保存前 N 个')
    parser.add_argument('--save_num_trajectories', type=int, default=None)
    parser.add_argument('--timing_shapes', type=int, default=None,
                        help='前多少个 shape 做无捕获独立计时')
    parser.add_argument('--timing_repeats', type=int, default=None)
    parser.add_argument('--stats_sample_points', type=int, default=None)
    parser.add_argument('--knn_k', type=int, default=None)
    parser.add_argument('--normal_confidence_threshold', type=float, default=None)
    parser.add_argument('--consistency_atol', type=float, default=None,
                        help='同一次捕获内部硬检查容差；独立 replay 仅按此阈值提示')

    parser.add_argument('--step_size', type=float, default=None,
                        help='缺省沿用 config.langevin.step_size；主曲线应为 0.3')
    parser.add_argument('--decay', type=float, default=None,
                        help='缺省沿用 config.langevin.decay；主曲线应为 0.95')
    parser.add_argument('--noise_std', type=float, default=None,
                        help='覆盖数据集 noise_std/TEST_NOISE，仅用于专门诊断')
    parser.add_argument('--patch_batch', type=int, default=None)
    parser.add_argument('--seed_ratio', type=int, default=None)
    parser.add_argument('--fuse_tau_ratio', type=float, default=None)

    parser.add_argument('--no_sor_metrics', action='store_true',
                        help='只评 raw；默认额外评同一步数的 SOR 结果')
    parser.add_argument('--no_p2m', action='store_true', help='缺 mesh 时可关闭 P2M')
    parser.add_argument('--no_patch_trajectory', action='store_true',
                        help='NPZ 不保存精确 patch_states（仍保存诊断整云轨迹）')
    parser.add_argument('--non_corresponding_points', action='store_true',
                        help='外部 noisy/clean 点顺序不对应；禁用所有索引对应统计')
    return parser


def _cfg_get(cfg, key, default):
    if cfg is None:
        return default
    if hasattr(cfg, 'get'):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _choose(cli_value, cfg, key, default):
    return cli_value if cli_value is not None else _cfg_get(cfg, key, default)


def _plain(value):
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, 'item'):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _json_safe(value):
    value = _plain(value)
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8')


def _write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _git_info():
    def run(*args):
        try:
            return subprocess.check_output(
                ['git', *args], cwd=str(REPO_ROOT), text=True,
                stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    status = run('status', '--short')
    return {
        'commit': run('rev-parse', 'HEAD'),
        'branch': run('branch', '--show-current'),
        'dirty': bool(status),
        'status_short': status,
    }


def _safe_name(name):
    keep = []
    for char in str(name):
        keep.append(char if char.isalnum() or char in ('-', '_', '.') else '_')
    return ''.join(keep).strip('._') or 'unnamed'


def _sync_cuda(torch):
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _percentile(values, percentile, np):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(np.percentile(finite, percentile)) if finite.size else float('nan')


def _quality_summary(quality_rows, timing_rows, aggregate_numeric_rows, np, max_steps):
    summary = aggregate_numeric_rows(quality_rows, ['update_step', 'variant'])
    by_key = {(int(row['update_step']), row['variant']): row for row in summary}

    for row in summary:
        step = int(row['update_step'])
        variant = row['variant']
        relevant = [r for r in timing_rows if int(r['budget']) == step]
        if relevant:
            if variant == 'sor':
                latencies = [r['end_to_end_seconds'] for r in relevant]
            else:
                latencies = [r['denoise_seconds'] for r in relevant]
            row['latency_mean_seconds'] = float(np.mean(latencies))
            row['latency_median_seconds'] = float(np.median(latencies))
            row['latency_p95_seconds'] = _percentile(latencies, 95, np)
            row['timing_records'] = len(latencies)
            row['actual_forward_calls'] = float(np.mean(
                [r['actual_forward_calls'] for r in relevant]))
        elif step == 0:
            # raw noisy 基线没有模型成本；noisy+SOR 没有独立计时，不能伪写成零成本。
            zero_or_nan = 0.0 if variant == 'raw' else float('nan')
            row['latency_mean_seconds'] = zero_or_nan
            row['latency_median_seconds'] = zero_or_nan
            row['latency_p95_seconds'] = zero_or_nan
            row['timing_records'] = 0
            row['actual_forward_calls'] = 0.0
        row['logical_nfe'] = step

    for variant in sorted({row['variant'] for row in summary}):
        baseline = by_key.get((0, variant))
        teacher = by_key.get((max_steps, variant))
        if baseline is None or teacher is None:
            continue
        for metric in ('cd_x1e4', 'p2m_x1e4', 'hd_sq_x1e4', 'hd95_sq_x1e4'):
            m0 = baseline.get(metric, float('nan'))
            mt = teacher.get(metric, float('nan'))
            denominator = m0 - mt if math.isfinite(m0) and math.isfinite(mt) else float('nan')
            for step in sorted({int(r['update_step']) for r in summary}):
                row = by_key.get((step, variant))
                if row is None:
                    continue
                mk = row.get(metric, float('nan'))
                row[f'{metric}_relative_gap_to_teacher'] = (
                    float((mk - mt) / max(abs(mt), 1e-12))
                    if math.isfinite(mk) and math.isfinite(mt) else float('nan'))
                row[f'{metric}_retained_gain'] = (
                    float((m0 - mk) / denominator)
                    if math.isfinite(mk) and math.isfinite(denominator) and abs(denominator) > 1e-12
                    else float('nan'))
    return sorted(summary, key=lambda row: (int(row['update_step']), row['variant']))


def _run(args):
    # 重依赖全部延迟到 --help 之后加载，使无 CUDA 环境仍能查看用法。先只用
    # PyYAML 读取顶层线程限制，让 NumPy/SciPy/Open3D 导入前就能看到环境变量。
    import yaml

    launch_cwd = Path.cwd().resolve()

    def resolve_input_path(value):
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        cwd_candidate = (launch_cwd / path).resolve()
        repo_candidate = (REPO_ROOT / path).resolve()
        if cwd_candidate.exists():
            return cwd_candidate
        if repo_candidate.exists():
            return repo_candidate
        return cwd_candidate

    config_path = resolve_input_path(args.config)
    ckpt_path = resolve_input_path(args.ckpt)
    explicit_output_dir = None
    if args.output_dir:
        requested_output = Path(args.output_dir).expanduser()
        if not requested_output.is_absolute():
            requested_output = launch_cwd / requested_output
        explicit_output_dir = requested_output.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f'配置不存在: {config_path}')
    if not ckpt_path.is_file():
        raise FileNotFoundError(f'checkpoint 不存在: {ckpt_path}')
    # config 的 _base_、dataset ROOT 和默认 mesh root 都沿用仓库的相对路径约定。
    os.chdir(REPO_ROOT)

    with config_path.open('r', encoding='utf-8') as handle:
        raw_top_config = yaml.safe_load(handle) or {}
    early_cpu_threads = int(raw_top_config.get('cpu_threads', 0) or 0)
    if early_cpu_threads > 0:
        for env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[env_key] = str(early_cpu_threads)

    import numpy as np
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError('该工具需要 CUDA；请在原训练/测试环境运行')
    torch.cuda.set_device(args.device)
    device = torch.device(f'cuda:{args.device}')

    from extensions.chamfer_dist import ChamferDistanceL2
    from tools import builder
    from tools.runner_finetune import (
        normalize_unit_sphere,
        patch_based_denoise,
        sor_filter,
    )
    from utils.config import cfg_from_yaml_file
    import utils.p2m_loss as p2m_module
    from utils.trajectory_metrics import (
        aggregate_numeric_rows,
        analyze_trajectory,
        bidirectional_distance_metrics,
        validate_budgets,
    )

    config = cfg_from_yaml_file(str(config_path))
    audit_cfg = getattr(config, 'stage0_audit', None)
    langevin_cfg = getattr(config, 'langevin', None)

    config_teacher_steps = int(_cfg_get(langevin_cfg, 'num_steps', 30))
    max_steps = int(_choose(args.max_steps, audit_cfg, 'max_steps', config_teacher_steps))
    configured_audit_steps = _cfg_get(audit_cfg, 'max_steps', None)
    if (args.max_steps is None and configured_audit_steps is not None and
            int(configured_audit_steps) != config_teacher_steps):
        raise ValueError(
            'stage0_audit.max_steps 与 langevin.num_steps 不一致；'
            '请同步配置，或用 --max_steps 明确做临时诊断: '
            f'{configured_audit_steps} != {config_teacher_steps}')
    if max_steps < 1:
        raise ValueError(f'max_steps 必须 >= 1，当前为 {max_steps}')
    budgets = _choose(args.budgets, audit_cfg, 'budgets', [1, 2, 4, 8, 15, max_steps])
    budgets = validate_budgets(budgets, max_steps=max_steps)
    if max_steps not in budgets:
        raise ValueError(f'budgets 必须包含完整教师步数 max_steps={max_steps}，当前为 {budgets}')

    step_size = float(_choose(args.step_size, langevin_cfg, 'step_size', 0.3))
    decay = float(_choose(args.decay, langevin_cfg, 'decay', 0.95))
    if (not math.isfinite(step_size) or step_size <= 0 or
            not math.isfinite(decay) or decay <= 0):
        raise ValueError('step_size 和 decay 必须为有限正数')
    if abs(step_size - 1.0) < 1e-12 and max_steps > 1:
        print('[警告] step_size=1.0 的多步曲线不是当前 30×0.3 教师前缀，可能过冲。')

    max_shapes = int(_choose(args.max_shapes, audit_cfg, 'max_shapes', 0))
    save_num = int(_choose(args.save_num_trajectories, audit_cfg, 'save_num_trajectories', 3))
    timing_shapes = int(_choose(args.timing_shapes, audit_cfg, 'timing_shapes', 3))
    timing_repeats = int(_choose(args.timing_repeats, audit_cfg, 'timing_repeats', 3))
    stats_sample_points = int(_choose(
        args.stats_sample_points, audit_cfg, 'stats_sample_points', 2048))
    knn_k = int(_choose(args.knn_k, audit_cfg, 'knn_k', 16))
    normal_threshold = float(_choose(
        args.normal_confidence_threshold, audit_cfg,
        'normal_confidence_threshold', 0.05))
    consistency_atol = float(_choose(
        args.consistency_atol, audit_cfg, 'consistency_atol', 1e-5))
    seed_ratio = int(_choose(args.seed_ratio, audit_cfg, 'seed_ratio', 3))
    patch_batch = int(_choose(
        args.patch_batch, config, 'test_patch_batch', 4))
    fuse_tau_ratio = float(_choose(
        args.fuse_tau_ratio, config, 'fuse_tau_ratio', 0.5))
    patch_size = int(_cfg_get(config, 'inference_patch_size', 1024))
    save_xyz_steps = [int(v) for v in _cfg_get(
        audit_cfg, 'save_xyz_steps', [0] + budgets)]
    evaluate_sor = bool(_cfg_get(audit_cfg, 'evaluate_sor', True)) and not args.no_sor_metrics
    compute_p2m = bool(_cfg_get(audit_cfg, 'compute_p2m', True)) and not args.no_p2m
    save_patch_trajectory = (
        bool(_cfg_get(audit_cfg, 'save_patch_trajectory', True))
        and not args.no_patch_trajectory)
    assume_correspondence = (
        bool(_cfg_get(audit_cfg, 'assume_correspondence', True))
        and not args.non_corresponding_points)
    configured_sor = bool(_cfg_get(config, 'sor_enable', True))
    surface_cfg = getattr(config, 'surface_projection', None)
    configured_surface = {
        'enable': bool(_cfg_get(surface_cfg, 'enable', False)),
        'k': int(_cfg_get(surface_cfg, 'k', 16)),
        'num_iters': int(_cfg_get(surface_cfg, 'num_iters', 1)),
        'blend': float(_cfg_get(surface_cfg, 'blend', 1.0)),
    }

    if max_shapes < 0 or save_num < 0 or timing_shapes < 0 or timing_repeats < 1:
        raise ValueError('max_shapes/save_num/timing_shapes 必须 >=0，timing_repeats 必须 >=1')
    if (stats_sample_points <= 0 or knn_k <= 0 or seed_ratio <= 0 or
            patch_batch <= 0 or patch_size <= 0):
        raise ValueError('stats_sample_points/knn_k/seed_ratio/patch_batch/patch_size 必须 >0')
    if (not math.isfinite(normal_threshold) or normal_threshold < 0 or
            not math.isfinite(fuse_tau_ratio) or
            not math.isfinite(consistency_atol) or consistency_atol <= 0):
        raise ValueError('normal_confidence_threshold 必须为有限非负数；'
                         'fuse_tau_ratio 必须有限；consistency_atol 必须为有限正数')
    if args.num_workers is not None and args.num_workers < 0:
        raise ValueError('num_workers 必须 >= 0')

    run_name = args.run_name or _datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = (explicit_output_dir if explicit_output_dir is not None else
                  REPO_ROOT / 'experiments' / 'stage0_teacher_analysis' / run_name)
    output_dir = output_dir.resolve()
    if output_dir.exists():
        if not output_dir.is_dir():
            raise NotADirectoryError(f'输出路径不是目录: {output_dir}')
        if any(output_dir.iterdir()):
            raise FileExistsError(
                '输出目录已存在且非空，为避免旧轨迹混入新实验，请更换 '
                f'--run_name/--output_dir: {output_dir}')
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = output_dir / 'trajectories'

    if args.num_workers is not None:
        config.dataset.test._base_.NUM_WORKERS = int(args.num_workers)
    cpu_threads = int(_cfg_get(config, 'cpu_threads', 0) or 0)
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        for env_key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                        'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[env_key] = str(cpu_threads)
    gpu_fraction = float(_cfg_get(config, 'gpu_mem_fraction', 0) or 0)
    if gpu_fraction > 0:
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_fraction, args.device)
        except Exception as exc:
            print(f'[资源限制警告] 设置 GPU 显存比例失败，继续运行: {exc}')

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    runtime_args = argparse.Namespace(distributed=False, local_rank=args.device)
    _, test_dataloader = builder.dataset_builder(runtime_args, config.dataset.test)
    builder.inject_ablation(config)
    model = builder.model_builder(config.model)
    builder.load_model(model, str(ckpt_path))
    model = model.to(device).eval()
    cd_metric = ChamferDistanceL2().to(device)

    test_mesh_root = _cfg_get(config.dataset.test._base_, 'TEST_MESH_ROOT', None)
    if test_mesh_root:
        p2m_module._MESH_ROOT = str(test_mesh_root)

    # effective_config 同时保存原 YAML 和 CLI 最终覆盖值，便于之后精确复现。
    effective_config = _plain(config)
    effective_config['stage0_effective'] = {
        'max_steps': max_steps,
        'budgets': budgets,
        'step_size': step_size,
        'decay': decay,
        'max_shapes': max_shapes,
        'patch_size': patch_size,
        'patch_batch': patch_batch,
        'seed_ratio': seed_ratio,
        'fuse_tau_ratio': fuse_tau_ratio,
        'evaluate_sor': evaluate_sor,
        'compute_p2m': compute_p2m,
        'assume_correspondence': assume_correspondence,
        'consistency_atol': consistency_atol,
    }
    (output_dir / 'effective_config.yaml').write_text(
        yaml.safe_dump(effective_config, allow_unicode=True, sort_keys=False),
        encoding='utf-8')

    dataset_cfg = config.dataset.test._base_
    manifest = {
        'schema_version': 1,
        'status': 'running',
        'created_at': _datetime.datetime.now().isoformat(timespec='seconds'),
        'command': sys.argv,
        'launch_cwd': str(launch_cwd),
        'git': _git_info(),
        'config_path': str(config_path),
        'checkpoint': {
            'path': str(ckpt_path),
            'size_bytes': ckpt_path.stat().st_size,
            'mtime': _datetime.datetime.fromtimestamp(
                ckpt_path.stat().st_mtime).isoformat(timespec='seconds'),
        },
        'output_dir': str(output_dir),
        'dataset': {
            'num_matched_shapes': len(test_dataloader.dataset),
            'root': _cfg_get(dataset_cfg, 'ROOT', None),
            'resolution': _cfg_get(dataset_cfg, 'TEST_RESOLUTION', None),
            'noise_dir': _cfg_get(dataset_cfg, 'TEST_NOISY_DIR', None),
            'noise_path': _cfg_get(dataset_cfg, 'TEST_NOISY_PATH', None),
            'clean_path': _cfg_get(dataset_cfg, 'TEST_CLEAN_PATH', None),
            'mesh_root': test_mesh_root or p2m_module._MESH_ROOT,
            'test_noise_fallback': _cfg_get(dataset_cfg, 'TEST_NOISE', None),
        },
        'schedule': {
            'config_langevin_num_steps': config_teacher_steps,
            'max_steps': max_steps,
            'budgets': budgets,
            'step_size': step_size,
            'decay': decay,
            'interpretation': '同一教师轨迹前缀；不是 step_size=1.0 的 Tweedie-1',
        },
        'inference': {
            'patch_size': patch_size,
            'seed_ratio': seed_ratio,
            'patch_batch': patch_batch,
            'fuse_tau_ratio': fuse_tau_ratio,
            'timing_scope': 'patch构造+patch内rollout+融合；end_to_end另加SOR，不含数据读取/指标/写盘',
        },
        'analysis': {
            'seed': int(args.seed),
            'max_shapes': max_shapes,
            'save_num_trajectories': save_num,
            'trajectory_names': args.trajectory_names,
            'timing_shapes': timing_shapes,
            'timing_repeats': timing_repeats,
            'stats_sample_points': stats_sample_points,
            'knn_k': knn_k,
            'normal_confidence_threshold': normal_threshold,
            'consistency_atol': consistency_atol,
            'replay_consistency_policy': (
                '独立无捕获 replay 受 CUDA FPS/scatter 非确定性影响，只记录 max/mean/RMS '
                '并在超过 consistency_atol 时警告，不作为失败条件。'),
            'assume_correspondence': assume_correspondence,
            'evaluate_sor': evaluate_sor,
            'compute_p2m': compute_p2m,
            'save_patch_trajectory': save_patch_trajectory,
            'quality_variants': {
                'raw': '无后处理',
                'sor': ('强制执行 SOR 的审计对照；由 stage0_audit.evaluate_sor 控制，'
                        '不隐含等同于任意配置的 official test pipeline'),
            },
            'configured_test_postprocess': {
                'sor_enable': configured_sor,
                'surface_projection': configured_surface,
            },
            'trajectory_semantics': (
                'patch_states 是真实教师状态；global_states 是用固定 patch_idx/fuse_weights '
                '得到的逐步诊断 readout，未回灌到下一步。'),
        },
        'environment': {
            'python': sys.version,
            'torch': torch.__version__,
            'cuda_runtime': torch.version.cuda,
            'device': str(device),
            'gpu_name': torch.cuda.get_device_name(args.device),
        },
    }
    _write_json(output_dir / 'manifest.json', manifest)
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    print(f'[输出] {output_dir}')
    print(f'[教师前缀] max_steps={max_steps}, budgets={budgets}, '
          f'step_size={step_size}, decay={decay}')
    print(f'[几何统计] sample_points={stats_sample_points}, k={knn_k}, '
          f'correspondence={assume_correspondence}')

    quality_rows = []
    timing_rows = []
    dynamics_rows = []
    saved_count = 0
    processed_shapes = []
    requested_names = set(args.trajectory_names or [])
    def evaluate_state(state_cpu, clean_gpu, center_gpu, scale_gpu, shape_name,
                       update_step, variant):
        if variant == 'sor':
            pred_cpu = sor_filter(state_cpu)
        else:
            pred_cpu = state_cpu
        pred_gpu = pred_cpu.to(device=device, dtype=torch.float32).unsqueeze(0)
        pred_world = pred_gpu * scale_gpu + center_gpu
        clean_world = clean_gpu * scale_gpu + center_gpu
        _, metric_center, metric_scale = normalize_unit_sphere(clean_world)
        pred_metric = (pred_world - metric_center) / metric_scale
        clean_metric = (clean_world - metric_center) / metric_scale
        cd_value = float((cd_metric(pred_metric, clean_metric) * 1e4).item())
        distance_metrics = bidirectional_distance_metrics(
            pred_metric[0].detach().cpu().numpy(),
            clean_metric[0].detach().cpu().numpy())
        p2m_value = float('nan')
        if compute_p2m:
            mesh_path = Path(p2m_module._MESH_ROOT) / 'test' / f'{shape_name}.off'
            if not mesh_path.is_file():
                raise FileNotFoundError(
                    f'P2M 已启用但 mesh 不存在: {mesh_path}；若该数据集确实无 mesh，请显式加 --no_p2m')
            p2m_value = float((p2m_module.compute_p2m(
                pred_world[0], shape_name, 'test') * 1e4).item())
            if not math.isfinite(p2m_value):
                raise RuntimeError(f'P2M 返回非有限值: shape={shape_name}, step={update_step}')
        result = {
            'shape': shape_name,
            'update_step': int(update_step),
            'teacher_t': int(max_steps - update_step),
            'variant': variant,
            'num_output_points': int(pred_cpu.shape[0]),
            'cd_x1e4': cd_value,
            'p2m_x1e4': p2m_value,
        }
        result.update(distance_metrics)
        del pred_gpu, pred_world, clean_world, pred_metric, clean_metric
        return result

    try:
        with torch.no_grad():
            for shape_idx, batch in enumerate(test_dataloader):
                if max_shapes > 0 and shape_idx >= max_shapes:
                    break
                pcl_noisy, pcl_clean, noise_std, center, scale, names = batch
                shape_name = str(names[0])
                print(f'\n[{shape_idx + 1}] {shape_name}: 捕获 {max_steps} 步教师轨迹')

                clean_cpu = pcl_clean[0].float().cpu().numpy()
                noisy_gpu = pcl_noisy[0].to(device=device, dtype=torch.float32)
                clean_gpu = pcl_clean.to(device=device, dtype=torch.float32)
                center_gpu = torch.as_tensor(center[0], dtype=torch.float32, device=device)
                scale_gpu = torch.as_tensor(scale[0], dtype=torch.float32, device=device)
                if args.noise_std is not None:
                    sigma0 = float(args.noise_std)
                elif noise_std is not None:
                    sigma0 = float(noise_std.reshape(-1)[0].item())
                else:
                    sigma0 = float(_cfg_get(dataset_cfg, 'TEST_NOISE', 0.01))
                if not math.isfinite(sigma0) or sigma0 <= 0:
                    raise ValueError(f'噪声 σ 必须为有限正数，shape={shape_name}, 当前为 {sigma0}')
                noise_std_gpu = torch.tensor([sigma0], dtype=torch.float32, device=device)

                captured_final, trajectory = patch_based_denoise(
                    model, noisy_gpu, noise_std_gpu,
                    patch_size=patch_size, seed_ratio=seed_ratio, patch_batch=patch_batch,
                    num_steps=max_steps, step_size=step_size, decay=decay,
                    fuse_tau_ratio=fuse_tau_ratio, return_trajectory=True,
                    raise_on_memory_pressure=True)
                global_states = trajectory['global_states']
                noisy_cpu_tensor = noisy_gpu.detach().cpu()
                patch_initial_diff = float((
                    trajectory['patch_states'][0] -
                    noisy_cpu_tensor[trajectory['patch_idx']]).abs().max().item())
                global_initial_diff = float((
                    global_states[0] - noisy_cpu_tensor).abs().max().item())
                final_fusion_diff = float((
                    global_states[-1] - captured_final.detach().cpu()).abs().max().item())
                max_capture_diff = max(
                    patch_initial_diff, global_initial_diff, final_fusion_diff)
                if max_capture_diff > consistency_atol:
                    raise RuntimeError(
                        '轨迹捕获改变了教师结果或索引不一致: '
                        f'shape={shape_name}, max_abs_diff={max_capture_diff:.3e}, '
                        f'atol={consistency_atol:.3e}')
                coverage = trajectory['coverage_count'].numpy()
                coverage_info = {
                    'coverage_min': int(coverage.min()),
                    'coverage_mean': float(coverage.mean()),
                    'coverage_max': int(coverage.max()),
                    'uncovered_ratio': float(np.mean(coverage == 0)),
                    'patch_batches': int(trajectory['patch_batches']),
                    'patch_initial_max_abs_diff': patch_initial_diff,
                    'global_initial_max_abs_diff': global_initial_diff,
                    'final_fusion_max_abs_diff': final_fusion_diff,
                }

                # 逐步动力学统计使用 raw、未 SOR 的诊断整云；SOR 会删点，不能进入轨迹分析。
                shape_dynamics, diagnostics = analyze_trajectory(
                    global_states.numpy(), clean_cpu,
                    sigma0=sigma0, step_size=step_size, decay=decay,
                    max_steps=max_steps, sample_points=stats_sample_points,
                    knn_k=knn_k, normal_confidence_threshold=normal_threshold,
                    assume_correspondence=assume_correspondence)
                for row in shape_dynamics:
                    row['shape'] = shape_name
                    row.update(coverage_info)
                    dynamics_rows.append(row)

                # 质量曲线来自同一次 max_steps rollout 的前缀，确保 patch 布局和日程完全相同。
                for update_step in [0] + budgets:
                    state_cpu = global_states[update_step]
                    raw_row = evaluate_state(
                        state_cpu, clean_gpu, center_gpu, scale_gpu,
                        shape_name, update_step, 'raw')
                    raw_row.update(coverage_info)
                    quality_rows.append(raw_row)
                    if evaluate_sor:
                        sor_row = evaluate_state(
                            state_cpu, clean_gpu, center_gpu, scale_gpu,
                            shape_name, update_step, 'sor')
                        sor_row.update(coverage_info)
                        quality_rows.append(sor_row)
                    print(f'  step={update_step:02d} raw CD={raw_row["cd_x1e4"]:.4f} '
                          f'P2M={raw_row["p2m_x1e4"]:.4f} '
                          f'HD95={raw_row["hd95_euclidean"]:.6f}')

                # 保存代表样本。patch_states 才是后续蒸馏可直接使用的精确教师状态；
                # global_states 主要用于质量/几何曲线和可视化。
                save_this = (shape_name in requested_names or
                             (not requested_names and saved_count < save_num))
                if save_this:
                    shape_dir = trajectories_dir / _safe_name(shape_name)
                    shape_dir.mkdir(parents=True, exist_ok=True)
                    npz_payload = {
                        'global_states': global_states.numpy().astype(np.float32),
                        'patch_idx': trajectory['patch_idx'].numpy().astype(np.int64),
                        'fuse_weights': trajectory['fuse_weights'].numpy().astype(np.float32),
                        'coverage_count': coverage.astype(np.int32),
                        'seeds': trajectory['seeds'].numpy().astype(np.float32),
                        'sigma_before': trajectory['sigma_before'].numpy(),
                        'sigma_after': trajectory['sigma_after'].numpy(),
                        'clean_normalized': clean_cpu.astype(np.float32),
                        'center': center_gpu.detach().cpu().numpy(),
                        'scale': scale_gpu.detach().cpu().numpy(),
                        'stats_sample_indices': diagnostics['sample_indices'],
                        'reference_normals': diagnostics['reference_normals'].astype(np.float32),
                        'normal_confidence': diagnostics['normal_confidence'].astype(np.float32),
                    }
                    if save_patch_trajectory:
                        npz_payload['patch_states'] = trajectory['patch_states'].numpy().astype(np.float32)
                    np.savez_compressed(shape_dir / 'trajectory.npz', **npz_payload)
                    center_np = center_gpu.detach().cpu().numpy()
                    scale_np = scale_gpu.detach().cpu().numpy()
                    for update_step in save_xyz_steps:
                        if 0 <= update_step <= max_steps:
                            world = global_states[update_step].numpy() * scale_np + center_np
                            np.savetxt(
                                shape_dir / (f'update_step_{update_step:03d}_'
                                             f'teacher_t_{max_steps-update_step:03d}.xyz'),
                                world, fmt='%.8f')
                    _write_json(shape_dir / 'metadata.json', {
                        'shape': shape_name,
                        'sigma0': sigma0,
                        'max_steps': max_steps,
                        'step_size': step_size,
                        'decay': decay,
                        'coverage': coverage_info,
                        'patch_states_saved': save_patch_trajectory,
                        'global_states_are_diagnostic_readout': True,
                    })
                    saved_count += 1

                # 独立无捕获计时：不含轨迹 CPU copy、指标和写盘。质量轨迹已先跑过，兼作 warmup。
                if shape_idx < timing_shapes:
                    for budget in budgets:
                        for repeat in range(timing_repeats):
                            _sync_cuda(torch)
                            start = time.perf_counter()
                            timed_pred = patch_based_denoise(
                                model, noisy_gpu, noise_std_gpu,
                                patch_size=patch_size, seed_ratio=seed_ratio,
                                patch_batch=patch_batch, num_steps=budget,
                                step_size=step_size, decay=decay,
                                fuse_tau_ratio=fuse_tau_ratio,
                                return_trajectory=False,
                                raise_on_memory_pressure=True)
                            _sync_cuda(torch)
                            denoise_seconds = time.perf_counter() - start
                            # fresh replay 会重新执行 CUDA FPS/scatter/index_add，多步后微小的
                            # 非确定性舍入会累积，因此它不是“轨迹捕获正确性”的硬不变量。
                            # 同一次捕获内部的 patch/global/final 融合检查仍在上面严格执行；
                            # 这里仅记录差异，非有限值才中止实验。
                            timed_cpu = timed_pred.detach().cpu()
                            prefix_delta = timed_cpu - global_states[budget]
                            prefix_abs = prefix_delta.abs()
                            prefix_capture_diff = float(prefix_abs.max().item())
                            prefix_capture_mean_diff = float(prefix_abs.mean().item())
                            prefix_capture_rms_diff = float(
                                torch.sqrt(torch.mean(prefix_delta ** 2)).item())
                            if not all(math.isfinite(value) for value in (
                                    prefix_capture_diff,
                                    prefix_capture_mean_diff,
                                    prefix_capture_rms_diff)):
                                raise RuntimeError(
                                    '无捕获 replay 出现非有限差异: '
                                    f'shape={shape_name}, budget={budget}')
                            if prefix_capture_diff > consistency_atol and repeat == 0:
                                print(
                                    '  [replay 数值提示] 独立 CUDA 重跑与捕获前缀存在微小差异：'
                                    f'budget={budget}, max={prefix_capture_diff:.3e}, '
                                    f'rms={prefix_capture_rms_diff:.3e}；已记录，不中断实验')
                            post_seconds = 0.0
                            if evaluate_sor:
                                post_start = time.perf_counter()
                                _ = sor_filter(timed_pred)
                                post_seconds = time.perf_counter() - post_start
                            num_seeds = max(1, int(seed_ratio * len(noisy_gpu) / min(patch_size, len(noisy_gpu))))
                            patch_batches = int(math.ceil(num_seeds / patch_batch))
                            timing_rows.append({
                                'shape': shape_name,
                                'budget': int(budget),
                                'repeat': int(repeat),
                                'logical_nfe': int(budget),
                                'num_seeds': int(num_seeds),
                                'patch_batches': patch_batches,
                                'actual_forward_calls': int(budget * patch_batches),
                                'prefix_capture_max_abs_diff': prefix_capture_diff,
                                'prefix_capture_mean_abs_diff': prefix_capture_mean_diff,
                                'prefix_capture_rms_diff': prefix_capture_rms_diff,
                                'denoise_seconds': float(denoise_seconds),
                                'sor_seconds': float(post_seconds),
                                'end_to_end_seconds': float(denoise_seconds + post_seconds),
                            })
                            del timed_pred, timed_cpu, prefix_delta, prefix_abs

                # 每个 shape 后落盘，内存保护或人工中断时仍保留已完成结果。
                processed_shapes.append(shape_name)
                _write_csv(output_dir / 'per_shape_by_budget.csv', quality_rows)
                _write_csv(output_dir / 'per_step_dynamics.csv', dynamics_rows)
                _write_csv(output_dir / 'timing_runs.csv', timing_rows)
                _write_json(output_dir / 'progress.json', {
                    'processed_shapes': processed_shapes,
                    'saved_trajectories': saved_count,
                    'last_shape': shape_name,
                })

                del captured_final, trajectory, global_states, diagnostics
                del noisy_gpu, clean_gpu, center_gpu, scale_gpu, noise_std_gpu
                torch.cuda.empty_cache()

        missing_requested = sorted(requested_names.difference(processed_shapes))
        if missing_requested:
            raise ValueError(
                '以下 --trajectory_names 未在本次处理范围内找到；请检查名称或 max_shapes: '
                f'{missing_requested}')

        aggregate_dynamics = aggregate_numeric_rows(
            dynamics_rows, ['update_step', 'teacher_t'])
        summary = _quality_summary(
            quality_rows, timing_rows, aggregate_numeric_rows, np, max_steps)
        _write_csv(output_dir / 'aggregate_per_step.csv', aggregate_dynamics)
        _write_csv(output_dir / 'summary_by_budget.csv', summary)

        manifest['status'] = 'complete'
        manifest['completed_at'] = _datetime.datetime.now().isoformat(timespec='seconds')
        manifest['result'] = {
            'processed_shapes': len(processed_shapes),
            'shape_names': processed_shapes,
            'saved_trajectories': saved_count,
            'quality_rows': len(quality_rows),
            'timing_rows': len(timing_rows),
            'dynamics_rows': len(dynamics_rows),
        }
        _write_json(output_dir / 'manifest.json', manifest)
        print(f'\n[完成] 共分析 {len(processed_shapes)} 个 shape；汇总: '
              f'{output_dir / "summary_by_budget.csv"}')
    except BaseException as exc:
        manifest['status'] = 'failed'
        manifest['failed_at'] = _datetime.datetime.now().isoformat(timespec='seconds')
        manifest['error'] = repr(exc)
        manifest['processed_shapes'] = processed_shapes
        _write_json(output_dir / 'manifest.json', manifest)
        _write_csv(output_dir / 'per_shape_by_budget.csv', quality_rows)
        _write_csv(output_dir / 'per_step_dynamics.csv', dynamics_rows)
        _write_csv(output_dir / 'timing_runs.csv', timing_rows)
        raise


def main():
    parser = _build_parser()
    args = parser.parse_args()
    _run(args)


if __name__ == '__main__':
    main()
