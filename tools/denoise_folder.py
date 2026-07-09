# -*- coding: utf-8 -*-
"""纯去噪脚本：对一个文件夹里的噪声 .xyz 逐个去噪并保存，不算 CD/P2M。

用于真实世界噪声（如 RueMadame 街景扫描）—— 没有干净点云、没有 mesh，无法评测，
只需要输出去噪结果。推理完全复用 test() 的 patch_based_denoise + (可选)SOR，
保证和正式测试同一套流程。

用法（仓库根目录、训练环境下跑）：
    python tools/denoise_folder.py \
        --config cfgs/PointGPT-L/finetune_scoredenoise.yaml \
        --ckpt   experiments/finetune_scoredenoise/done-best/L_consistency_plus/ckpt-best.pth \
        --input_dir  data/ScoreDenoise/examples/RueMadame \
        --noise_std 0.02

- 默认对 input_dir 下所有 .xyz 去噪，**自动跳过文件名含 "denoised" 的**（那是已去噪的输出，
  如 RueMadame_3_denoised.txt.xyz）；额外可用 --skip 指定要跳过的文件名。
- 输出默认存到 <input_dir>/denoised/<原文件名>，不覆盖输入。
- Langevin / test_patch_batch / fuse_tau_ratio / sor_enable 默认从 config 读，可命令行覆盖。

关于 --noise_std（重要）：真实噪声没有已知 σ，这个值是 Langevin 的起始噪声级、直接决定
去噪强度。模型在 σ∈[0.005,0.02]（单位球归一化空间）训练。把整片场景归一化到单位球后
真实 LiDAR 噪声的等效 σ 不确定，建议在 0.01~0.03 之间试几个值、按去噪后 .xyz 的观感挑。
"""
import os
import sys
import argparse

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import cfg_from_yaml_file                    # noqa: E402
from tools import builder                                      # noqa: E402
from tools.runner_finetune import patch_based_denoise, sor_filter  # noqa: E402
from datasets.scoredenoise.transforms import NormalizeUnitSphere  # noqa: E402


def load_points(path):
    """读 .xyz（可能是 .txt.xyz、可能多列），只取前 3 列 → [N,3] float tensor。"""
    arr = np.loadtxt(path, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return torch.from_numpy(arr[:, :3]).float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='cfgs/PointGPT-L/finetune_scoredenoise.yaml')
    ap.add_argument('--ckpt', required=True, help='微调好的 ckpt 路径')
    ap.add_argument('--input_dir', required=True, help='噪声 .xyz 所在文件夹')
    ap.add_argument('--output_dir', default=None, help='默认 <input_dir>/denoised')
    ap.add_argument('--noise_std', type=float, default=0.02, help='Langevin 起始 σ（去噪强度，需调）')
    ap.add_argument('--skip', nargs='*', default=[], help='额外要跳过的文件名（如 RueMadame_2.txt.xyz）')
    ap.add_argument('--sor', choices=['config', 'on', 'off'], default='config',
                    help='SOR 后处理：config=按 yaml 的 sor_enable；on/off 强制。真实噪声一般 on 更好')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = cfg_from_yaml_file(args.config)

    # 推理超参（与 test() 同款，从 config 读）
    lv = getattr(config, 'langevin', None)
    lv_steps = int(getattr(lv, 'num_steps', 1)) if lv is not None else 1
    lv_step_size = float(getattr(lv, 'step_size', 1.0)) if lv is not None else 1.0
    lv_decay = float(getattr(lv, 'decay', 0.95)) if lv is not None else 0.95
    test_patch_batch = int(getattr(config, 'test_patch_batch', 4))
    fuse_tau_ratio = float(getattr(config, 'fuse_tau_ratio', 0.5))
    if args.sor == 'config':
        sor_enable = bool(getattr(config, 'sor_enable', True))
    else:
        sor_enable = (args.sor == 'on')

    print(f'[Config] Langevin: steps={lv_steps} step_size={lv_step_size} decay={lv_decay}')
    print(f'[Config] test_patch_batch={test_patch_batch} fuse_tau_ratio={fuse_tau_ratio} '
          f'SOR={"on" if sor_enable else "off"}  noise_std(σ0)={args.noise_std}')

    # 建模 + 载权重
    model = builder.model_builder(config.model)
    builder.load_model(model, args.ckpt)
    model = model.to(device).eval()

    out_dir = args.output_dir or os.path.join(args.input_dir, 'denoised')
    os.makedirs(out_dir, exist_ok=True)

    skip_set = set(args.skip)
    files = [f for f in sorted(os.listdir(args.input_dir)) if f.endswith('.xyz')]
    for fn in files:
        if 'denoised' in fn:
            print(f'[跳过] {fn}（文件名含 denoised，视为已去噪输出）')
            continue
        if fn in skip_set:
            print(f'[跳过] {fn}（--skip 指定）')
            continue

        path = os.path.join(args.input_dir, fn)
        pcl = load_points(path)                                # [N,3] world
        N = pcl.shape[0]
        pcl_norm, center, scale = NormalizeUnitSphere.normalize(pcl)   # 单位球
        pcl_norm = pcl_norm.to(device)

        with torch.no_grad():
            denoised = patch_based_denoise(
                model, pcl_norm, float(args.noise_std),
                patch_batch=test_patch_batch,
                num_steps=lv_steps, step_size=lv_step_size, decay=lv_decay,
                fuse_tau_ratio=fuse_tau_ratio,
            )                                                  # [N,3] 归一化空间
            if sor_enable:
                denoised = sor_filter(denoised)                # 返回 CPU tensor（点数可能变少）
            else:
                denoised = denoised.cpu()

        # 反归一化回原始世界坐标（center/scale 为 CPU tensor）
        denoised_world = denoised * scale + center             # [M,3]
        save_path = os.path.join(out_dir, fn)
        np.savetxt(save_path, denoised_world.numpy(), fmt='%.8f')
        print(f'[完成] {fn}: {N} → {denoised_world.shape[0]} 点  →  {save_path}')

    print(f'\n全部完成，去噪结果在: {out_dir}')


if __name__ == '__main__':
    main()
