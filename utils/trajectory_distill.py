"""固定教师终点的少步轨迹蒸馏工具。

第一阶段只处理已经由 Stage 0 验证过的固定 ``16 -> 4`` 映射。教师仍按原始
``step_size=0.3, decay=0.95`` 逐步更新；学生一次前向预测一个教师区间的累计
归一化位移，并在 4 次前向后到达教师第 16 步附近。

这里处理的是训练数据已经切好的 1024 点 patch，不做完整点云的外层 FPS/KNN
切块或融合。完整点云验证仍由 ``runner_finetune.patch_based_denoise`` 完成。
"""

from collections import OrderedDict

import torch
import torch.nn.functional as F


def load_finetuned_checkpoint(model, checkpoint_path, allow_conditioning_missing=False):
    """精确加载第一阶段微调 checkpoint，不执行预训练输出头重置。"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    raw_state = checkpoint.get('base_model', checkpoint.get('model'))
    if raw_state is None:
        raise RuntimeError(f'checkpoint 不含 base_model/model: {checkpoint_path}')
    state = {(key[len('module.'):] if key.startswith('module.') else key): value
             for key, value in raw_state.items()}
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if allow_conditioning_missing:
        missing = [key for key in missing
                   if not key.startswith('distill_condition_mlp.')]
    if missing or unexpected:
        raise RuntimeError(
            '微调 checkpoint 与模型结构不兼容：'
            f'missing={missing}, unexpected={unexpected}')
    return checkpoint


def validate_teacher_indices(teacher_indices, student_steps=None):
    """规范并验证学生阶段对应的教师状态索引。"""
    indices = [int(x) for x in teacher_indices]
    if len(indices) < 2 or indices[0] != 0:
        raise ValueError('teacher_indices 必须从 0 开始且至少包含两个状态')
    if any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError(f'teacher_indices 必须严格递增，当前为 {indices}')
    if student_steps is not None and len(indices) != int(student_steps) + 1:
        raise ValueError(
            f'teacher_indices 长度应为 student_steps+1={int(student_steps)+1}，'
            f'当前为 {len(indices)}')
    return indices


def uniform_teacher_indices(teacher_endpoint, student_steps):
    """生成均匀整数映射；要求终点可被学生步数整除，避免含糊的舍入日程。"""
    teacher_endpoint = int(teacher_endpoint)
    student_steps = int(student_steps)
    if teacher_endpoint < 1 or student_steps < 1:
        raise ValueError('teacher_endpoint/student_steps 必须 >= 1')
    if teacher_endpoint % student_steps != 0:
        raise ValueError(
            f'第一阶段要求 teacher_endpoint({teacher_endpoint}) 可被 '
            f'student_steps({student_steps}) 整除')
    stride = teacher_endpoint // student_steps
    return [i * stride for i in range(student_steps + 1)]


def make_distill_condition(sigma_t, stage_idx, student_steps, teacher_update_step,
                           teacher_max_steps, teacher_endpoint, sigma_ref=0.01):
    """构造 ``[log(σ/σ_ref), stage, teacher_time, endpoint]`` 四维条件。"""
    if sigma_t.ndim != 1:
        sigma_t = sigma_t.reshape(-1)
    if torch.any(~torch.isfinite(sigma_t)) or torch.any(sigma_t <= 0):
        raise ValueError('sigma_t 必须为有限正数')
    denom_stage = max(int(student_steps) - 1, 1)
    denom_teacher = max(int(teacher_max_steps), 1)
    values = (
        torch.log(sigma_t / float(sigma_ref)),
        torch.full_like(sigma_t, float(stage_idx) / denom_stage),
        torch.full_like(sigma_t, float(teacher_update_step) / denom_teacher),
        torch.full_like(sigma_t, float(teacher_endpoint) / denom_teacher),
    )
    return torch.stack(values, dim=-1)


def teacher_sigma(sigma0, update_step, decay):
    """返回教师第 ``update_step`` 次更新前使用的 σ。"""
    return sigma0.reshape(-1) * (float(decay) ** int(update_step))


@torch.no_grad()
def rollout_teacher_patch(teacher, noisy_patch, sigma0, teacher_indices,
                          step_size=0.3, decay=0.95):
    """运行冻结教师并只保留所需 landmark 状态，返回 ``OrderedDict[int, Tensor]``。"""
    indices = validate_teacher_indices(teacher_indices)
    endpoint = indices[-1]
    wanted = set(indices)
    states = OrderedDict([(0, noisy_patch.detach())])
    x = noisy_patch.detach()
    for update_step in range(endpoint):
        sigma_t = teacher_sigma(sigma0, update_step, decay).to(x.device)
        teacher_denoised = teacher(
            x, None, 'val', '', noise_std=sigma_t)
        # teacher_denoised = x + σ_t ε；原教师实际只走 step_size 比例。
        x = x + float(step_size) * (teacher_denoised - x)
        next_step = update_step + 1
        if next_step in wanted:
            states[next_step] = x.detach()
    if list(states.keys()) != indices:
        raise RuntimeError(
            f'教师 landmark 捕获不完整：期望 {indices}，得到 {list(states.keys())}')
    return states


def predict_student_jump(student, x, sigma0, stage_idx, teacher_indices,
                         teacher_decay=0.95, teacher_max_steps=30):
    """预测一个学生阶段的累计归一化位移与下一状态。"""
    indices = validate_teacher_indices(teacher_indices)
    student_steps = len(indices) - 1
    if not 0 <= int(stage_idx) < student_steps:
        raise IndexError(f'stage_idx 超界：{stage_idx}, student_steps={student_steps}')
    teacher_step = indices[int(stage_idx)]
    sigma_t = teacher_sigma(sigma0, teacher_step, teacher_decay).to(x.device)
    condition = make_distill_condition(
        sigma_t, stage_idx, student_steps, teacher_step,
        teacher_max_steps, indices[-1])
    pred_jump = student(
        x, None, 'distill', '', noise_std=sigma_t,
        distill_condition=condition, return_pred_score=True)
    x_next = x + sigma_t.view(-1, 1, 1) * pred_jump
    return pred_jump, x_next, sigma_t


def teacher_jump_target(state_from, state_to, sigma_t):
    """把教师区间累计位移归一化到学生输出空间。"""
    return (state_to - state_from) / sigma_t.view(-1, 1, 1)


def teacher_forced_jump_loss(student, teacher_states, sigma0, stage_idx,
                             teacher_indices, teacher_decay=0.95,
                             teacher_max_steps=30, loss_type='smooth_l1'):
    """在教师 landmark 输入上监督单个跳步，是显存友好的第一阶段预热损失。"""
    indices = validate_teacher_indices(teacher_indices)
    left = indices[int(stage_idx)]
    right = indices[int(stage_idx) + 1]
    x_from = teacher_states[left]
    x_to = teacher_states[right]
    pred, _, sigma_t = predict_student_jump(
        student, x_from, sigma0, stage_idx, indices,
        teacher_decay=teacher_decay, teacher_max_steps=teacher_max_steps)
    target = teacher_jump_target(x_from, x_to, sigma_t)
    if loss_type == 'mse':
        loss = F.mse_loss(pred, target)
    elif loss_type == 'smooth_l1':
        loss = F.smooth_l1_loss(pred, target)
    else:
        raise ValueError(f'不支持的 jump loss: {loss_type}')
    return loss, pred, target


def rollout_student_patch(student, noisy_patch, sigma0, teacher_indices,
                          teacher_decay=0.95, teacher_max_steps=30):
    """完整展开少步学生；不 detach，允许端点损失反传到所有学生阶段。"""
    indices = validate_teacher_indices(teacher_indices)
    states = [noisy_patch]
    fields = []
    x = noisy_patch
    for stage_idx in range(len(indices) - 1):
        pred, x, _ = predict_student_jump(
            student, x, sigma0, stage_idx, indices,
            teacher_decay=teacher_decay, teacher_max_steps=teacher_max_steps)
        fields.append(pred)
        states.append(x)
    return states, fields


def rollout_distill_loss(student_states, student_fields, teacher_states,
                         clean_patch, sigma0, teacher_indices,
                         teacher_decay=0.95, jump_weight=1.0,
                         trajectory_weight=0.25, endpoint_weight=1.0,
                         clean_weight=0.5, jump_loss_type='smooth_l1'):
    """计算完整学生 rollout 的跳步、轨迹、教师端点和 clean 锚定损失。"""
    indices = validate_teacher_indices(teacher_indices)
    if len(student_states) != len(indices):
        raise ValueError('student_states 与 teacher_indices 长度不一致')
    if len(student_fields) != len(indices) - 1:
        raise ValueError('student_fields 与学生步数不一致')

    jump_terms = []
    for stage_idx, pred in enumerate(student_fields):
        left, right = indices[stage_idx], indices[stage_idx + 1]
        sigma_t = teacher_sigma(sigma0, left, teacher_decay).to(pred.device)
        target = teacher_jump_target(
            teacher_states[left], teacher_states[right], sigma_t)
        if jump_loss_type == 'mse':
            jump_terms.append(F.mse_loss(pred, target))
        elif jump_loss_type == 'smooth_l1':
            jump_terms.append(F.smooth_l1_loss(pred, target))
        else:
            raise ValueError(f'不支持的 jump loss: {jump_loss_type}')
    jump = torch.stack(jump_terms).mean()

    intermediate = [
        F.smooth_l1_loss(student_states[j], teacher_states[indices[j]])
        for j in range(1, len(indices) - 1)
    ]
    trajectory = (torch.stack(intermediate).mean() if intermediate
                  else student_states[-1].new_zeros(()))
    endpoint = F.smooth_l1_loss(student_states[-1], teacher_states[indices[-1]])
    clean = F.smooth_l1_loss(student_states[-1], clean_patch)
    total = (float(jump_weight) * jump +
             float(trajectory_weight) * trajectory +
             float(endpoint_weight) * endpoint +
             float(clean_weight) * clean)
    terms = {
        'jump': jump,
        'trajectory': trajectory,
        'endpoint': endpoint,
        'clean': clean,
    }
    return total, terms
