"""One CUDA training batch, checkpoint reload and one-cloud rollout check; no long training."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(args, result):
    import numpy as np
    import torch
    from tools import builder, runner_distill as runner
    from utils.config import cfg_from_yaml_file

    started = time.monotonic()
    config = cfg_from_yaml_file(args.config)
    nodes = runner.configured_teacher_nodes(config)
    assert config.total_bs == config.student_patch_batch == 8 and config.teacher_patch_batch == 1
    assert config.model.depth == 24 and config.model.trans_dim == 1024
    device = torch.device('cuda', args.device)
    torch.cuda.set_device(device)
    torch.set_num_threads(int(config.cpu_threads))
    torch.cuda.set_per_process_memory_fraction(float(config.gpu_mem_fraction), device)
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)

    def emit(key, value):
        result[key] = value
        print(json.dumps({key: value}, allow_nan=False), flush=True)

    emit('protocol', dict(teacher_nodes=list(nodes), stage_gaps=[u-t for t,u in zip(nodes[:-1],nodes[1:])],
                          seed=args.seed, gpu=torch.cuda.get_device_name(device), batch=8,
                          teacher_batch=1, shadow_search=False, formal_training=False))
    result['source_sha256'] = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                              for folder in ('models','tools','datasets','utils','cfgs')
                              for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.yaml')}
    teacher_path = Path(args.teacher_ckpt)
    checkpoint_stat = (teacher_path.stat().st_size, teacher_path.stat().st_mtime_ns)
    teacher = builder.model_builder(config.model).to(device)
    builder.load_model(teacher, str(teacher_path))
    student = runner.enable_student_condition(copy.deepcopy(teacher))
    for name, value in teacher.state_dict().items():
        assert torch.equal(value, student.state_dict()[name]), name
    assert teacher.step_condition is None
    student.requires_grad_(True).train()
    runner.freeze_teacher(teacher)
    teacher_versions = {name:p._version for name,p in teacher.named_parameters()}
    teacher_buffers = {name:b.detach().cpu().clone() for name,b in teacher.named_buffers()}
    optimizer = torch.optim.AdamW(student.parameters(), lr=float(config.learning_rate),
                                 weight_decay=float(config.weight_decay))
    loader = runner._train_loader(config)
    iterator = iter(loader)
    noisy, clean, sigmas, _, _, names = next(iterator)
    del iterator
    assert noisy.shape == clean.shape == (8,1024,3)
    emit('sampling', dict(dataset_patches=len(loader.dataset), batches=len(loader), names=list(names),
                          sigma0=sigmas.tolist()))
    teacher_calls = []
    def teacher_hook(model, inputs, kwargs):
        assert not torch.is_grad_enabled() and not model.training and inputs[0].shape == (1,1024,3)
        teacher_calls.append(1)
    handle = teacher.register_forward_pre_hook(teacher_hook, with_kwargs=True)
    states = runner.capture_teacher(teacher, noisy, sigmas, 1, teacher_nodes=nodes)
    handle.remove()
    assert len(teacher_calls)==128 and states.shape==(5,8,1024,3) and not states.requires_grad
    calls, encodings = [], []
    def student_hook(model, inputs, kwargs):
        k = len(calls)
        t,u = nodes[k:k+2]
        assert (kwargs['start_step'],kwargs['target_step'])==(t,u)
        assert torch.is_grad_enabled() and model.training and inputs[0].shape==(8,1024,3)
        assert torch.equal(inputs[0].cpu(),states[k])
        torch.testing.assert_close(kwargs['noise_std'],sigmas.to(device)*.95**t,rtol=0,atol=0)
        calls.append(dict(start=t,target=u,sigma=kwargs['noise_std'].tolist(),batch=8))
    a=student.register_forward_pre_hook(student_hook,with_kwargs=True)
    b=student.step_condition.mlp[0].register_forward_pre_hook(
        lambda _, inputs: encodings.append(inputs[0].detach().cpu().clone()))
    diagnostics=[]
    optimizer.zero_grad(set_to_none=True)
    losses=runner.backward_stages(student,states,sigmas,stage_diagnostics=diagnostics,teacher_nodes=nodes)
    a.remove(); b.remove()
    assert len(calls)==4
    for encoding,(t,u) in zip(encodings,zip(nodes[:-1],nodes[1:])):
        torch.testing.assert_close(encoding,torch.tensor([t,u,u-t]).float().expand(8,-1)/16,rtol=0,atol=0)
    runner.check_gradients(teacher,student)
    def grad_norm(parameters):
        grads=[p.grad for p in parameters if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        value=float(torch.stack([g.float().norm() for g in grads]).norm())
        assert value>0 and np.isfinite(value)
        return value
    emit('gradient_norms', dict(condition=grad_norm(student.step_condition.parameters()),
                                encoder=grad_norm(student.encoder.parameters()),
                                transformer=grad_norm(student.blocks.parameters()),
                                decoder=grad_norm(student.generator_blocks.parameters()),
                                total=grad_norm(student.parameters())))
    emit('stage_losses',losses)
    for k,key in enumerate(('mean_D_move','mean_E_imit','mean_PCD')):
        emit(key,[v[k] for v in diagnostics])
    emit('stage_calls',calls)
    selected=[student.step_condition.mlp[-1].weight,next(student.encoder.parameters())]
    before=[p.detach().clone() for p in selected]
    torch.nn.utils.clip_grad_norm_(student.parameters(),float(config.grad_norm_clip),error_if_nonfinite=True)
    optimizer.step()
    assert all(not torch.equal(old,p) for old,p in zip(before,selected))
    assert all(torch.isfinite(p).all() for p in student.parameters())
    emit('optimizer_steps',1)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    emit('trajectory_validation',runner.validate_trajectory(student,[(states,sigmas)],8,teacher_nodes=nodes))

    # Reload the exact new checkpoint through both test and trajectory-validation interfaces.
    # The temporary model checkpoint is removed on exit, never used as a formal best/last.
    with tempfile.TemporaryDirectory(prefix='checkpoint_reload_',dir=args.output_dir) as directory:
        checkpoint=Path(directory)/'smoke-student.pth'
        runner._save_checkpoint(checkpoint,student,None,0,teacher_path,config)
        expected={name:p.detach().cpu().clone() for name,p in student.step_condition.named_parameters()}
        del student
        torch.cuda.empty_cache()
        restored=builder.model_builder(config.model).to(device)
        runner.load_student_checkpoint(restored,checkpoint,builder,expected_nodes=nodes)
        for name,p in restored.step_condition.named_parameters(): assert torch.equal(p.cpu(),expected[name])
        emit('reloaded_trajectory_validation',runner.validate_trajectory(restored,[(states,sigmas)],8,teacher_nodes=nodes))
        del restored
        torch.cuda.empty_cache()
        test_output=Path(args.output_dir)/'test_one_cloud'
        test_output.mkdir()
        runner.test(argparse.Namespace(max_shapes=1,save_trajectory=True),config,builder,device,checkpoint,test_output)
        test_summary=json.loads((test_output/'test_summary.json').read_text())
        emit('one_cloud_rollout_test',test_summary)
        trajectory_path=next(test_output.glob('*_trajectory.npz'))
        with np.load(trajectory_path) as trajectory:
            assert trajectory['global_states'].shape[0]==5
            before_sigmas=trajectory['sigma_before']
            np.testing.assert_allclose(before_sigmas/before_sigmas[0],np.array([.95**t for t in nodes[:-1]]),rtol=1e-6)
    assert all(not m.training for m in teacher.modules())
    for name,p in teacher.named_parameters():
        assert not p.requires_grad and p.grad is None and p._version==teacher_versions[name]
    for name,b in teacher.named_buffers(): assert torch.equal(b.cpu(),teacher_buffers[name])
    assert (teacher_path.stat().st_size,teacher_path.stat().st_mtime_ns)==checkpoint_stat
    emit('integrity',dict(teacher_frozen_eval_unchanged=True,strict_checkpoint_load=True,
                          student_initialized_from_teacher=True,condition_and_sigma_checked=True,
                          original_data_loader=True,checkpoint_reload_passed=True,shadow_calls=0))
    emit('peak_gpu_allocated_gib',torch.cuda.max_memory_allocated()/1024**3)
    emit('elapsed_seconds',time.monotonic()-started)
    result['passed']=True


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='cfgs/PointGPT-L/distill_16to4_nonuniform.yaml')
    parser.add_argument('--teacher_ckpt',required=True)
    parser.add_argument('--output_dir',required=True)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--seed',type=int,default=0)
    args=parser.parse_args()
    output=Path(args.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    if any(output.iterdir()): raise ValueError('Use a new empty smoke output directory')
    result=dict(passed=False)
    try:
        run(args,result)
    except Exception as exc:
        result.update(error=str(exc),traceback=traceback.format_exc())
        traceback.print_exc()
    finally:
        (output/'smoke_result.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
        print(json.dumps({'passed':result['passed']}),flush=True)
    return 0 if result['passed'] else 1


if __name__=='__main__':
    sys.exit(main())
