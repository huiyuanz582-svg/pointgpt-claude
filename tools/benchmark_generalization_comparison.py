"""Teacher10, Vanilla Student4 and frozen local+endpoint Student4 generalization.

Manual evaluation only. PUNet lap/cov use paired SOR off/on; PCNet uses off.
The shared coordinator runs one model/scene at a time in a fresh process.
"""

import copy
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for entry in (ROOT, ROOT / 'tools'):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

import benchmark_teacher4_generalization as generalization
from utils.teacher4_generalization import validate_comparison_jobs

benchmark = generalization.benchmark
EXTRA_SOURCES = generalization.EXTRA_SOURCES + ('tools/benchmark_generalization_comparison.py',)
TEACHER_SHA256 = 'b02b7a1acfc8b2345cc33cbb7ff99ddd15ae53042fb94a9a1e722d760b4c730d'
VANILLA_SHA256 = '19b09fd485092c02c208dd59eb834f888035c2da231216ebe1895d6230a92403'
METHOD_SHA256 = '94da5dd125e92bec221bb3a9a72b0896ca48eaef86c05bd057516517950e5a12'


def student_variant(path, variant_id, expected_hash, expected_epoch, model_config):
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f'Missing {variant_id} checkpoint: {path}\n'
                                f'Project root: {ROOT}. In Docker, use a mounted, container-visible path.')
    actual = benchmark.sha256_file(path)
    if actual != expected_hash.lower():
        raise ValueError(f'{variant_id}: checkpoint SHA256 differs from the frozen model identity '
                         f'(expected {expected_hash.lower()}, got {actual})')
    return dict(id=variant_id, kind='student', steps=4, checkpoint=str(path), checkpoint_sha256=actual,
                checkpoint_validation='fixed_10to4', expected_epoch=expected_epoch,
                model_config=copy.deepcopy(model_config))


def make_plan(args):
    plan = generalization.make_plan(args, teacher_steps=10, extra_sources=EXTRA_SOURCES)
    scene_jobs = plan['jobs']
    model = scene_jobs[0]['config']['model']
    variants = plan['variants'] + [
        student_variant(args.vanilla_ckpt, 'student_baseline', args.expected_vanilla_sha256, 6, model),
        student_variant(args.method_ckpt, 'student_local_endpoint', args.expected_method_sha256, 1, model)]
    if len({v['checkpoint_sha256'] for v in variants}) != len(variants):
        raise ValueError('Teacher10, Vanilla and Method must use distinct checkpoints')
    plan.update(schema_version=3, benchmark_mode='generalization_comparison', variants=variants)
    plan['settings'].update(student_steps=4, student_teacher_nodes=list(benchmark.NODES),
                            student_eta=1.0, student_decay=list(benchmark.SCHEDULE['student_decay']))
    plan['jobs'] = [generalization.scene_job(
        {k: value for k, value in job.items() if k not in ('variant', 'config')}, variant, job['config'])
        for job in scene_jobs for variant in variants]
    validate_comparison_jobs(plan)
    return plan


def main():
    parser = generalization.argument_parser()
    parser.description = __doc__
    parser.add_argument('--vanilla_ckpt', type=Path, help='Frozen ordinary distillation ckpt-best.pth')
    parser.add_argument('--method_ckpt', type=Path, help='Frozen joint local+endpoint ckpt-best.pth')
    parser.add_argument('--expected_vanilla_sha256', default=VANILLA_SHA256)
    parser.add_argument('--expected_method_sha256', default=METHOD_SHA256)
    parser.set_defaults(expected_checkpoint_sha256=TEACHER_SHA256)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker_plan:
        if args.job_index is None:
            parser.error('--worker_plan requires --job_index')
        generalization.run_worker(args.worker_plan.resolve(), args.job_index)
        return
    if args.report_only:
        root = args.report_only.expanduser().resolve()
        generalization.build_report(root, benchmark.load_json(root / 'plan.json'))
        benchmark.package(root)
        return
    if any(path is None for path in (args.teacher_ckpt, args.vanilla_ckpt, args.method_ckpt)):
        parser.error('Provide --teacher_ckpt, --vanilla_ckpt and --method_ckpt')
    if not 0 <= args.seed < 2 ** 32 or args.device < 0 or min(args.patch_batch, args.warmup, args.repeats) < 1:
        parser.error('Require uint32 seed, nonnegative device and positive patch_batch/warmup/repeats')
    if args.resume and args.output_root is None:
        parser.error('--resume requires the existing --output_root')
    generalization.execute_plan(args, make_plan(args), __file__, 'generalization_comparison_sor',
                                'docs/generalization_comparison.md')


if __name__ == '__main__':
    main()
