from tools import pretrain_run_net as pretrain
from tools import finetune_run_net as finetune
from tools import distill_run_net as distill
from tools import test_run_net as test_net
from utils import parser, dist_utils, misc
from utils.logger import *
from utils.config import *
import time
import os
import torch
from tensorboardX import SummaryWriter
from torchstat import stat


def apply_resource_limits(args, config, logger=None):
    """主动给进程加 CPU/GPU 资源上限（共享服务器防吃满核 / 防单进程把整卡显存吃爆崩机）。

    与 runner 里"显存超阈值就保存退出"(check_memory_and_exit) 互补：那是被动监控+退出，
    这里是主动封顶。两个 yaml 旋钮（缺省/<=0 = 不限制，保持旧行为）：
      - cpu_threads:       PyTorch + numpy/MKL CPU 线程数上限
      - gpu_mem_fraction:  单进程可用显存占整卡比例上限 (0,1]，超限会触发可被 try/except 捕获的
                           OOM（跳过该 batch）而不是把整卡/服务器拖崩
    """
    def _log(msg):
        print(msg)
        if logger is not None:
            logger.info(msg)

    cpu_threads = int(config.get('cpu_threads', 0) or 0)
    if cpu_threads > 0:
        torch.set_num_threads(cpu_threads)
        # env 给 DataLoader worker 子进程 / numpy / MKL 继承（worker 在此之后 fork）
        for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[_k] = str(cpu_threads)
        _log(f'[资源限制] CPU 线程上限 = {cpu_threads}')

    gpu_frac = float(config.get('gpu_mem_fraction', 0) or 0)
    if gpu_frac > 0 and args.use_gpu:
        try:
            torch.cuda.set_per_process_memory_fraction(gpu_frac, args.local_rank)
            _log(f'[资源限制] GPU 单进程显存上限 = {gpu_frac:.0%}（device {args.local_rank}）')
        except Exception as e:
            _log(f'[资源限制] 设置 GPU 显存比例失败（已忽略）: {e}')


def main():
    # args
    args = parser.get_args()
    # CUDA
    args.use_gpu = torch.cuda.is_available()
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
    else:
        args.distributed = True
        dist_utils.init_dist(args.launcher)
        # re-set gpu_ids with distributed training mode
        _, world_size = dist_utils.get_dist_info()
        args.world_size = world_size
    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = os.path.join(args.experiment_path, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, name=args.log_name)
    # define the tensorboard writer
    if not args.test:
        if args.local_rank == 0:
            train_writer = SummaryWriter(
                os.path.join(args.tfboard_path, 'train'))
            val_writer = SummaryWriter(os.path.join(args.tfboard_path, 'test'))
        else:
            train_writer = None
            val_writer = None
    # config
    config = get_config(args, logger=logger)
    # batch size
    if args.distributed:
        assert config.total_bs % world_size == 0
        config.dataset.train.others.bs = config.total_bs // world_size
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs // world_size * 2
        config.dataset.val.others.bs = config.total_bs // world_size * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs // world_size
    else:
        config.dataset.train.others.bs = config.total_bs
        if config.dataset.get('extra_train'):
            config.dataset.extra_train.others.bs = config.total_bs * 2
        config.dataset.val.others.bs = config.total_bs * 2
        if config.dataset.get('test'):
            config.dataset.test.others.bs = config.total_bs
    # log
    log_args_to_file(args, 'args', logger=logger)
    log_config_to_file(config, 'config', logger=logger)
    # exit()
    logger.info(f'Distributed training: {args.distributed}')
    # 主动资源限制（CPU 线程数 / GPU 单进程显存比例），从 yaml 读，缺省不限制
    apply_resource_limits(args, config, logger=logger)
    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        # seed + rank, for augmentation
        misc.set_random_seed(args.seed + args.local_rank,
                             deterministic=args.deterministic)
    if args.distributed:
        assert args.local_rank == torch.distributed.get_rank()

    if args.shot != -1:
        config.dataset.train.others.shot = args.shot
        config.dataset.train.others.way = args.way
        config.dataset.train.others.fold = args.fold
        config.dataset.val.others.shot = args.shot
        config.dataset.val.others.way = args.way
        config.dataset.val.others.fold = args.fold

    # run
    if args.test:
        test_net(args, config)
    else:
        if args.distill_model:
            distill(args, config, train_writer, val_writer)
        elif args.finetune_model or args.scratch_model:
            finetune(args, config, train_writer, val_writer)
        else:
            pretrain(args, config, train_writer, val_writer)


if __name__ == '__main__':
    main()
