import yaml
from easydict import EasyDict
import os
from .logger import print_log


def _resolve_config_path(path, relative_to=None):
    """Resolve repository-style YAML references without depending on cwd."""
    if os.path.isabs(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidates = [
        os.path.normpath(path),
        os.path.normpath(os.path.join(repo_root, path)),
    ]
    if relative_to is not None:
        candidates.append(os.path.normpath(os.path.join(relative_to, path)))
    return next((candidate for candidate in candidates if os.path.exists(candidate)),
                candidates[-1])


def log_args_to_file(args, pre='args', logger=None):
    for key, val in args.__dict__.items():
        print_log(f'{pre}.{key} : {val}', logger=logger)


def log_config_to_file(cfg, pre='cfg', logger=None):
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            print_log(f'{pre}.{key} = edict()', logger=logger)
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        print_log(f'{pre}.{key} : {val}', logger=logger)


def merge_new_config(config, new_config):
    for key, val in new_config.items():
        if not isinstance(val, dict):
            if key == '_base_':
                base_path = _resolve_config_path(new_config['_base_'])
                with open(base_path, 'r', encoding='utf-8') as f:
                    try:
                        val = yaml.load(f, Loader=yaml.FullLoader)
                    except:
                        val = yaml.load(f)
                config[key] = EasyDict()
                merge_new_config(config[key], val)
            else:
                config[key] = val
                continue
        if key not in config:
            config[key] = EasyDict()
        merge_new_config(config[key], val)
    return config


def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r', encoding='utf-8') as f:
        try:
            new_config = yaml.load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.load(f)
    # 顶层 _base_ 用于实验配置继承。历史数据集配置也使用 _base_，但它位于
    # dataset.train/val/test 内部，由 merge_new_config 保持原语义；这里只处理顶层键。
    # 这样第二阶段实验可以只覆盖蒸馏相关参数，不必复制整份第一阶段配置。
    base_file = new_config.pop('_base_', None) if isinstance(new_config, dict) else None
    if base_file is not None:
        # 兼容从仓库根目录、其他工作目录以及 experiment/config.yaml 恢复训练。
        base_file = _resolve_config_path(base_file, os.path.dirname(cfg_file))
        config = cfg_from_yaml_file(base_file)
    else:
        config = EasyDict()
    merge_new_config(config=config, new_config=new_config)
    return config


def get_config(args, logger=None):
    if args.resume:
        cfg_path = os.path.join(args.experiment_path, 'config.yaml')
        if not os.path.exists(cfg_path):
            print_log("Failed to resume", logger=logger)
            raise FileNotFoundError()
        print_log(f'Resume yaml from {cfg_path}', logger=logger)
        args.config = cfg_path
    config = cfg_from_yaml_file(args.config)
    if not args.resume and args.local_rank == 0:
        save_experiment_config(args, config, logger)
    return config


def save_experiment_config(args, config, logger=None):
    config_path = os.path.join(args.experiment_path, 'config.yaml')
    os.system('cp %s %s' % (args.config, config_path))
    print_log(
        f'Copy the Config file from {args.config} to {config_path}', logger=logger)
