import os
from datetime import datetime
import uuid
import json
import copy
from typing import Union, Callable
from dotmap import DotMap
import torch
import torch.nn as nn


def prepare_opt(parser) -> DotMap:
    # Parser to dict
    opt_parser = vars(parser.parse_args())
    # Config file to dict
    config_path = opt_parser['config']
    if not os.path.isfile(config_path):
        config_path = os.path.join('./config/', config_path + '.json')
    with open(config_path, 'r') as config_file:
        opt_config = json.load(config_file)
    # Merge dicts to dotmap
    for k, v in opt_parser.items():
        if v is not None:
            opt_config[k] = v
    return DotMap(**opt_config)


class Logger(object):
    def __init__(self, data: str, algo: str, flag_run: str='', dir: tuple=None):
        super(Logger, self).__init__()

        # init log directory
        self.seed_str = str(uuid.uuid4())[:6]
        self.seed = int(self.seed_str, 16)
        if not flag_run:
            flag_run = datetime.now().strftime("%m%d") + '-' + self.seed_str
        elif flag_run.count('date') > 0:
            flag_run.replace('date', datetime.now().strftime("%m%d"))
        else:
            pass

        if dir is None:
            self.dir_save = os.path.join("./save/", data, algo, flag_run)
        else:
            self.dir_save = os.path.join(*dir)
        self.path_exists = os.path.exists(self.dir_save)

        # init log file
        self.flag_run = flag_run
        self.file_log = self.path_join('log.txt')
        self.file_config = self.path_join('config.json')

        flag_run = flag_run.split('-')[0]
        seed = int(flag_run) if flag_run.isdigit() else 11
        if seed < 10:
            self.lvl_log = 0         # 0~9: nothing
        elif seed < 20:
            self.lvl_log = 1         # 10~19: funtional ouput
        elif seed < 30:
            self.lvl_log = 2         # 20~29: result per epoch
        else:
            self.lvl_log = 3         # 30~: print to file_log
        if seed < 5:
            self.lvl_config = 0      # 0~4: nothing
        elif seed < 15:
            self.lvl_config = 1      # 5~14: simple
        elif seed < 25:
            self.lvl_config = 2      # 15~: detailed
        else:
            self.lvl_config = 3      # 25~: save to file_config

    def path_join(self, *args) -> str:
        """
        Generate file path in current directory.
        """
        return os.path.join(self.dir_save, *args)

    def print(self, s, sf=None, lvl=None) -> None:
        """
        Print string to console and write log file.
        """
        lvl = self.lvl_log if lvl is None else lvl
        if lvl > 0:
            print(s, flush=True)
        if lvl > 2:
            sf = s if sf is None else sf
            with open(self.file_log, 'a') as f:
                f.write(str(sf) + '\n')

    def print_on_top(self, s) -> None:
        """
        Print string on top of log file.
        """
        if self.lvl_log > 0:
            print(s)
        if self.lvl_log > 2:
            with open(self.file_log, 'a') as f:
                pass
            with open(self.file_log, 'r+') as f:
                temp = f.read()
                f.seek(0, 0)
                f.write(str(s) + '\n')
                f.write(temp)

    def print_header(self, hs, s) -> None:
        if self.lvl_log > 0:
            if os.path.isfile(self.file_log):
                print(hs)
            else:
                self.print(hs, hs.replace('|', ','), lvl=self.lvl_config)
            self.print(s, lvl=self.lvl_config)

    def save_opt(self, opt: DotMap) -> None:
        if self.lvl_log > 2:
            os.makedirs(self.dir_save, exist_ok=True)
            with open(self.file_config, 'w') as f:
                json.dump(opt.toDict(), fp=f, indent=4, sort_keys=False)
                f.write('\n')
            print("Option saved.")
            print("Config path: {}".format(self.file_config))
            print("Option dict: {}\n".format(opt.toDict()))

    def load_opt(self) -> DotMap:
        with open(self.file_config, 'r') as config_file:
            opt = DotMap(json.load(config_file))
        print("Option loaded.")
        print("Config path: {}".format(self.file_config))
        print("Option dict: {}\n".format(opt.toDict()))
        return opt

    def str_csv(self, data, algo, seed, thr_a, thr_w,
               acc_test, conv_epoch, epoch, time_train, macs_train,
               time_test, macs_test, numel_a, numel_w):
        hstr, cstr = '', ''
        hstr += f"      Data|     Model|  Seed|     ThA|     ThW| "
        cstr += f"{data:10s},{algo:10s},{seed:6d},{thr_a:7.2e},{thr_w:7.2e},"
        hstr += f"   Acc|  Cn|  EP| "
        cstr += f"{acc_test:7.5f},{conv_epoch:4d},{epoch:4d},"
        hstr += f" Ttrain|  Ctrain| "
        cstr += f"{time_train:8.4f},{macs_train:8.3f},"
        hstr += f"  Ttest|   CTest|  NumelA|  NumelW"
        cstr += f"{time_test:8.4f},{macs_test:8.4f},{numel_a:8.3f},{numel_w:8.3f}"
        return hstr, cstr

    def str_csvg(self, data, algo, seed, thr_a, thr_w,
               acc_test, conv_epoch, epoch, time_train, macs_train,
               macs_a, macs_wtr, macs_wte,
               time_test, macs_test, numel_a, numel_w, hop, layer, time_pre):
        hstr, cstr = '', ''
        hstr += f"      Data|     Model|  Seed|    ThA|    ThW| "
        cstr += f"{data:10s},{algo:10s},{seed:6d},{thr_a:7.1e},{thr_w:7.1e},"
        hstr += f"   Acc|  Cn|  EP| "
        cstr += f"{acc_test:7.5f},{conv_epoch:4d},{epoch:4d},"
        hstr += f" Ttrain|  Ctrain| "
        cstr += f"{time_train:8.4f},{macs_train:8.3f},"
        hstr += f"  Ttest|   CTest|  NumelA|  NumelW| "
        cstr += f"{time_test:8.4f},{macs_test:8.4f},{numel_a:8.3f},{numel_w:8.3f},"
        hstr += f"   CPre|     CTr|     CTe| Hop| Lay|   TPre "
        cstr += f"{macs_a:8.4f},{macs_wtr:8.4f},{macs_wte:8.4f},{hop:4d},{layer:4d},{time_pre:8.4f}"
        return hstr, cstr

class ModelLogger(object):
    """
    Log, save, and load model, with given path, certain prefix, and changeable suffix.
    """
    def __init__(self, logger: Logger, patience: int=99999,
                 prefix: str='model', storage: str='model_gpu',
                 cmp: Union[Callable[[float, float], bool], str]='>'):
        super(ModelLogger, self).__init__()
        self.logger = logger
        self.patience = patience
        self.prefix = prefix
        self.model = None

        # Storage type
        assert storage in ['model', 'state', 'model_ram', 'state_ram', 'model_gpu', 'state_gpu']
        self.storage = storage

        # Comparison function for metric
        if cmp in ['>', 'max']:
            self.cmp = lambda x, y: x > y
        elif cmp in ['<', 'min']:
            self.cmp = lambda x, y: x < y
        else:
            self.cmp = cmp

    @property
    def state_dict(self):
        return self.model.state_dict()

    # ===== Load and save
    def __set_model(self, model: nn.Module) -> nn.Module:
        self.model = model
        return self.model

    def register(self, model: nn.Module, save_init: bool=True) -> None:
        """
        Get model from parameters.

        Args:
            model: model instance
            save_init (bool, optional): Whether save initial model. Defaults to True.
        """
        self.__set_model(model)
        if save_init:
            self.save('0')

    def load(self, *suffix, model: nn.Module=None, map_location='cpu') -> nn.Module:
        """
        Get model from file.
        """
        name = '_'.join((self.prefix,) + suffix)
        path = self.logger.path_join(name + '.pth')

        if self.storage == 'state':
            assert self.model is not None
            if model is None:
                model = self.model
            state_dict = torch.load(path, map_location=map_location)
            model.load_state_dict(state_dict)
        elif self.storage in ['state_ram', 'state_gpu']:
            assert self.model is not None
            assert hasattr(self, 'mem')
            if model is None:
                model = self.model
            if hasattr(self.model, 'remove'):
                self.model.remove()
            model.load_state_dict(self.mem)
            # model.to(map_location)
        elif self.storage == 'model':
            model = torch.load(path, map_location=map_location)
        elif self.storage in ['model_ram', 'model_gpu']:
            model = copy.deepcopy(self.mem)
            # model.to(map_location)

        return self.__set_model(model)

    def save(self, *suffix) -> None:
        """
        Save model with given name string.
        """
        name = '_'.join((self.prefix,) + suffix)
        path = self.logger.path_join(name + '.pth')

        if self.storage == 'state':
            torch.save(self.state_dict, path)
        elif self.storage == 'model':
            torch.save(self.model, path)
        elif self.storage == 'state_gpu':
            # Alternative way is to use BytesIO
            if hasattr(self, 'mem'): del self.mem
            self.mem = copy.deepcopy(self.state_dict)
        elif self.storage == 'state_ram':
            if hasattr(self, 'mem'): del self.mem
            self.mem = copy.deepcopy(self.state_dict)
            self.mem = {k: v.cpu() for k, v in self.mem.items()}
        elif self.storage == 'model_gpu':
            if hasattr(self, 'mem'): del self.mem
            self.mem = copy.deepcopy(self.model)
        elif self.storage == 'model_ram':
            # TODO: reduce mem fro 2xmem(model) to mem(model)
            if hasattr(self, 'mem'): del self.mem
            device = next(self.model.parameters()).device
            self.mem = copy.deepcopy(self.model.cpu())
            self.model.to(device)

    def get_last_epoch(self) -> int:
        """
        Get last saved model epoch. Useful for deciding load model path.

        Returns:
            int: number of last epoch
        """
        name_pre = '_'.join((self.prefix,) + ('',))
        last_epoch = -2

        for fname in os.listdir(self.logger.dir_save):
            fname = str(fname)
            if fname.startswith(name_pre) and fname.endswith('.pth'):
                suffix = fname.replace(name_pre, '').replace('.pth', '')
                if suffix == 'init':
                    this_epoch = -1
                elif suffix.isdigit():
                    # correct the `epoch + 1` in `save_epoch()`
                    this_epoch = int(suffix) - 1
                else:
                    this_epoch = -2
                if this_epoch > last_epoch:
                    last_epoch = this_epoch
        return last_epoch

    # ===== Save during training
    def save_epoch(self, epoch: int, period: int=1) -> None:
        """
        Save model each epoch period.

        Args:
            epoch (int): Current epoch. Start from 0 (display as epoch + 1).
            period (int, optional): Save period. Defaults to 1 (save every epochs).
        """
        if (epoch + 1) % period == 0:
            self.save(str(epoch+1))

    def save_best(self, score: float, epoch: int=-1,
                  print_log: bool=False) -> int:
        """
        Save model if the current epoch is the best.

        Args:
            acc_curr (float): Current metric.
            epoch (int, optional): Current epoch. Defaults to -1.
            print_log (bool, optional): Whether to print log line. Defaults to False.
            compare_fn (callable, optional): Custom comparison function. Defaults to greater than.

        Returns:
            bool: If the current epoch is the best.
        """
        if self.is_best(score, epoch):
            self.save('best')
            if print_log:
                self.logger.print('[best saved] {:>.4f}'.format(self.score_best))
        return self.score_best

    def is_best(self, score: float, epoch: int=-1) -> bool:
        res = (not hasattr(self, 'score_best'))
        if res or self.cmp(score, self.score_best):
            self.score_best = score
            self.epoch_best = epoch
            res = True
        return res

    def is_early_stop(self, epoch: int=-1) -> bool:
        return epoch - self.epoch_best >= self.patience


class LayerNumLogger(object):
    def __init__(self, name: str=None):
        self.name = name
        self.numel_before = None
        self.numel_after = None

    @property
    def ratio(self) -> float:
        return self.numel_after / self.numel_before

    def __str__(self) -> str:
        s = f"{self.numel_after}/{self.numel_before} ({1-self.ratio:6.2%})"
        return s
