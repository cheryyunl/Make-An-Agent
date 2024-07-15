import os
import ast
import argparse


class Config(dict):
    def __init__(self, seq=None, **kwargs):
        if seq is None:
            seq = {}
        super(Config, self).__init__(seq, **kwargs)

    def __setattr__(self, key, value):
        self[key] = value

    def __getattr__(self, item):
        return self[item]

    def __str__(self):
        disc = []
        for k in self:
            if k.startswith("_"):
                continue
            disc.append(f"{k}: {repr(self[k])},\n")
        return "".join(disc)

    def copy(self):
        return Config(self)

    def load_saved(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Error: file {path} not exists")
        lines = open(path, 'r').readlines()
        dic = {}
        for l in lines:
            key, value = l.strip().split(':', 1)
            if value == "":
                break
            key = key.strip()
            value = value.strip().rstrip(',')
            dic[key] = ast.literal_eval(value)
        self.update(dic)
        return self


class ARGConfig(Config):
    def __init__(self, seq=None, **kwargs):
        seq = {} if seq is None else seq
        super(ARGConfig, self).__init__(seq, **kwargs)
        self._arg_dict = dict(seq, **kwargs)
        self._arg_help = dict()

    def add_arg(self, key, value, help_str=""):
        self._arg_dict[key] = value
        self._arg_help[key] = f"{help_str} (default: {value})"
        self[key] = value

    def parser(self, desc=""):
        # compiling arg-parser
        parser = argparse.ArgumentParser(description=desc)
        for k in self._arg_dict:
            arg_name = k.replace(' ', '_').replace('-', '_')
            help_msg = self._arg_help[k] if k in self._arg_help else ""
            parser.add_argument(f"--{arg_name}", type=str,
                                default=self._arg_dict[k] if isinstance(self._arg_dict[k], str) else repr(self._arg_dict[k]),
                                help=help_msg)

        pared_args = parser.parse_args().__dict__

        for k in self._arg_dict:
            arg_name = k.replace(' ', '_').replace('-', '_')
            self[k] = self._value_from_string(pared_args[arg_name], type(self[k]))

    @staticmethod
    def _value_from_string(string: str, typeinst: type):
        if typeinst == str:
            return string
        elif typeinst == int:
            return int(string)
        elif typeinst == float:
            return float(string)
        elif typeinst == bool:
            return string.lower() == "true"
        elif typeinst == tuple or typeinst == list:
            return typeinst(ast.literal_eval(string))
        else:
            raise TypeError(f"unknown type (str, tuple, list, int, float, bool), but get {typeinst}")

mw_config = Config({
    "seed": [720, 920],
    "tag": "metaworld",
    "algor": "SAC",
    "start_steps": 5e3,
    "cuda": True,
    "num_steps": 1000001,
    "device": 0,
    "reward_type": "dense",
    
    "env_name": "window-open-v2-goal-observable", 
    "eval": True,
    "eval_episodes": 1,
    "eval_interval": 10,
    "replay_size": 1000000,

    "policy": "Gaussian",   # 'Policy Type: Gaussian | Deterministic (default: Gaussian)'
    "gamma": 0.99, 
    "tau": 0.005,
    "lr": 0.0003,
    "alpha": 0.2,
    "quantile": 0.9,
    "automatic_entropy_tuning": True,
    "use_opt": False,
    "use_elu": False,
    "batch_size": 128, 
    "updates_per_step": 1,
    "target_update_interval": 2,
    "hidden_size": 128,
    "msg": "default"
})
