import pathlib
from typing import List
from EasyDatas.Core import *
from pathlib import Path
import hashlib
import inspect

class LoadData(CachedTransform):
    def get_func_name(self, func):
        md5 = hashlib.md5()
        lines = inspect.getsourcelines(func)[0]
        for lid in range(len(lines)):
            lines[lid] = lines[lid].strip()
        md5.update("\n".join(lines).encode("utf-8"))
        return str(md5.hexdigest())

    def set_global_func(self, func):
        g = globals()
        g[self.func_name] = func

    def get_global_func(self, func_name = None):
        if not func_name:
            func_name = self.func_name
        g = globals()
        return g[func_name]

    def __init__(self, override_args: dict = {}, function = lambda x : open(str(x),"r").readlines()):
        super().__init__(override_args=override_args)
        self.func_name = self.get_func_name(function)
        self.set_global_func(function)

    def _default_init(self):
        super()._default_init()
        self.args["data_name"] = "data"
    
    def deal_a_data(self, data):
        path : Path = data["path"]
        data.pop("path")
        data_name = self.args["data_name"]
        load_data = self.get_global_func()(path)
        data[data_name] = load_data
        return data