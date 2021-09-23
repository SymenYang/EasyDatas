from EasyDatas.Core import EasyDatasBase
import inspect
import hashlib
import logging
from pathlib import Path

class Picker(EasyDatasBase):
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

    def __init__(self, override_args: dict = {}, pick_func = lambda x,id,l : True):
        super().__init__(override_args=override_args)
        self.func_name = self.get_func_name(pick_func)
        self.set_global_func(pick_func)
        self.look_up_table = []
        assert self.need_previous, "{} shold have a previous dataset".format(self.__class__.__name__)

    def name_args(self):
        args = super().name_args()
        args["func"] = str(self.func_name)
        return args

    def _default_init(self):
        super()._default_init()
        self.args["need_previous"] = True
        self.args["need_cache"] = True
    
    def deal_datas(self):
        length = len(self.previous)
        g = globals()
        for i in range(length):
            if self.get_global_func()(self.get(i, False),i,length):
                self.put(self.get(i, False))