import pathlib
from typing import List
from EasyDatas.Core import *
from pathlib import Path

class LoadData(CachedTransform):
    def __init__(self, override_args: dict = {}, function = lambda x : open(str(x),"r").readlines()):
        super().__init__(override_args=override_args)
        self.function = function

    def _default_init(self):
        super()._default_init()
        self.args["data_name"] = "data"
    
    def deal_a_data(self, data):
        path : Path = data["path"]
        data.pop("path")
        data_name = self.args["data_name"]
        load_data = self.function(path)
        data[data_name] = load_data
        return data