from EasyDatas.Prefab.Common.LoadData import LoadData
from EasyDatas.Core import *
import h5py
from pathlib import Path

def load(path : str):
    f = h5py.File(path,"r")
    keys = f.keys()
    ret = {}
    for key in keys:
        ret[key] = f[key][:]
    return ret

class H5pyLoad(LoadData):
    def __init__(self, override_args: dict = {}):
        super().__init__(override_args=override_args, function=load)

    def deal_a_data(self, data):
        path : Path = data["path"]
        data.pop("path")
        data_name = self.args["data_name"]
        load_data = self.function(path)
        for key in load_data:
            data[key] = load_data[key]
        return data