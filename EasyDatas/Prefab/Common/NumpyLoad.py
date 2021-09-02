from EasyDatas.Prefab.Common.LoadData import LoadData
from EasyDatas.Core import *
import numpy as np

class NumpyLoad(LoadData):
    def __init__(self, override_args: dict = {}):
        super().__init__(override_args=override_args, function=lambda x : np.loadtxt(str(x)))

    def deal_a_data(self, data):
        return super().deal_a_data(data)

class NumpyLoadNPY(LoadData):
    def __init__(self, override_args: dict = {}):
        super().__init__(override_args=override_args, function=lambda x : np.load(str(x)))

    def deal_a_data(self, data):
        return super().deal_a_data(data)