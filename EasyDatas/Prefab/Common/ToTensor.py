import torch
import numpy as np
from EasyDatas.Core import Transform

class ToTensor(Transform):
    def deal_a_data(self,data):
        for key in data:
            if isinstance(data[key],np.ndarray):
                data[key] = torch.from_numpy(data[key])
            elif isinstance(data[key],list):
                data[key] = torch.tensor(data[key])
            else:
                continue
            if data[key].dtype == torch.float64:
                data[key] = data[key].float()
        return data