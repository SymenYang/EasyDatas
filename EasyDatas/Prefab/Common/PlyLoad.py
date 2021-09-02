from pathlib import Path

import numpy as np
import plyfile
from EasyDatas.Core import *
from EasyDatas.Prefab.Common.LoadData import LoadData


def load_data(path : str, element_name = None, element_id = 0):
    data = plyfile.PlyData.read(path)
    if element_name is None:
        element_name = data.elements[element_id].name
    
    properties = data[element_name].properties
    length = len(data[element_name])
    
    ret = {}
    for key in properties:
        ret[key.name] = np.array(data[element_name][key.name])
    return ret

def load_combining(path : str, element_name = None, element_id = 0):
    data = plyfile.PlyData.read(path)
    if element_name is None:
        element_name = data.elements[element_id].name
    
    properties = data[element_name].properties
    length = len(data[element_name])

    p = {'x' : 0, 'y' : 1, 'z' : 2}
    c = {'red' : 0, "green" : 1, "blue" : 2}
    c2 = {'r' : 0, "g" : 1, "b" : 2}
    points = [None,None,None]
    p_count = 3
    colors = [None,None,None]
    c_count = 3

    ret = {}
    for key in properties:
        if key.name in p:
            points[p[key.name]] = np.array(data[element_name][key.name])
            p_count -= 1
        elif key.name in c:
            colors[c[key.name]] = np.array(data[element_name][key.name])
            c_count -= 1
        elif key.name in c2:
            colors[c2[key.name]] = np.array(data[element_name][key.name])
            c_count -= 1
        else:
            ret[key.name] = np.array(data[element_name][key.name])
    if not p_count:
        ret["points"] = np.stack(points).transpose()
    if not c_count:
        ret["colors"] = np.stack(colors).transpose()
    return ret

class PlyLoad(CachedTransform):
    def custom_init(self):
        self.element_name = self.get_attr("element_name",None)
        self.element_id = self.get_attr("element_id",0)
    
    def deal_a_data(self, data):
        path : Path = data["path"]
        data = load_data(path,self.element_name,self.element_id)
        return data

class PlyLoadCombining(CachedTransform):
    def custom_init(self):
        self.element_name = self.get_attr("element_name",None)
        self.element_id = self.get_attr("element_id",0)
    
    def deal_a_data(self, data):
        path : Path = data["path"]
        data = load_combining(path,self.element_name,self.element_id)
        return data
