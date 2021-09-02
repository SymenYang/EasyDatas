import pathlib
from typing import List
from EasyDatas.Core import *
from pathlib import Path
import re

class ListFile(RawDatas):
    def _default_init(self):
        super()._default_init()
        self.args["pattern"] = ".+"
        self.args["files"] = "True"
        self.args["dirs"] = "True"

    def name_args(self):
        return {
            "name" : Path(self.args["path"]).name,
            "pattern" : self.args["pattern"],
            "files" : self.args["files"],
            "dirs" : self.args["dirs"]
        }

    def deal_datas(self):
        path = Path(self.args["path"])
        pattern = self.args["pattern"]
        files,dirs = self.args["files"],self.args["dirs"]
        for item in sorted(path.iterdir()):
            if item.is_dir() and not dirs:
                continue
            if item.is_file() and not files:
                continue
            if re.match(pattern,item.name):
                self.put({
                    "path" : item
                })

class SpanFiles(CachedTransform):
    def _default_init(self):
        super()._default_init()
        self.args["pattern"] = ".+"
        self.args["files"] = "True"
        self.args["dirs"] = "True"
    
    def name_args(self):
        return {
            "pattern" : self.args["pattern"],
            "files" : self.args["files"],
            "dirs" : self.args["dirs"]
        }

    def deal_a_data(self,data : dict):
        path = data["path"]
        pattern = self.args["pattern"]
        files,dirs = self.args["files"],self.args["dirs"]
        for item in sorted(path.iterdir()):
            if item.is_dir() and not dirs:
                continue
            if item.is_file() and not files:
                continue
            if re.match(pattern,item.name):
                self.put({
                    "path" :  item
                })

class RecursionFiles(ListFile):
    """
    Find all files that absolute path match the pattern and is file or dir according to the args
    """
    def recursion_find(self,now : Path, pattern : str, files : bool, dirs : bool):
        for item in sorted(now.iterdir()):
            if item.is_dir():
                self.recursion_find(item,pattern,files,dirs)
                if not dirs:
                    continue
            if item.is_file() and not files:
                continue
            if re.match(pattern,str(item)):
                self.put({
                    "path" :  item
                })


    def deal_datas(self):
        path = Path(self.args["path"])
        pattern = self.args["pattern"]
        files,dirs = self.args["files"],self.args["dirs"]
        self.recursion_find(path,pattern,files,dirs)