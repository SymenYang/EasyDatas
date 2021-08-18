from os import name
from torch.utils.data import Dataset
import pickle
import json
from pathlib import Path
import collections
import hashlib

class EasyDatasBase(Dataset):
    _class_count = 0
    def __init__(self, override_args : dict = {}):
        self.args = {}

        # Default init
        self._default_init()

        # Load default args
        _default_json_file = Path.cwd() / "EasyDatas.json"
        if _default_json_file.exists():
            with _default_json_file.open("r") as f:
                args = json.load(f)
            for key in args:
                self.args[key] = args[key]

        # Custom init
        self.custom_init()

        # Override args
        for key in override_args:
            self.args[key] = override_args[key]

        self.finished = False
        self.cache_root = self.args["cache_root"]
        self.need_cache = self.args["need_cache"]
        self.need_previous = self.args["need_previous"]

        self.previous : EasyDatasBase = None
        self.auto_get_idx = 0
        self.auto_put_idx = 0
        self.data_length = 0
        self.__datas = []
        self.class_id = self.__class__._class_count
        self.__class__._class_count += 1

        # Get cache file name
        self.cache_path : Path = Path(self.cache_root) / self.__get_cache_name(readable = self.args["readable"])

    def _default_init(self):
        """
        Hardcoded default args
        """
        self.args["readable"] = False
        self.args["need_cache"] = False
        self.args["need_previous"] = False
        self.args["cache_root"] = Path(__file__).parent

    def custom_init(self):
        """
        Override if needed.
        """
        pass

    def __get_cache_name(self, readable = False):
        """
        Format name_args dict to a cache file name. If readable, return a readable string.
        Class name will be at the head.
        """
        name_args = self.name_args()
        name_lists = []
        for key in name_args:
            value = name_args[key]
            if isinstance(value, list):
                value = tuple(value)
            assert isinstance(value,collections.Hashable), "Values for key {} in name args need to be hashable".format(key)
            name_lists.append((key,value))
        name_lists.sort(key = lambda x : x[0])
        name_str = ""
        for item in name_lists:
            name_str = name_str + "-" + str(item[0]) + "_" + str(item[1])

        if readable:
            return self.__class__.__name__ + str(self.class_id) + name_str + '.pkl'
        else:
            md5 = hashlib.md5()
            md5.update(name_str.encode("utf-8"))
            return self.__class__.__name__ + str(self.class_id) + "-" + str(md5.hexdigest()) + '.pkl'

    def name_args(self):
        """
        Return args dict for getting cache file's name
        Default to return all hashable values in self.args
        """
        ret = {}
        for key in self.args:
            if isinstance(self.args[key],collections.Hashable):
                if isinstance(self.args[key], str):
                    if "/" in self.args[key]:
                        continue
                ret[key] = self.args[key]
        return ret

    def deal_datas(self):
        """
        Deal all datas in one function
        """
        if self.need_previous:
            data = self.get()
            while data is not None:
                out_data = self.deal_a_data(data.copy())
                self.put(out_data)
                data = self.get()
        else:
            pass

    def deal_a_data(self,data):
        """
        Deal an item of datas. Default to return the input without any change.
        """
        if not "__warned_deal_a_data" in dir(self):
            print("[WARNING] The dataset {} is doing nothing. deal_a_data function should be override".format(self.__class__.__name__))
            self.__warned_deal_a_data = True
        return data

    def get(self,idx):
        assert self.previous is not None, "Dataset {} needs a previous dataset".format(self.__class__.__name__)
        return self.previous[idx]

    def get(self):
        """
        Automaticaly get a data. If all datas in previous dataset have been got, return None.
        """
        assert self.previous is not None, "Dataset {} needs a previous dataset".format(self.__class__.__name__)
        if self.auto_get_idx >= len(self.previous):
            return None
        self.auto_get_idx += 1
        return self.previous[self.auto_get_idx - 1]

    def put(self,idx,data_dict : dict):
        if idx >= self.data_length:
            self.__datas.extend([None] * (idx - self.data_length + 1))
        self.__datas[idx] = data_dict
        self.data_length = max(self.data_length,idx + 1)

    def put(self,data_dict : dict):
        if self.auto_put_idx >= len(self.__datas):
            if self.auto_put_idx == 0:
                self.__datas = [None]
            else:
                self.__datas.extend(self.__datas)
        self.__datas[self.auto_put_idx] = data_dict
        self.auto_put_idx += 1
        self.data_length = max(self.data_length,self.auto_put_idx)

    def clean_datas_overhead(self, force = False):
        """
        For time complexity, do nothing
        For memory effiency, delete overhead parts
        """
        if force:
            self.__datas = self.__datas[:self.data_length]

    def resolve(self):
        """
        Return true means finished, no need to access previous datasets.
        Return false means unfinished, need to access previous datasets.
        """
        if self.need_cache:
            have_cache = self._access_cache()
            if have_cache:
                self.finished = True
                return True

        if self.need_previous:
            return False
        else:
            return self._after_resolve()

    def _after_resolve(self):
        self.deal_datas()

        if self.need_cache:
            self._save_cache()
        
        self.finished = True
        return True

    def _access_cache(self):
        """
        Return True if have cache and loaded.
        """
        if self.cache_path.exists():
            with self.cache_path.open("rb") as f:
                self.__datas = pickle.load(f)
                self.data_length = len(self.__datas)
            return True
        return False

    def _save_cache(self):
        self.clean_datas_overhead(force = True)
        with self.cache_path.open("wb") as f:
            pickle.dump(self.__datas,f)

    def __len__(self):
        assert self.finished, "Please use {} after {}.resolve()".format(self.__class__.__name__,self.__class__.__name__)
        return self.data_length

    def __getitem__(self,idx):
        assert self.finished, "Please use {} after {}.resolve()".format(self.__class__.__name__,self.__class__.__name__)
        return self.__datas[idx]