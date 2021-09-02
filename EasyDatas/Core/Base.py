from os import name
from torch.utils.data import Dataset
import pickle
import json
from pathlib import Path
import collections
import hashlib
import logging
import copy

class EasyDatasBase(Dataset):
    _class_count = 0
    _global_hash_string = ""

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

        # Override args
        for key in override_args:
            self.args[key] = override_args[key]

        # Custom init
        self.custom_init()

        self.finished = False
        self.cache_root = Path(self.args["cache_root"])
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
        self._cache_str = None
    
    def get_attr(self, name, default):
        if not name in self.args:
            self.args[name] = default
        return self.args[name]

    @property
    def cache_str(self):
        if self._cache_str != None:
            return self._cache_str
        
        # Get self-determined hash str
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

        if self.args["readable"]:
            pass
        else:
            md5 = hashlib.md5()
            md5.update(name_str.encode("utf-8"))
            name_str = "-" + str(md5.hexdigest())

        name_str = self.__class__.__name__ + name_str

        # Get previous hash chain
        if self.need_previous:
            md5 = hashlib.md5()
            md5.update((self.previous.cache_str + name_str).encode("utf-8"))
            self._cache_str = name_str + "-" + str(md5.hexdigest())[:8]
        else:
            md5 = hashlib.md5()
            md5.update(name_str.encode("utf-8"))
            self._cache_str = name_str + "-" + str(md5.hexdigest())[:8]
        return self._cache_str

    def _default_init(self):
        """
        Hardcoded default args
        """
        self.args["readable"] = False
        self.args["need_cache"] = False
        self.args["need_previous"] = False
        self.args["cache_root"] = str(Path.cwd())

    def custom_init(self):
        """
        Override if needed.
        """
        pass

    def __get_cache_path(self):
        """
        A previous hash string will be at the end
        """
        return Path(self.cache_root) / (self.cache_str  + '.pkl')

    def name_args(self):
        """
        Return args dict for getting cache file's name
        Default to return all hashable values in self.args except cache_root
        """
        ret = {}
        for key in self.args:
            if isinstance(self.args[key],collections.Hashable):
                if key == "cache_root":
                    continue
                ret[key] = self.args[key]
                if isinstance(self.args[key], str):
                    if "/" in self.args[key]:
                        ret[key] = ret[key].replace("/","|")
        return ret

    def deal_datas(self):
        """
        Deal all datas in one function
        """
        if self.need_previous:
            data = self.get()
            while data is not None:
                out_data = self.deal_a_data(data)
                if out_data is not None:
                    self.put(out_data)
                data = self.get()
        else:
            pass

    def deal_a_data(self,data):
        """
        Deal an item of datas. Default to return the input without any change.
        """
        if not "__warned_deal_a_data" in dir(self):
            logging.warning("[WARNING] The dataset {} is doing nothing. deal_a_data function should be override".format(self.__class__.__name__))
            self.__warned_deal_a_data = True
        return data

    def get(self,idx = None,do_copy = True):
        """
        Automaticaly get a data. If all datas in previous dataset have been got, return None.
        """
        assert self.previous is not None, "Dataset {} needs a previous dataset".format(self.__class__.__name__)
        if idx != None:
            if do_copy:
                return copy.deepcopy(self.previous[idx])
            else:
                return self.previous[idx]
        if self.auto_get_idx >= len(self.previous):
            return None
        self.auto_get_idx += 1
        if do_copy:
            try:
                return copy.deepcopy(self.previous[self.auto_get_idx - 1])
            except Exception as e:
                print(self.previous[self.auto_get_idx - 1])
                print(e)
                exit(0)
        else:
            return self.previous[self.auto_get_idx - 1]

    def _put_idx(self,idx,data_dict : dict):
        if idx >= self.data_length:
            self.__datas.extend([None] * (idx - self.data_length + 1))
        self.__datas[idx] = data_dict
        self.data_length = max(self.data_length,idx + 1)

    def put(self,data_dict : dict,idx = -1):
        if idx != -1:
            return self._put_idx(idx,data_dict)
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
        logging.debug("Resolving class {}".format(self.__class__.__name__))
        if self.finished:
            return True # To not resolve an instance twice

        if self.need_cache:
            self.cache_path : Path = self.__get_cache_path()
            have_cache = self._access_cache()
            if have_cache:
                self.finished = True
                return True

        if self.need_previous:
            return False
        else:
            return self._after_resolve()

    def _after_resolve(self):
        logging.info("Dealing datas of {}".format(self.__class__.__name__))
        self.deal_datas()

        if self.need_cache:
            self._save_cache()
        
        self.finished = True
        return True

    def _access_cache(self):
        """
        Return True if have cache and loaded.
        """
        logging.info("Accessing cache file: {} for class {}".format(self.cache_path,self.__class__.__name__))
        if self.cache_path.exists():
            with self.cache_path.open("rb") as f:
                self.__datas = pickle.load(f)
                self.data_length = len(self.__datas)
                logging.info("Cache readed")
            return True
        logging.info("Cache not find")
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