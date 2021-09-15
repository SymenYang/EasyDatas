import multiprocessing
from typing import Dict, Iterable,List
from EasyDatas.Core.Base import EasyDatasBase
from abc import abstractmethod
from collections.abc import Hashable
import hashlib

class RawDatas(EasyDatasBase):
    def _default_init(self):
        super()._default_init()
        self.args["need_cache"] = True

    @abstractmethod
    def deal_datas(self):
        pass

class Transform(EasyDatasBase):
    def _default_init(self):
        super()._default_init()
        self.args["need_previous"] = True

    def deal_datas(self):
        assert self.previous is not None, "Dataset {} needs a previous dataset".format(self.__class__.__name__)
    
    def __len__(self):
        return len(self.previous)
    
    def __getitem__(self, idx):
        assert self.finished, "Please use {} after {}.resolve()".format(self.__class__.__name__,self.__class__.__name__)
        return self.deal_a_data(self.get(idx))

    @abstractmethod
    def deal_a_data(self, data):
        return super().deal_a_data(data)

class CachedTransform(EasyDatasBase):
    def _default_init(self):
        super()._default_init()
        self.args["need_previous"] = True
        self.args["need_cache"] = True
        self.args["del_previous"] = True

    def name_args(self):
        args = super().name_args()
        args.pop("del_previous")
        return args

    @abstractmethod
    def deal_a_data(self, data):
        return super().deal_a_data(data)
    
    def _save_cache(self, clean = True):
        if self.args["del_previous"]:
            self.previous._EasyDatasBase__datas = []
        return super()._save_cache(clean=clean)

class ParallelCachedTransform(CachedTransform):
    def _default_init(self):
        import multiprocessing
        self.args["workers"] = self.get_attr("workers",multiprocessing.cpu_count() // 2)
        return super()._default_init()        
        
    def name_args(self):
        args = super().name_args()
        args.pop("workers")
        return args

    def start_dealing(self, iter, start, length, rank):
        pass

    def after_dealing(self, iter, start, length, rank):
        pass

    def deal_a_part(self, start, length, rank):
        self.rank = rank
        self.auto_get_idx = start
        for i in range(length):
            data = self.get()
            self.start_dealing(start + i, start, length, rank)
            out_data = self.deal_a_data(data)
            self.after_dealing(start + i, start, length, rank)
            if out_data is not None:
                self.put(out_data)
        return self._EasyDatasBase__datas[:self.data_length]

    def deal_datas(self, backend : str = "multiprocessing"):
        from joblib import Parallel, delayed
        self.workers = self.args["workers"]
        length = len(self.previous)
        if self.workers > length:
            self.workers = length
        step = length // self.workers
        steps = [step] * self.workers
        steps[-1] += length - sum(steps)
        datas = Parallel(n_jobs = self.workers,backend=backend)(
            delayed(self.deal_a_part)(
                step * worker,
                steps[worker],
                worker) for worker in range(self.workers)
        )
        import itertools
        self._EasyDatasBase__datas = list(itertools.chain(*datas))
        self.data_length = len(self._EasyDatasBase__datas)
    
    @abstractmethod
    def deal_a_data(self, data):
        return super().deal_a_data(data)

class Chain(EasyDatasBase):
    def __init__(self, chain : List[EasyDatasBase], args : dict = {}):
        super(Chain,self).__init__(args)
        self.chain : List[EasyDatasBase] = chain
        assert not self.need_cache, "Chain should not have cache."
        assert len(self.chain) > 0, "Empty chain was given to {}.".format(self.__class__.__name__)
        for idx in range(1,len(self.chain)):
            self.chain[idx].previous = self.chain[idx - 1]
        self.stop_idx = -1

        self._previous : EasyDatasBase = None
    
    @property
    def previous(self):
        return self._previous
    
    @previous.setter
    def previous(self,previous : EasyDatasBase):
        self._previous = previous
        if len(self.chain) > 0 and self.chain[0].need_previous:
            self.chain[0].previous = previous
    
    @property
    def cache_str(self):
        if self._cache_str != None:
            return self._cache_str
        
        self._cache_str = self.chain[-1].cache_str
        return self._cache_str

    def chain_need_previous(self):
        def recursion_resolve(idx : int, chain : List[EasyDatasBase]):
            if idx < 0:
                return True # need previous
            finished = chain[idx].resolve()
            if finished:
                self.stop_idx = idx # for _after_resolve
                return False # don't need previous
            else:
                return recursion_resolve(idx - 1, chain)
        
        return recursion_resolve(len(self.chain) - 1, self.chain)

    def resolve(self):
        """
        Return true means finished, no need to access previous datasets.
        Return false means unfinished, need to access previous datasets.
        """
        self.need_previous = self.chain_need_previous()
        return super().resolve()
    
    def deal_datas(self):
        for idx in range(self.stop_idx + 1,len(self.chain)):
            self.chain[idx]._after_resolve()
    
    def __len__(self):
        assert self.finished, "Please use {} after {}.resolve()".format(self.__class__.__name__,self.__class__.__name__)
        return len(self.chain[-1])

    def __getitem__(self,idx):
        assert self.finished, "Please use {} after {}.resolve()".format(self.__class__.__name__,self.__class__.__name__)
        return self.chain[-1][idx]

class Merge(EasyDatasBase):
    def __init__(self, inputs : List[EasyDatasBase], args : dict = {}):
        super(Merge,self).__init__(args)
        self.inputs = inputs
        assert len(self.inputs) > 1, "{} needs more than one input".format(self.__class__.__name__)
        assert not self.need_previous, "{} shold not need previous dataset".format(self.__class__.__name__)
        assert not self.need_cache, "{} should not have cache.".format(self.__class__.__name__)
    
    @property
    def cache_str(self):
        if self._cache_str != None:
            return self._cache_str
        
        # Get self-determined hash str
        name_args = self.name_args()
        name_lists = []
        for key in name_args:
            value = name_args[key]
            assert isinstance(value,Hashable), "Values for key {} in name args need to be hashable".format(key)
            name_lists.append((str(key),value))
        name_lists.sort(key = lambda x : x[0])
        name_str = ""
        for item in name_lists:
            name_str = name_str + "-" + str(item[0]) + "_" + str(item[1])

        md5 = hashlib.md5()
        md5.update(name_str.encode("utf-8"))
        name_str = "-" + str(md5.hexdigest())

        name_str = self.__class__.__name__ + name_str

        # Get previous hash chain
        md5 = hashlib.md5()
        md5.update("".join([item.cache_str for item in self.inputs]).encode("utf-8"))
        self._cache_str = name_str + "-" + str(md5.hexdigest())[:8]
        
        return self._cache_str

    def resolve(self):
        finished = True
        for dataset in self.inputs:
            finished = finished and dataset.resolve()
        assert finished, "{} shold not need previous dataset".format(self.__class__.__name__)
        self.data_length = len(self.inputs[0])
        for item in self.inputs:
            assert self.data_length == len(item), "All inputs need to be same length in {}".format(self.__class__.__name__)

        return self._after_resolve()
    
    def merge_dicts(self,dict_list : List[Dict]):
        """
        Given any number of dicts, shallow copy and merge into a new dict,
        precedence goes to key value pairs in latter dicts.
        """
        result = {}
        for dictionary in dict_list:
            result.update(dictionary)
        return result
    
    def __getitem__(self, idx):
        assert self.finished, "Please use {} after {}.resolve()".format(self.__class__.__name__,self.__class__.__name__)
        return self.merge_dicts([item[idx] for item in self.inputs])

class Stack(Merge):
    def __init__(self, inputs : List[EasyDatasBase], args : dict = {}):
        super(Merge,self).__init__(args)
        self.inputs = inputs
        assert len(self.inputs) > 1, "{} needs more than one input".format(self.__class__.__name__)
        assert not self.need_previous, "{} shold not need previous dataset".format(self.__class__.__name__)
        assert not self.need_cache, "{} should not have cache.".format(self.__class__.__name__)
    
    def resolve(self):
        finished = True
        for dataset in self.inputs:
            finished = finished and dataset.resolve()
        assert finished, "{} shold not need previous dataset".format(self.__class__.__name__)
        self.data_length = 0
        self.data_lengthes = [0]
        for item in self.inputs:
            self.data_length += len(item)
            self.data_lengthes.append(self.data_length)

        return self._after_resolve()
    
    def __getitem__(self,idx):
        assert self.finished, "Please use {} after {}.resolve()".format(self.__class__.__name__,self.__class__.__name__)
        for input_id in range(len(self.data_lengthes)):
            if idx >= self.data_lengthes[input_id] and idx < self.data_lengthes[input_id + 1]:
                return self.inputs[input_id][idx - self.data_lengthes[input_id]]