from typing import Dict, Iterable,List
from EasyDatas.Core.Base import EasyDatasBase
from abc import abstractmethod

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

    @abstractmethod
    def deal_a_data(self, data):
        return super().deal_a_data(data)

class CachedTransform(EasyDatasBase):
    def _default_init(self):
        super()._default_init()
        self.args["need_previous"] = True
        self.args["need_cache"] = True

    @abstractmethod
    def deal_a_data(self, data):
        return super().deal_a_data(data)

class Chain(EasyDatasBase):
    def __init__(self, chain : List[EasyDatasBase], args : dict = {}):
        super(Chain,self).__init__(args)
        self.chain : List[EasyDatasBase] = chain
        assert not self.need_cache, "Chain should not have cache."
        assert len(self.chain) > 0, "Empty chain was given to {}.".format(self.__class__.__name__)
        self.stop_idx = -1

    def chain_need_previous(self):
        def recurrent_resolve(idx : int, chain : List[EasyDatasBase]):
            if idx < 0:
                return True # need previous
            finished = chain[idx].resolve()
            if finished:
                self.stop_idx = idx # for _after_resolve
                return False # don't need previous
            else:
                return recurrent_resolve(idx - 1, chain)
        
        return recurrent_resolve(len(self.chain) - 1, self.chain)

    def resolve(self):
        """
        Return true means finished, no need to access previous datasets.
        Return false means unfinished, need to access previous datasets.
        """
        self.need_previous = self.chain_need_previous()
        return super().resolve()
    
    def deal_datas(self):
        if self.stop_idx == 0:
            self.chain[0].previous = self.previous
        for idx in range(self.stop_idx + 1,len(self.chain)):
            self.chain[idx].previous = self.chain[idx - 1]
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
        return self.merge_dicts([item[idx] for item in self.inputs])
