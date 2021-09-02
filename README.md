# EasyDatas
An easy way to build PyTorch datasets. Modularly build data sets and automatically cache processed results

## Installation
```bash
pip install git+https://github.com/SymenYang/EasyDatas
```

## Usage
### Find files in disk
```python
from EasyDatas.Prefab import ListFile, RecursionFiles, SpanFiles
from EasyDatas.Prefab import Chain

# Type 1: Find files recursively
# Example:
RFiles = RecursionFiles({
    "path" : path_to_root,
    "pattern" : ".*\.npy",
    "files" : True, # default to be true
    "dirs" : False # default to be true
})
RFiles.resolve()
print(len(RFiles)) # Total num of npy files in path_to_root
print(RFiles[0]) # {"path" : "/xxxx/xxxx/xxxx.npy"(pathlib.Path object)}

# Or Type 2: Hierarchically find files
HFiles = Chain([
    ListFile({
        "path" : path_to_root,
        "pattern" : ".*",
        "files" : False, # default to be true
    }),
    SpanFiles({
        "pattern" : ".*\.npy"
        "dirs" : False # default to be true
    })
])
HFiles.resolve()
print(len(HFiles)) # Total num of npy in files in path_to_root's depth-one sub-dir
print(HFiles[0]) # {"path" : "path_to_root/xxxx/xxxx.npy"(pathlib.Path object)}
```
ListFile, RecursionFiles, SpanFiles will output files/dirs in the dictionary order 

### Load files to memory
```python
from EasyDatas.Prefab import LoadData, NumpyLoad,NumpyLoadNPY
#Type 1: use numpy.load to load a npy format file
LoadChain = Chain([
    RFiles, # defined in the previous section. Or any other EasyDatas module providing path
    NumpyLoadNPY({
        "data_name" : "data" # default to be "data"
    })
])
LoadChain.resolve()
print(len(loadChain)) # The same with RFiles
print(LoadChain[0]) # {"data" : np.ndarray}

# Type 2: write your own codes to load
import numpy as np
LoadChainCustom = Chain([
    HFiles,
    LoadData({
        "data_name" : "custom_data" # default to be "data"
        },
        function = lambda x : np.loadtxt(str(x))
    )
])
LoadChainCustom.resolve()
print(len(LoadChainCustom)) # The same with HFiles
print(LoadChainCustom[0]) # {"custom_data" : np.ndarray}

# The custom LoadData could be replaced by NumpyLoad module.
```

### Preprocessing
```python
from EasyDatas.Prefab import Picker, ToTensor
from EasyDatas.Core import Transform, CachedTransform

class customTransform1(CachedTransform): 
    # Cached Transform will process all datas and cache the results in disk.
    def custom_init(self):
        self.times = self.get_attr("times", 2) # default value is 2

    def deal_a_data(self, data : dict):
        data["data"] = data["data"] * self.times
        return data


class customTransform2(Transform): 
    # Non-cached transform will process a data when it is been needed.
    def deal_a_data(self, data : dict):
        data["data"] = data["data"] + 1
        return data


TrainDataset = Chain([
    LoadChain,
    Picker(
        pick_func = lambda data,idx,total_num : idx <= 0.8 * total_num
    ),
    customTransform1({
        "times" : 3
    }),
    customTransform1(),
    customTransform2(),
    ToTensor()
])
TrainDataset.resolve()
print(len(TrainDataset)) # 0.8 * len(LoadChain)
print(TrainDataset[0]) # {"data" : torch.Tensor with (raw data * 3 * 2 + 1) }

# Or we can write all of them in one chain and only resolve once
TrainDataset = Chain([
    RecursionFiles({
        "path" : path_to_root,
        "pattern" : ".*\.npy",
        "dirs" : False # default to be true
    }),
    NumpyLoadNPY({
        "data_name" : "data" # default to be "data"
    }),
    Picker(
        pick_func = lambda data,idx,total_num : idx <= 0.8 * total_num
    ),
    customTransform1({
        "times" : 3
    }),
    customTransform1(),
    customTransform2(),
    ToTensor()
])
TrainDataset.resolve()
print(len(TrainDataset)) # 0.8 * len(LoadChain)
print(TrainDataset[0]) # {"data" : torch.Tensor with (raw data * 3 * 2 + 1) }
```
**All EasyDatas modules are the child of torch.utils.data.Dataset. Thus we can directly put them into a dataloader**

## About caches
An EasyDatas module will store caches only if the `args["need_cache"]` is True. The defualt setting is False. Cache will be save in the `args["cache_root"]` path, which is set to CWD in default. The cache name will contain two parts. The first is about the module's args when it was created, the second is about the module's previous modules cache name. All the information are encoded to a string and EasyDatas will use that string to determine whether there is a valid cache for this module instance. Therefore, if one module's args have been changed, all modules' cache after this module will be recomputed.
### Custom cache name
One can override `name_args(self)` function to change the properties that need to be considerd into cache name. The default implementation is:
```python
class EasyDatasBase
    ...
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
        return ret
    ...
```

## Processing Datas
All EasyDatas module have two functions to deal datas. The first is `deal_datas` and the second is `deal_a_data`. In default, `deal_datas` will send all datas to `deal_a_data` one-by-one and collect the return value as the output of this module. In most situation, customizing `deal_a_data` is safe, clear and enough. But in some other situation, we want to deal all datas by our own, we could override `deal_datas` function. There are two useful functions in `EasyDatasBase` class that will be helpful in `deal_datas`, which are `self.get()`and `self.put()`
```python
class EasyDatasBases:
    def get(self,idx = None,do_copy = True) -> dict|None:
        pass

    def put(self,data_dict : dict,idx = -1) -> None:
        pass
```
If idx is not provided, `get` will automaticaly get datas from previous module one-by-one. If it meets the end, it will return None. A module with no previous module could not use `get` function. If the `do_copy` is set to False, it will directly return previous module's data, which is a reference. Otherwise, it will deep copy the data and return.  
`put` function will automaticaly put datas in to return and cache list. if idx is provided, the `data_dict` will be put in to the position. The total number of datas will be count automaticaly in `put` function.  
Besides, in `deal_a_data` function, one can use `put` functions and return None for increasing the data items. 

## Other modules
There are some other modules that are not introduced beyond. 
### EasyDatas.Core.EasyDatasBase
Defined base functions, logging and default processing

### EasyDatas.Core.RawDatas
Base class for ListFile, RecursionFiles. RawDatas needs no previous dataset and the `deal_datas` function needs to be overrided

### EasyDatas.Core.Merge
Merge multiple EasyDatas modules by merge their data dict. The modules need to have the same length.
```python
# assume A is an EasyDatas module with A[0] == {"data_1" : xxx}
# assume B is an EasyDatas module with B[0] == {"data_2" : xxx}
M = Merge([A,B])
print(len(M)) # The same with A and B
print(M[0]) # {"data_1" : xxx, "data_2" : xxx}
```

### EasyDatas.Core.Stack
Stack multiple EasyDatas modules by combine their items.
```python
# assume A is an EasyDatas module with A[0] == {"data_1" : xxx} and len(A) = 1000
# assume B is an EasyDatas module with B[0] == {"data_2" : xxx} and len(B) = 500
S = Stack([A,B])
print(len(S)) # 1500 which is len(A) + len(B)
print(S[999]) # {"data_1" : xxx}
print(S[1000]) # {"data_2" : xxx}
```
In most cases, Stack are used to stack modules which have same data format.
