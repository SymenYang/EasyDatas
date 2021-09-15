import py
import pytest
from pathlib import Path
from torch.utils.data.dataset import T
from EasyDatas.Core.FunctionalDatasets import *
import json

@pytest.fixture(
    autouse = True
)
def mkdir(tmpdir_factory):
    cache_dir = tmpdir_factory.mktemp("./caches")
    with open("./EasyDatas.json","w") as f:
        json.dump({
            "cache_root" : str(cache_dir)
        }, f)
    yield str(cache_dir)

    Path("./EasyDatas.json").unlink()


class customRaw(RawDatas):
    def custom_init(self):
        super().custom_init()
        self.length = self.get_attr("length", 10)

    def deal_datas(self):
        for i in range(self.length):
            self.put({'data' : i})

class cusromRaw2(customRaw):
    def deal_datas(self):
        for i in range(self.length):
            self.put({'data2' : i})

def get_raw(length, two = False):
    raw = customRaw({"length" : length})
    if two:
        raw = cusromRaw2({"length" : length})
    raw.resolve()
    return raw

class CustomTransform(Transform):
    def deal_a_data(self,data):
        data["data"] += 1
        return data

class CustomTransformCached(CachedTransform):
    def deal_a_data(self,data):
        data["data"] += 1
        return data

class CustomTransformParallel(ParallelCachedTransform):
    def deal_a_data(self,data):
        data["data"] += 1
        return data

@pytest.mark.parametrize(
    'class_type',
    [
        CustomTransform,
        CustomTransformCached,
        CustomTransformParallel  
    ],
)
@pytest.mark.parametrize(
    'length',
    [1,10,100],
)
def test_transform_basic(class_type, length):
    T = class_type()
    T.previous = get_raw(length)
    T.resolve()
    T._after_resolve()

    assert len(T) == length
    for i in range(0,length,10):
        assert T[i]['data'] == i + 1
    with pytest.raises(IndexError) as e:
        t = T[length]
    exec_msg = e.value.args[0]
    assert exec_msg == "The given idx({}) is larger than the data length({})".format(length,length)

@pytest.mark.parametrize(
    'class_type',
    [
        CustomTransformCached,
        CustomTransformParallel  
    ],
)
@pytest.mark.parametrize(
    'length',
    [1,10,100],
)
def test_transform_cache(class_type, length):
    T = class_type()
    T.previous = get_raw(length)
    T.resolve()
    T._after_resolve()

    def deal_datas(self):
        assert False, "Should not call this"

    T = class_type()
    T.deal_datas = deal_datas.__get__(T,class_type)
    T.previous = get_raw(length)
    t = T.resolve()
    assert t
    assert len(T) == length
    for i in range(0,length,10):
        assert T[i]['data'] == i + 1
    with pytest.raises(IndexError) as e:
        t = T[length]
    exec_msg = e.value.args[0]
    assert exec_msg == "The given idx({}) is larger than the data length({})".format(length,length)

@pytest.mark.parametrize(
    'length',
    [1,10,100],
)
def test_chain_basic(length):
    C = Chain([
        get_raw(length),
        CustomTransform()
    ])
    C.resolve() 
    
    assert len(C) == length
    for i in range(0,length,10):
        assert C[i]['data'] == i + 1
    with pytest.raises(IndexError) as e:
        t = C[length]
    exec_msg = e.value.args[0]
    assert exec_msg == "The given idx({}) is larger than the data length({})".format(length,length)

def test_chain_seq():
    length = 10
    C1 = Chain([
        get_raw(length),
        CustomTransform()
    ])
    C2 = Chain([
        CustomTransformCached(),
        CustomTransformParallel()
    ])
    C3 = Chain([
        CustomTransform(),
        CustomTransform(),
        CustomTransformCached()
    ])
    C = Chain([C1,C2])
    C.resolve()
    assert len(C) == length
    for i in range(0,length,10):
        assert C[i]['data'] == i + 3
    with pytest.raises(IndexError) as e:
        t = C[length]
    exec_msg = e.value.args[0]
    assert exec_msg == "The given idx({}) is larger than the data length({})".format(length,length)

    C = Chain([C1,C3])
    C.resolve()
    assert len(C) == length
    for i in range(0,length,10):
        assert C[i]['data'] == i + 4
    with pytest.raises(IndexError) as e:
        t = C[length]
    exec_msg = e.value.args[0]
    assert exec_msg == "The given idx({}) is larger than the data length({})".format(length,length)

def test_chain_cache():
    T1 = CustomTransformCached()
    T2 = CustomTransformCached()
    T3 = CustomTransform()

    C = Chain([
        get_raw(10),
        T1,
        T2,
        T3
    ])
    C.resolve()

    def deal_datas(self):
        assert False, "should not deal datas here"
    
    T1 = CustomTransformCached()
    T2 = CustomTransformCached()
    T3 = CustomTransform()
    T1.deal_datas = deal_datas.__get__(T1,CustomTransformCached)
    C = Chain([
        get_raw(10),
        T1,
        T2,
        T3
    ])
    C.resolve()
    assert len(C) == 10
    for i in range(0,10):
        assert C[i]['data'] == i + 3
    with pytest.raises(IndexError) as e:
        t = C[10]
    exec_msg = e.value.args[0]
    assert exec_msg == "The given idx({}) is larger than the data length({})".format(10,10)

def test_merge_basic():
    raw_1 = get_raw(10)
    raw_2 = get_raw(10, True)
    M = Merge([
        raw_1, raw_2
    ])
    M.resolve()
    
    assert len(M) == 10
    for i in range(0, 10):
        assert M[i]['data'] == i
        assert M[i]['data2'] == i

def test_merge_fails():
    raw_1 = get_raw(10)
    raw_2 = get_raw(20, True)
    with pytest.raises(AssertionError) as e:
        M = Merge([raw_1])
        M.resolve()
    assert e.value.args[0] == "Merge needs more than one input"

    M = Merge([raw_1,raw_2])
    with pytest.raises(AssertionError) as e:
        M.resolve()
    assert e.value.args[0] == "All inputs need to be same length in Merge"

def test_merge_cache():
    raw_1 = get_raw(20)
    raw_2 = get_raw(20, True)

    M = Merge([raw_1,raw_2])
    C = Chain([M, CustomTransformCached()])
    C.resolve()
    tmp = C.cache_str

    M = Merge([raw_1,raw_2])
    C = Chain([M, CustomTransformCached()])
    C.resolve()
    assert C.cache_str == tmp

    M = Merge([raw_1,raw_1])
    C = Chain([M, CustomTransformCached()])
    C.resolve()

    assert C.cache_str != tmp

def test_stack_basic():
    raw_1 = get_raw(10)
    raw_2 = get_raw(10, True)
    S = Stack([raw_1,raw_2])
    S.resolve()

    assert len(S) == 20
    for i in range(10):
        assert "data" in S[i]
        assert not "data2" in S[i]
    for i in range(10,20):
        assert "data2" in S[i]
        assert not "data" in S[i]
