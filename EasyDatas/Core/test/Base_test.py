from logging import log
import py
import pytest
from pathlib import Path
from torch.utils.data.dataset import T
from EasyDatas.Core import EasyDatasBase

@pytest.fixture
def mkdir(tmpdir_factory):
    cache_dir = tmpdir_factory.mktemp("./caches")
    return str(cache_dir)


@pytest.fixture(
    params = [
        {
            "need_cache" : True,
            "need_previous" : True
        },
        {
            "need_cache" : False,
            "need_previous" : True
        },
        {
            "need_cache" : True,
            "need_previous" : False
        },
        {
            "need_cache" : False,
            "need_previous" : False
        }
    ]
)
def EDB(request):
    args = request.param
    args["cache_root"] = "./caches"
    EDB = EasyDatasBase(
        override_args = args
    )
    yield EDB

@pytest.fixture
def EDB_D():
    # EDB default
    D = EasyDatasBase()
    D.resolve()
    return D

@pytest.fixture
def EDB_0():
    EDB = EasyDatasBase(
        override_args = {
            "cache_root" : "./caches",
            "need_previous" : False,
            "need_cache" : False,
            "type" : 0
        }
    )
    EDB.resolve()
    return EDB

@pytest.fixture
def EDB_10():
    EDB = EasyDatasBase(
        override_args = {
            "cache_root" : "./caches",
            "need_previous" : False,
            "need_cache" : False,
            "type" : 10
        }
    )
    EDB.resolve()
    EDB._EasyDatasBase__datas = [i for i in range(10)]
    EDB.data_length = 10
    return EDB

@pytest.fixture
def Pre_10(EDB_10):
    D = EasyDatasBase()
    D.previous = EDB_10
    D.resolve()
    return D

@pytest.fixture
def Pre_0(EDB_0):
    D = EasyDatasBase()
    D.previous = EDB_0
    D.resolve()
    return D

def test_default_init():
    EDB = EasyDatasBase()
    assert not EDB.need_cache
    assert not EDB.need_previous
    assert not EDB.args["readable"]

def test_name_args(EDB):
    EDB.args["hashable1"] = [1,2,3]
    EDB.args["hashable2"] = (1,2,3)
    EDB.args["unhashable"] = {1 : 1,2 : 2,3 : 3}
    EDB.args["path"] = "/abc/def/ghi.aedf"
    args = EDB.name_args()
    assert not "cache_root" in args
    assert not "unhashable" in args
    assert "hashable1" in args
    assert "hashable2" in args
    assert "path" in args
    assert "need_cache" in args
    assert "need_previous" in args
    assert args["hashable1"] == (1,2,3)
    assert args["hashable2"] == (1,2,3)
    assert args["path"] == "|abc|def|ghi.aedf"

def test_null_get():
    D = EasyDatasBase()
    D.resolve()
    with pytest.raises(AssertionError) as e:
        D.get()
    exec_msg = e.value.args[0]
    assert exec_msg == "Dataset EasyDatasBase needs a previous dataset"

def test_getitem(EDB_10):
    assert len(EDB_10) == 10
    assert EDB_10[0] == 0
    assert EDB_10[3] == 3
    assert EDB_10[6] == 6
    assert EDB_10[9] == 9
    with pytest.raises(IndexError) as e:
        t = EDB_10[10]

def test_get_0_default(Pre_0):
    D = Pre_0

    getted = D.get()
    assert getted is None
    getted = D.get(do_copy = False)
    assert getted is None

def test_get_0_idx(Pre_0):
    D = Pre_0

    with pytest.raises(IndexError) as e:
        getted = D.get(1)

    with pytest.raises(IndexError) as e:
        getted = D.get(1,do_copy = False)

def test_get_10_default(Pre_10):
    D = Pre_10

    getted = [D.get() for _ in range(12)]
    assert getted[:10] == [i for i in range(10)]
    assert getted[10] is None
    assert getted[11] is None

def test_get_10_idx(Pre_10):
    D = Pre_10
    for _ in range(7):
        D.get()
    
    t = D.get(5)
    assert t == 5
    with pytest.raises(IndexError) as e:
        t = D.get(11)
    with pytest.raises(IndexError) as e:
        t = D.get(11, do_copy = False)
    
    assert D.get() == 7

def test_get_10_copy():
    EDB = EasyDatasBase({"need_previous" : False})
    EDB.resolve()
    EDB._EasyDatasBase__datas = [[i] for i in range(10)]
    EDB.data_length = 10

    D = EasyDatasBase()
    D.previous = EDB
    D.resolve()

    t = D.get()
    assert t == [0]
    t[0] = 1
    assert EDB[0] == [0]

    t = D.get(1)
    assert t == [1]
    t[0] = 2
    assert EDB[1] == [1]

    t = D.get(do_copy = False)
    assert t == [1]
    t[0] = 3
    assert EDB[1] == [3]

    t = D.get(2, do_copy = False)
    assert t == [2]
    t[0] = 4
    assert EDB[2] == [4]

def test_put(EDB_D):
    for i in range(3):
        EDB_D.put(i)
    
    assert len(EDB_D) == 3
    assert EDB_D[0] == 0
    assert EDB_D[1] == 1
    with pytest.raises(IndexError) as e:
        t = EDB_D[4]
    exec_msg = e.value.args[0]
    assert exec_msg == "The given idx(4) is larger than the data length(3)"

    EDB_D.put(5,5)
    assert EDB_D[5] == 5
    assert len(EDB_D) == 6

    EDB_D.put(6,2)
    assert EDB_D[2] == 6

    EDB_D.put(7)
    assert EDB_D[3] == 7
    assert len(EDB_D) == 6

@pytest.mark.slow
def test_put_effi(EDB_D):
    import time
    t = time.time()
    for i in range(10000000):
        EDB_D.put(i)
    
    duration = time.time() - t
    print("\nPut operation for 1 million times costs {:.3f}s".format(duration))

def test_deal_datas(EDB_10, monkeypatch):
    
    def deal_a_data(self, data):
        data = data + 1
        return data
    
    D = EasyDatasBase({"need_previous" : True})
    D.previous = EDB_10
    monkeypatch.setattr(D, "deal_a_data", deal_a_data.__get__(D,EasyDatasBase))
    t = D.resolve()
    assert not t
    D._after_resolve()

    assert len(D) == 10
    assert D[7] == 8

def test_deal_datas_nopre(monkeypatch):
    
    def deal_a_data(self, data):
        assert False
        return data

    D = EasyDatasBase()
    monkeypatch.setattr(D, "deal_a_data", deal_a_data.__get__(D,EasyDatasBase))
    t = D.resolve()
    assert t

def test_cache(EDB_10, EDB_0, mkdir):
    D_1 = EasyDatasBase(
        {"need_cache" : True, "need_previous" : True, "cache_root" : mkdir, "test" : 0}
    )
    D_1.previous = EDB_10
    D_2 = EasyDatasBase(
        {"need_cache" : True, "need_previous" : True, "cache_root" : mkdir, "test" : 1}
    )
    D_2.previous = EDB_10
    D_3 = EasyDatasBase(
        {"need_cache" : True, "need_previous" : True, "cache_root" : mkdir, "test" : 0}
    )
    D_3.previous = EDB_0
    D_4 =  EasyDatasBase(
        {"need_cache" : True, "need_previous" : True, "cache_root" : mkdir, "test" : 0}
    )
    D_4.previous = EDB_10
    
    assert D_1.cache_str != D_2.cache_str
    assert D_1.cache_str != D_3.cache_str
    assert D_1.cache_str == D_4.cache_str

    def deal_datas(self):
        assert False, "Should not deal datas"

    D_4.deal_datas = deal_datas.__get__(D_4,EasyDatasBase)
    t = D_1.resolve()
    assert not t
    D_1._after_resolve()
    t = D_4.resolve()
    assert t
    assert len(D_1) == 10
    assert len(D_4) == 10
    
    # avoid resolve twice
    t = D_4.resolve()
    assert t
    t = D_1.resolve()
    assert t

def test_get_attr(EDB_D):
    t = EDB_D.get_attr("test_attr", 0)
    assert t == 0
    assert "test_attr" in EDB_D.args
    assert EDB_D.args["test_attr"] == 0
    t = EDB_D.get_attr("test_attr", 1)
    assert t == 0

def test_configs(mkdir):
    import json
    default_args = {
        "cache_root" : "default",
        "need_previous" : "default",
        "need_cache" : "default"
    }
    with open("./EasyDatas.json","w") as f:
        json.dump(default_args,f)

    class testD(EasyDatasBase):
        def custom_init(self):
            self.args["need_cache"] = "custom"
    
    D = testD({"need_previous" : "override", "need_cache" : "override"})
    assert not D.args["readable"]
    assert D.args["cache_root"] == "default"
    assert D.args["need_previous"] == "override"
    assert D.args["need_cache"] == "custom"

    Path("./EasyDatas.json").unlink()