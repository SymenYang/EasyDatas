from EasyDatas.Core import EasyDatasBase
import inspect
import hashlib

class Picker(EasyDatasBase):
    def __init__(self, override_args: dict = {}, pick_func = lambda x,id,l : True):
        super().__init__(override_args=override_args)
        self.pick_func = pick_func
        self.look_up_table = []
        assert self.need_previous, "{} shold have a previous dataset".format(self.__class__.__name__)

    def name_args(self):
        args = super().name_args()
        md5 = hashlib.md5()
        lines = inspect.getsourcelines(self.pick_func)[0]
        for lid in range(len(lines)):
            lines[lid] = lines[lid].strip()
        md5.update("\n".join(lines).encode("utf-8"))
        args["func"] = str(md5.hexdigest())
        return args

    def _default_init(self):
        super()._default_init()
        self.args["need_previous"] = True
        self.args["need_cache"] = True
    
    def deal_datas(self):
        length = len(self.previous)
        for i in range(length):
            if self.pick_func(self.previous[i],i,length):
                self.put({"idx" : i})
    
    def __getitem__(self, idx):
        assert self.finished, "Please use {} after {}.resolve()".format(self.__class__.__name__,self.__class__.__name__)
        return self.previous[super().__getitem__(idx)["idx"]]