
from utils.env_utils import EnvContext
from typing import Dict
import importlib

def get_obj(class_path, kwargs: Dict = None):
    parts = class_path.split('.')
    module_path = '.'.join(parts[:2])
    class_name = parts[-1]
    # 导入包含目标类的模块
    my_module = importlib.import_module(module_path)
    # 获取目标类
    my_class = getattr(my_module, class_name)
    # 创建新的类实例
    if kwargs is None:
        my_instance = my_class()
    else:
        my_instance = my_class(**kwargs)
    return my_instance

def get_svc_rd():
    obj = get_obj(class_path=EnvContext["SVC_MGR_SRSD_CLASS_PATH"])
    return obj

def get_svc_model_mapping():
    obj =  get_obj(class_path=EnvContext["SVC_MGR_MAPPING_CLASS_PATH"])
    return obj