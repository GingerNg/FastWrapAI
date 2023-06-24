#pip3 install python-consul
import consul
import random
from utils.env_utils import EnvContext
from typing import List
import json
import fcntl
class AbstractSvcMgr(object):
    """
    Env center
    Service Registration(Rgst) and Service Discovery(Dscvy)
    """
    def __init__(self) -> None:
        pass

    def register_svc(self):
        pass

    def get_svc(self, name: str):
        """
        name: svc-name
        """
        pass

    def load(self):
        pass

    def update(self, models: List, svc_name: str):
        pass

    def _validate(svc_name: str, models: List, svc2model_mapping: SvcModelMapping) -> bool:
        """
        validate svc-name and models by comparing with svc2model_mapping
        """
        exists_models = []
        msg = ""
        model2svc_mapping = svc2model_mapping.reverse()
        # check svc-name 是否已经存在
        if svc_name in svc2model_mapping:
            # svc-name已经存在，则判断当前的models和新的models是否相同
            registed_models = svc2model_mapping.get(svc_name)
            if set(registed_models) == set(models):
                return True, msg
            else:
                msg = "服务已经存在，但是models和当前models不一致，需要人工check"
                return False, msg
        else:
            # svc-name不存在, 则check model的唯一性
            for model in models:
                if model in model2svc_mapping:
                    exists_models.append(model)
            if len(exists_models) > 0:
                msg = "服务不存在，但是model已经被注册，需要人工check"
                return False, msg
            else:
                return True, msg

# ********* env center for SvcModelMapping *********

class ConsulSvcModelMapping(AbstractSvcMgr):
    def __init__(self) -> None:
        super().__init__()

from common.protocol.worker_api_protocol import SvcModelMapping
class FileSvcModelMapping(AbstractSvcMgr):
    def __init__(self) -> None:
        super().__init__()
        self.mapping_path = EnvContext["SVC_MGR_MAPPING_FILR_PATH"]

    def load(self):
        svc2model_mapping = SvcModelMapping(json.load(open(self.mapping_path, "r")))
        model2svc_mapping = svc2model_mapping.reverse()
        print(f"model2svc_mapping: {model2svc_mapping}")
        # SVC2Model = json.load(open(self.mapping_path, "r"))
        return model2svc_mapping

    def update(self, models: List, svc_name: str):
        try:
            # 打开文件
            f_rw = open(self.mapping_path, "rw")
            # 获取文件锁
            fcntl.flock(f_rw, fcntl.LOCK_EX)
            svc2model_mapping = SvcModelMapping(json.load(f_rw))
            valid_pass, valid_msg = self._validate(svc_name, models, svc2model_mapping)
            if valid_pass:
                svc2model_mapping[svc_name] = models
                json.dump(svc2model_mapping.dict(), f_rw)
            else:
                raise Exception(valid_msg)
        except Exception as e:
            raise Exception(e.__repr__())
        finally:
            f_rw.close()
            fcntl.flock(f_rw, fcntl.LOCK_UN)

# ********* SRSD *********

class ConsulSvcRgstDscvy(AbstractSvcMgr):
    """
    """
    def __init__(self, host=None, port=None):
        if host is None or port is None:
            host = EnvContext.get("SVC_MGR_SRSD_CONSUL_HOST")
            port = EnvContext.get("SVC_MGR_SRSD_CONSUL_PORT")
        '''初始化，连接consul服务器'''
        self._consul = consul.Consul(host, port)

    def register_svc(self, name, service_id, host, port, tags=None):
        tags = tags or []
        # 注册服务
        self._consul.agent.service.register(
            name,
            service_id,
            host,
            port,
            tags,
            # 健康检查ip端口，检查时间：5,超时时间：30，注销时间：30s
            check=consul.Check().tcp(host, port, "5s", "30s", "30s"))

    def get_svc(self, name):
        endpoint = None
        _, service_list = self._consul.catalog.service(name)
        # print(service_list)
        # 遍历服务实例列表，选择健康状态良好的服务进行调用
        endpoints = ['http://{0}:{1}'.format(service['ServiceAddress'], service["ServicePort"]) for service in service_list]
        # print(endpoints)
        if len(endpoints)>0:
            endpoint = random.choice(endpoints)
        return endpoint


class DevSvcRgstDscvy(AbstractSvcMgr):
    def get_svc(self, name):
        return "http://127.0.0.1:XXXX"


if __name__ == '__main__':
    consul_client=ConsulSvcRgstDscvy()

    res=consul_client.get_svc("openai-worker")
    print(res)