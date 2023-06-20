#pip3 install python-consul
import consul
import random
from utils.env_utils import EnvContext
ConsulHost = EnvContext.get("SRSD_CONSUL_HOST")
ConsulPort = EnvContext.get("SRSD_CONSUL_PORT")
class SvcRgstDscvy(object):
    """
    Service Registration(Rgst) and Service Discovery(Dscvy)
    """
    def __init__(self, host=ConsulHost, port=ConsulPort):
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

if __name__ == '__main__':
    consul_client=SvcRgstDscvy(host=ConsulHost,port=ConsulPort)
    res=consul_client.get_svc("openai-worker")
    print(res)