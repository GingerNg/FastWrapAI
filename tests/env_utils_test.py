import yaml
from utils.env_utils import flatten_dict
def test_env():
    pth = f'conf/env.yaml'
    with open(pth, 'r') as file:
        env_conf = yaml.safe_load(file)
        print(env_conf)
        print(flatten_dict(env_conf))