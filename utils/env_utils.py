from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ***********************************************

from immutabledict import immutabledict
import yaml
import os
root_dir = os.path.dirname(os.path.dirname(__file__))

from enum import Enum
class EnvKeys(Enum):
    OpenAIApiBase = "OPENAI_API_BASE"
    OpenAIApikey = "OPENAI_API_KEY"
    ServerPort = "SERVER_PORT"

flatted_ = {}
def flatten_dict(d, parent_key='', sep='_'):
    """flatten nested dict"""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key.upper(), sep=sep).items())
        else:
            items.append((new_key.upper(), v))
    return dict(items)

EnvContext = None

def init_env(flag="prod"):
    if flag == "dev":
        pth = f'{root_dir}/conf/env.dev.yaml'
    else:
        pth = f'{root_dir}/conf/env.yaml'
    with open(pth, 'r') as file:
        env_conf = yaml.safe_load(file)
    global EnvContext
    EnvContext = immutabledict(flatten_dict(env_conf))
    print("load conf success!")

init_env()





