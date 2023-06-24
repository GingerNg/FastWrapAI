from enum import Enum
from typing import Dict, Optional, List
from fastapi import BackgroundTasks
from pydantic import BaseModel

class WorkerGeneratorPath(Enum):
    TEXT_COMPLETION_STREAM = "/worker_generate_completion_stream"
    TEXT_COMPLETION = "/worker_generate_completion"
    TEXT_EMBEDDING = "/worker_generate_embedding"

# VO
class CommonVo(BaseModel):
    payload: Dict
    path: str

# ----------------------------------
class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0

class EmbeddingRet(BaseModel):
    data: List[Dict]
    usage_info: UsageInfo
    code: int = 0

class CompletionRet(BaseModel):
    data: List[Dict]
    usage_info: UsageInfo
    code: int = 0

class StreamCompletionRet(BaseModel):
    data: List[Dict]
    code: int = 0


# ---------------------------------- scaffold of worker-svc ----------------------------------
import sys
def usage():
    """
    print usage message and exit.
    """
    print('Usage: {} svc-host svc-port'.format(sys.argv[0]))
    sys.exit(1)

# SRSD
from common.factory import get_svc_rd
svc_rd = get_svc_rd()

# 信号量
import asyncio
model_semaphore = None
def release_model_semaphore():
    model_semaphore.release()


def acquire_model_semaphore():
    global model_semaphore
    if model_semaphore is None:
        model_semaphore = asyncio.Semaphore(2)
    return model_semaphore.acquire()

def create_background_tasks():
    background_tasks = BackgroundTasks()
    background_tasks.add_task(release_model_semaphore)
    return background_tasks

# ----------------------------------

class GenBase:
    def __init__(self):
        pass

    def generate_completion(self, params):
        raise NotImplementedError

    def generate_completion_stream(self, params):
        raise NotImplementedError

    def generate_embedding(self, params):
        raise NotImplementedError

class SingleModelGen(GenBase):
    """SingleModelGen: set svc_name as model_name"""
    def __init__(self, models: List):
        self.supported_models = models
        self.svc_name = models[0]

class MultiModelGen(GenBase):
    """MultiModelGen: set svc_name as class_name"""
    def __init__(self, models: List):
        self.supported_models = models
        self.svc_name = self.__class__.__name__

# ----------------------------------
class Mapping(BaseModel):
    mapping: Dict
    def __setitem__(self, key, value):
        self.mapping[key] = value

    def __getitem__(self, key):
        return self.mapping[key]

    def get(self, key, default_val=None):
        return self.mapping.get(key, default_val)

    def __repr__(self):
        return self.mapping

class ModelSvcMapping(Mapping):
    def __init__(self, model_svc_mapping):
        super().__init__(mapping=model_svc_mapping)

class SvcModelMapping(Mapping):
    def __init__(self, svc_model_mapping):
        super().__init__(mapping=svc_model_mapping)

    def reverse(self) -> ModelSvcMapping:
        return ModelSvcMapping(
            {model: svc for svc, models in self.mapping.items() for model in models}
            )



