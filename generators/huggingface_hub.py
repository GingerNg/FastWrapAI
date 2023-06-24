from common.protocol.worker_api_protocol import MultiModelGen
from utils.logger_utils import get_logger
logger = get_logger()
import os
import numpy as np
from common.protocol.openai_api_protocol import EmbeddingsItem
from common.protocol.worker_api_protocol import WorkerGeneratorPath, CommonVo, EmbeddingRet, CompletionRet, StreamCompletionRet, UsageInfo
from fastapi import HTTPException
from typing import List

import requests
hf_token = "XXXXXXX"

headers = {"Authorization": f"Bearer {hf_token}"}

class HfHubGen(MultiModelGen):
    """HfHub: Huggingface Hub
    """
    def __init__(self, **kwargs):
        super().__init__(models=["sentence-transformers/all-MiniLM-L6-v2"])

    def generate_embedding(self, params)->EmbeddingRet:
        """
        https://huggingface.co/blog/getting-started-with-embeddings
        """
        try:
            model_id = params.get("model")
            api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_id}"
            input_texts = params.get("input")
            if isinstance(input_texts, List):
                input_texts = [" ".join(input_texts)]
            response = requests.post(api_url, headers=headers, json={"inputs": input_texts, "options":{"wait_for_model":True}})
            embedding = response.json()[0]
            item = EmbeddingsItem(embedding=embedding)
            resp = EmbeddingRet(data=[item], usage_info=UsageInfo(), code=1)
            return resp
        except Exception as e:
            logger.error(e.__repr__())
            raise HTTPException(status_code=500, detail=e.__repr__())