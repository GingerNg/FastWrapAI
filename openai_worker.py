import uvicorn
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from sse_starlette.sse import EventSourceResponse
from utils.env_utils import EnvKeys, EnvContext, app

import json
from typing import List, Optional, Dict

from dataclasses import dataclass
from typing import List
from utils.openai_utils import Openai, num_tokens_from_messages
model_semaphore = None
import asyncio
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

def generate_stream_response_start():
    return {
        "id": "chatcmpl-77QWpn5cxFi9sVMw56DZReDiGKmcB",
        "object": "chat.completion.chunk", "created": 1682004627,
        "model": "gpt-3.5-turbo-0301",
        "choices": [{"delta": {"role": "assistant"}, "index": 0, "finish_reason": None}]
    }

openai_obj = Openai()

def eval_llm(params):
    if params.get("path", None) is None:
        return openai_obj.chat_complete(params, stream=False)
    else:
        del params["path"]
        return openai_obj.complete(params, stream=False)

import time
def eval_llm_stream(params: Dict):
    """
    """
    # tokens = []
    # time.sleep(20)
    if params.get("path", None) is None:
        first = True
        ensure_ascii = False
        for response in openai_obj.chat_complete(params):
            # one = response.choices[0].delta.content if 'content' in response.choices[0].delta else ''
            # if first:
            #     first = False
            #     yield json.dumps(generate_stream_response_start(),
            #                     ensure_ascii=ensure_ascii)
            # tokens.append(one)
            yield json.dumps(response, ensure_ascii=ensure_ascii)
        # yield json.dumps(generate_stream_response_stop(), ensure_ascii=ensure_ascii)

        # token counter
        # messages.append({"role": "system", "content": "".join(tokens)})
        # num_tokens = num_tokens_from_messages(messages=messages, model=model)
        # await sender.send(str(num_tokens))
        # yield '[DONE]'
    else:
        ensure_ascii = False
        for response in openai_obj.complete(params):
            yield json.dumps(response, ensure_ascii=ensure_ascii)

@app.post("/worker_generate_completion")
async def api_generate_completion(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    completion = eval_llm(params)
    background_tasks = create_background_tasks()
    return JSONResponse(content=completion, background=background_tasks)

@app.post("/worker_generate_completion_stream")
async def api_generate_completion_stream(request: Request):
    params = await request.json()
    await acquire_model_semaphore()
    generator = eval_llm_stream(params)
    background_tasks = create_background_tasks()
    return StreamingResponse(generator, background=background_tasks)

from common.svc_rgst_dscvy import SvcRgstDscvy
import sys
def usage():
    """
    print usage message and exit.
    """
    print('Usage: {} svc-host svc-port svc-name(optional)'.format(sys.argv[0]))
    sys.exit(1)
if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
    svc_port = int(sys.argv[2])
    host=sys.argv[1]
    svc_rd = SvcRgstDscvy()
    if len(sys.argv) == 4:
        name = sys.argv[3]
    else:
        name = "openai-worker"
    svc_rd.register_svc(name, f"{name}-{host}-{svc_port}",host, svc_port)
    uvicorn.run(app, host="0.0.0.0", port=svc_port)

