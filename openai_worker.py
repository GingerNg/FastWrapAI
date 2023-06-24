import uvicorn
from fastapi import FastAPI, HTTPException, Request, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json
from typing import List, Optional, Dict
from dataclasses import asdict, dataclass
from typing import List

from sse_starlette.sse import EventSourceResponse
from utils.env_utils import EnvKeys, EnvContext, app
from utils.openai_utils import Openai, num_tokens_from_messages
from utils.logger_utils import get_logger
from common.protocol.worker_api_protocol import WorkerGeneratorPath, CommonVo, EmbeddingRet, CompletionRet, StreamCompletionRet, UsageInfo
from common.protocol.openai_api_protocol import OpenaiGeneratorPath
from common.protocol.worker_api_protocol import release_model_semaphore, acquire_model_semaphore, create_background_tasks
from common.protocol.worker_api_protocol import usage, svc_rd
import sys
logger = get_logger()

openai_obj = Openai()

def eval_llm(params):
    try:
        path = params.get("path", None)
        if path is not None:
                del params["path"]
        if path is None or path == "/v1/chat/completions":
            resp =  openai_obj.chat_complete(params, stream=False)
            return CompletionRet(data=resp["choices"],
                                usage_info=UsageInfo(**resp["usage"]),
                                code=1)
        elif path == OpenaiGeneratorPath.TEXT_COMPLETION.value:
            resp =  openai_obj.complete(params, stream=False)
            return CompletionRet(data=resp["choices"],
                                usage_info=UsageInfo(**resp["usage"]),
                                code=1)
        elif path == OpenaiGeneratorPath.TEXT_EMBEDDING.value:
            resp = openai_obj.embedding(params)
            return EmbeddingRet(data=resp["data"],
                                usage_info=UsageInfo(**resp["usage"]),
                                code=1)
        else:
            raise HTTPException(status_code=500, detail="path not supported")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=e.__repr__())

def eval_llm_stream(params: Dict):
    """
    stream
    """
    ensure_ascii = False
    try:
        path = params.get("path", None)
        if path is not None:
                del params["path"]
        if path is None or path == OpenaiGeneratorPath.TEXT_CHAT_COMPLETION.value:
            for response in openai_obj.chat_complete(params):
                logger.debug(response)
                ret = StreamCompletionRet(data=response["choices"], code=1)
                yield json.dumps(ret.dict(), ensure_ascii=ensure_ascii)
                # yield json.dumps(response, ensure_ascii=ensure_ascii)
        elif path == OpenaiGeneratorPath.TEXT_COMPLETION.value:
            for response in openai_obj.complete(params):
                ret = StreamCompletionRet(data=response["choices"], code=1)
                yield json.dumps(ret.dict(), ensure_ascii=ensure_ascii)
                # yield json.dumps(response, ensure_ascii=ensure_ascii)
        else:
            raise HTTPException(status_code=500, detail="path not supported")
    except Exception as e:
        logger.error(e)
        raise HTTPException(status_code=500, detail=e.__repr__())


@app.post(WorkerGeneratorPath.TEXT_COMPLETION_STREAM.value)
async def api_generate_completion_stream(request: Request):
    try:
        params = await request.json()
        await acquire_model_semaphore()
        generator = eval_llm_stream(params)
        background_tasks = create_background_tasks()
        return StreamingResponse(generator, background=background_tasks)
    except Exception as e:
        logger.error(e)
        release_model_semaphore()
        raise HTTPException(status_code=500, detail=e.__repr__())

@app.post(WorkerGeneratorPath.TEXT_COMPLETION.value)
async def api_generate_completion(request: Request):
    try:
        params = await request.json()
        await acquire_model_semaphore()
        completion = eval_llm(params)
        background_tasks = create_background_tasks()
        return JSONResponse(content=completion.dict(), background=background_tasks)
    except Exception as e:
        logger.error(e)
        release_model_semaphore()
        raise HTTPException(status_code=500, detail=e.__repr__())

@app.post(WorkerGeneratorPath.TEXT_EMBEDDING.value)
async def api_generate_embedding(request: Request):
    try:
        params = await request.json()
        await acquire_model_semaphore()
        resp = eval_llm(params)
        background_tasks = create_background_tasks()
        return JSONResponse(content=resp.dict(), background=background_tasks)
    except Exception as e:
        logger.error(e.__repr__())
        release_model_semaphore()
        raise HTTPException(status_code=500, detail=e.__repr__())


if __name__ == '__main__':
    if len(sys.argv) < 3:
        usage()
    svc_port = int(sys.argv[2])
    host=sys.argv[1]
    if len(sys.argv) == 4:
        name = sys.argv[3]
    else:
        name = "openai-worker"
    svc_rd.register_svc(name, f"{name}-{host}-{svc_port}",host, svc_port)
    uvicorn.run(app, host="0.0.0.0", port=svc_port)

