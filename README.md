# openai-api-wrapper
This is a simple Implemenation of OpenAI API Wrapper, which is based on FastAPI and Consul(As SRSD(Service Register and Service Discovery)).

The main purpose of this project is to provide a simple openai-protocol server to wrap other models or api to OpenAI style API.
## Launch
Fistly, you need to install the requirements, and configure the host and port of consul in the env.yaml file.
Then, You can run several workers to provide services for the api.
```
python3 openai_worker.py host port
```
openai_worker.py is a simple proxy worker of openai-api. Or, you can implement your own generator hosting your model or wrapping other api, and run the template_worker.py like this:
```
python3 template_worker.py host port generators.XXX.XxxGen
```
At last, start the openai procotol server, through this server, you can use your models or apis just like OpenAI.
```
python3 main_openai_wrapper_v2.py wrapper-server-host wrapper-server-port
```

## Usage
```
openai.api_base = "http://wrapper-server-host:wrapper-server-port/v1"
openai.api_key = '******'
```

## Implemented Apis
- /v1/chat/completions
- /v1/completions
- /v1/embeddings

## Architecture
Svc-Mgr(Service Management) component store the mapping relation among model(s), service(svc) and worker_address(work_addr)
![svc-model-mapping](docs/imgs/svc-model-mapping.png)
When launching the worker server, model(s), svc name and work_addr will registered into Svc-Mrg.
When calling the wrapper server by OpenAI package, server will find the corresponding svc and work_addr(service discovery and load balance) by Svc-Mgr.
![flow chart](docs/imgs/流程图-v0.1.0.png)



### Motivated By
- [Server-Sent Events using FastAPI, Starlette, and ReactJS.](https://github.com/harshitsinghai77/server-sent-events-using-fastapi-and-reactjs)
- [chatglm-openai-api - Provide OpenAI style API for ChatGLM-6B and Chinese Embeddings Model](https://github.com/ninehills/chatglm-openai-api)
- [api2d](https://api2d.com/)
- [FastChat: An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and FastChat-T5.](https://github.com/lm-sys/FastChat)
