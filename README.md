# openai-api-wrapper
This is a simple Implemenation of OpenAI API, which is based on FastAPI and Consul(As SRSD(Service Register and Service Discovery)).
## Usage
Fistly, you need to install the requirements, and configure the host and port of consul in the env.yaml file.
```
python3 openai_worker.py host port
```
You can run several workers to provide services for the api.
```
python3 main_openai_wrapper_v2.py host port
```

## Implemented Apis
- /v1/chat/completions
- /v1/completions

### Motivated By
- [Server-Sent Events using FastAPI, Starlette, and ReactJS.](https://github.com/harshitsinghai77/server-sent-events-using-fastapi-and-reactjs)
- [chatglm-openai-api - Provide OpenAI style API for ChatGLM-6B and Chinese Embeddings Model](https://github.com/ninehills/chatglm-openai-api)
- [api2d](https://api2d.com/)
- [FastChat: An open platform for training, serving, and evaluating large language models. Release repo for Vicuna and FastChat-T5.](https://github.com/lm-sys/FastChat)
