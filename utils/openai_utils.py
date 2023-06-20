import os
import openai
from .env_utils import EnvContext, EnvKeys

# openai.api_base = EnvContext.get(EnvKeys.OpenAIApiBase)
openai.api_key = EnvContext.get(EnvKeys.OpenAIApikey)

from utils.logger_utils import get_logger
logger = get_logger()

class Openai(object):

    @staticmethod
    def complete(params, stream=True):
        kwargs = {
            # "model": model,
            # "prompt": messages,
            # "timeout": 5,
            "stream": stream,
            "presence_penalty": 1,
            # "max_tokens": 100,
            "temperature": 0.8
        }
        kwargs.update(params)
        print(f"kwargs:{kwargs}")
        try:
            response = openai.Completion.create(**kwargs)
            print(response)
            if stream:
                return response
            else:
                return response
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {e}")

    @staticmethod
    def chat_complete(params, stream=True):
        """
        Call OpenAI Chat Completion API with text prompt.
        """
        kwargs = {
            # "model": model,
            # "messages": messages,
            "timeout": 5,
            "stream": stream,
            "presence_penalty": 1,
            # "max_tokens": 100,
            "temperature": 0.8
        }
        kwargs.update(params)
        print(f"kwargs:{kwargs}")
        try:
            response = openai.ChatCompletion.create(**kwargs)
            if stream:
                return response
            else:
                return response["choices"][0]["message"]
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"OpenAI API error: {e}")


# *******************************************************
# https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
import tiktoken
def num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301"):
    """Returns the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model == "gpt-3.5-turbo":
        print("Warning: gpt-3.5-turbo may change over time. Returning num tokens assuming gpt-3.5-turbo-0301.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0301")
    elif model == "gpt-4":
        print("Warning: gpt-4 may change over time. Returning num tokens assuming gpt-4-0314.")
        return num_tokens_from_messages(messages, model="gpt-4-0314")
    elif model == "gpt-3.5-turbo-0301":
        tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
        tokens_per_name = -1  # if there's a name, the role is omitted
    elif model == "gpt-4-0314":
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens.""")
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens

# ******************************************************* openai_protocol *******************************************************