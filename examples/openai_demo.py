import os
import openai

class Openai(object):

    @staticmethod
    def chat_complete(messages, stream=True):
        """
        Call OpenAI Chat Completion API with text prompt.
        """
        kwargs = {
            "model": "gpt-3.5-turbo",
            "messages": messages,
            "timeout": 5,
            "stream": stream,
            "presence_penalty": 1,
            # "max_tokens": 100,
            "temperature": 0.8,
        }
        try:
            response = openai.ChatCompletion.create(**kwargs)
            if stream:
                return response
            else:
                return response["choices"][0]["message"]['content']
        except Exception as e:
            print(f"OpenAI API error: {e}")


if __name__ == "__main__":
    messages = [
        {"role": "system", "content": "你是一个聊天助手"},
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好"},
        {"role": "user", "content": "你能做什么"},
    ]
    openai.api_base = "http://XX.XX.XX.XX:XXX/v1"
    openai.api_key = '*********'
    openai_obj = Openai()
    result = []
    resp = openai_obj.chat_complete(messages, stream=True)
    # print(resp)
    for r in resp:
        one = r.choices[0].delta.content if 'content' in r.choices[0].delta else ''
        print(one)
        result.append(one)