import time

import numpy as np
import openai

from factscore.lm import LM


class OpenAIModel(LM):

    def __init__(self, model_name, key_path="api.key"):
        self.model_name = model_name
        self.key_path = key_path
        self.temp = 0.0
        self.save_interval = 100
        super().__init__()

    def load_model(self):
        # load api key
        openai.api_key = self.key_path
        self.model = self.model_name

    def _generate(self, prompt, index, max_sequence_length=2048, max_output_length=128):
        message = [{"role": "user", "content": prompt}]
        response = call_ChatGPT(message, self.key_path, temp=self.temp, max_len=max_sequence_length)
        output = response.choices[0].message.content
        return index, output, response


# TODO: make it parallelized
def call_ChatGPT(
        message,
        openai_key,
        model_name="gpt-3.5-turbo",
        max_len=1024,
        temp=0.0
):
    response = None
    received = False
    num_rate_errors = 0
    while not received:
        try:
            client = openai.OpenAI(api_key=openai_key)
            response = client.chat.completions.create(model=model_name,
                                                      messages=message,
                                                      max_tokens=max_len,
                                                      temperature=temp)
            received = True
        except Exception as e:
            print(e)
            num_rate_errors += 1
            if num_rate_errors >= 5:
                raise RuntimeError('Not able to get response from GPT-3.5')
            time.sleep(np.power(2, num_rate_errors))
    return response
