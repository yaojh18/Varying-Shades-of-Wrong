import collections
from concurrent.futures import ProcessPoolExecutor, as_completed


class LM(object):
    def __init__(self):
        self.model = None

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    def _generate(self, prompt, index, max_sequence_length=2048, max_output_length=128):
        raise NotImplementedError()

    def generate(self, prompts, max_sequence_length=2048, max_output_length=128):
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(self._generate, prompt.strip(), index, max_sequence_length, 1) if prompt.strip().endswith(" True or False?\nAnswer:") else
                executor.submit(self._generate, prompt.strip(), index, max_sequence_length, max_output_length)
                for index, prompt in enumerate(prompts)
            ]
            response_dict = collections.defaultdict(str)
            output_dict = collections.defaultdict(str)
            for job in as_completed(futures):
                index, output, res = job.result(timeout=None)
                response_dict[index] = res
                output_dict[index] = output

        return [output_dict[i] for i in range(len(prompts))], [response_dict[i] for i in range(len(prompts))]

    def save_cache(self):
        pass

    def load_cache(self, allow_retry=True):
        pass



