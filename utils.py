import time
import collections
import openai
import random
import torch
import torch.nn.functional as F
from torch.utils.data import random_split
from tqdm import tqdm
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
from typing import List

OPENAI_KEY = 'sk-proj-BJiODPrgTGI8keUENjdmT3BlbkFJCAQOWpDkd50EHnvp2ILL'
idx2letter = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]


class PreferenceDataset:
    dataset: List[dict]
    dataset_name: str
    model_name: str
    split_ratio: float = 0.8

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def precess_dataset(self):
        pass

    @abstractmethod
    def save_dataset(self):
        pass

    def generate_answer(self, instruction_name):
        queries = []
        for data in self.train_dataset:
            queries.append(data['query'])
        with open(f'./instruction/{instruction_name}.txt', encoding='utf-8') as f:
            instruction = ''.join(f.readlines())
        if self.model_name == 'gpt-4':
            log_probs, responses = batch_query_openai(queries, instruction=instruction, model='gpt-4o')
        elif self.model_name == 'llama-3':
            log_probs, responses = batch_query_open_sourced_llm(queries, instruction=instruction,
                                                                model_name='meta-llama/Meta-Llama-3-8B-Instruct')
        else:
            raise NotImplementedError
        for data, log_prob, response in zip(self.train_dataset, log_probs, responses):
            data['log_probs'] = log_prob
            data['responses'] = response
        return log_probs, responses

    def train_test_split(self):
        train_dataset_size = round(len(self.dataset) * self.split_ratio)
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_dataset_size, len(self.dataset) - train_dataset_size])
        self.train_dataset = list(self.train_dataset)
        self.test_dataset = list(self.test_dataset)

    @abstractmethod
    def extract_answer(self):
        pass


def query_openai(prompt, index, model):
    client = openai.OpenAI(api_key=OPENAI_KEY)
    generate_kwargs = {
        "top_p": 0.8,
        "temperature": 1.0,
        "logprobs": True,
        "n": 10
    }
    retry_count = 100
    retry_interval = 10
    responses = []
    log_probs = []

    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                **generate_kwargs
            )
            for choice in response.choices:
                responses.append(choice.message.content)
                normalized_log_prob = []
                for log_prob in choice.logprobs.content:
                    normalized_log_prob.append(-log_prob.logprob)
                log_probs.append(sum(normalized_log_prob) / len(normalized_log_prob))
            return index, log_probs, responses

        except Exception as e:
            print("Error info: ", e)
            print('Retrying....')
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response for prompt: ', prompt)
    return index, [0.0] * generate_kwargs['n'], [''] * generate_kwargs['n']


def batch_query_openai(prompt_list, instruction, model="gpt-3.5-turbo"):
    with ProcessPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(query_openai, prompt + instruction, index, model) for index, prompt in
                   enumerate(prompt_list)]
        response_dict = collections.defaultdict(str)
        log_prob_dict = collections.defaultdict(str)
        for job in tqdm(as_completed(futures), total=len(futures), desc="querying openai..."):
            index, log_prob, res = job.result(timeout=None)
            response_dict[index] = res
            log_prob_dict[index] = log_prob

    return [log_prob_dict[i] for i in range(len(prompt_list))], [response_dict[i] for i in range(len(prompt_list))]


def batch_query_open_sourced_llm(prompt_list, instruction, model_name):
    """
    :param batch_size:
    :param model_name:
    :param prompt_list:
    :param instruction:
    :return:
    """
    login(token='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')
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize all instructions and answers
    generate_kwargs = {
        "do_sample": True,
        "top_p": 0.8,
        "temperature": 1.0,
        "max_new_tokens": 1024,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "num_return_sequences": 10,
        "return_dict_in_generate": True,
        "output_logits": True,
    }

    # Get the log probabilities from the model
    log_probs = []
    responses = []
    with torch.no_grad():
        for i in tqdm(range(len(prompt_list)), desc="generating answers..."):
            # query_tokens = tokenizer([prompt_list[i] + instruction], padding=True)
            query_tokens = tokenizer.apply_chat_template(
                [[{'role': 'user', 'content': prompt_list[i] + instruction}]],
                add_generation_prompt=True,
                padding=True,
                return_dict=True
            )
            input_ids = torch.LongTensor(query_tokens["input_ids"]).to(model.device)
            attention_mask = torch.LongTensor(query_tokens["attention_mask"]).to(model.device)
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)

            sequences = outputs.sequences[:, input_ids.shape[-1]:].cpu()
            logits = [logit.cpu() for logit in outputs.logits]
            log_prob = -F.log_softmax(torch.stack(logits, dim=1), dim=-1)
            answer_log_prob = log_prob.gather(-1, sequences[:, :, None]).squeeze(-1)
            texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            normalized_log_prob = []
            for j in range(generate_kwargs["num_return_sequences"]):
                normalized_log_prob.append(
                    torch.masked_select(answer_log_prob[j, :], sequences[j, :] != tokenizer.eos_token_id).mean().item()
                )
            log_probs.append(normalized_log_prob)
            responses.append(texts)

    return log_probs, responses


def generate_choices_from_candidates(options, correct_answers):
    def get_random_choice(correct_count):
        correct_indices = random.sample(range(3), correct_count)
        result = []
        for i in range(3):
            if i in correct_indices:
                result.append(correct_answers[i])
            else:
                incorrect_options = [opt for opt in options[f'blank {i + 1}'] if opt != correct_answers[i]]
                incorrect_option = random.choice(incorrect_options)
                result.append(incorrect_option)
        return result

    choices = []
    for correct_count in range(4):
        choice = get_random_choice(correct_count)
        choices.append((choice, correct_count))
    random.shuffle(choices)
    return [choice for choice, _ in choices], [count for _, count in choices]


def get_normalized_probabilities(instructions, answers, model_name="meta-llama/Meta-Llama-3-8B", batch_size=2):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)

    tokenizer.pad_token = tokenizer.eos_token
    concat_queries = [instruction + tokenizer.bos_token + answer for instruction, answer in zip(instructions, answers)]

    # Tokenize all instructions and answers
    instruction_tokens = tokenizer(instructions, return_tensors="pt", padding=True)
    answer_tokens = tokenizer(answers, return_tensors="pt", padding=True)
    query_tokens = tokenizer(concat_queries, return_tensors="pt", padding=True)

    # Get the log probabilities from the model
    probabilities = []
    with torch.no_grad():
        for begin in tqdm(range(0, len(instructions), batch_size), desc="generating probability..."):
            end = begin + batch_size if begin + batch_size < len(instructions) else len(instructions)
            input_ids = query_tokens["input_ids"][begin: end].to(model.device)
            attention_mask = query_tokens["attention_mask"][begin: end].to(model.device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
            log_probs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
            for i in range(min(end - begin, batch_size)):
                instruction_len = instruction_tokens["attention_mask"][i + begin].sum()
                answer_len = answer_tokens["attention_mask"][i + begin].sum()

                # Get log probabilities for the answer tokens
                answer_log_probs = log_probs[i, instruction_len: instruction_len + answer_len - 1, :]
                answer_log_probs = answer_log_probs.gather(-1, answer_tokens["input_ids"][begin + i, 1: answer_len, None].to(model.device)).squeeze(-1)

                # Calculate the log probability of the entire answer
                normalized_prob = answer_log_probs.mean().cpu().item()
                probabilities.append(normalized_prob)

    return probabilities
