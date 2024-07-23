import time
import json
import os
import collections
import openai
import random
import torch
import re
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
letter2idx = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15,
    'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23,
    'Y': 24, 'Z': 25, 'None': None
}


class PreferenceDataset:

    dataset_name: str
    model_name: str
    dataset: List[dict] = []
    output_name: str = ''
    split_ratio: float = 0.8
    clean_extracted_answer_pattern: str = ''

    def __init__(self, dataset_name, model_name, dataset_sample_size=-1, response_sample_size=10, load_from_exist=False):
        random.seed(42)
        self.dataset_sample_size = dataset_sample_size
        self.response_sample_size = response_sample_size
        self.dataset_name = dataset_name
        self.model_name = model_name
        assert (self.output_name != '')
        if load_from_exist and os.path.exists(f'./output/{self.model_name}/{self.output_name}.jsonl') and os.path.exists(f'./output/{self.model_name}/{self.output_name}_test.jsonl'):
            self.train_dataset = []
            self.test_dataset = []
            with open(f'./output/{self.model_name}/{self.output_name}.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    self.train_dataset.append(json.loads(line.strip()))
            with open(f'./output/{self.model_name}/{self.output_name}_test.jsonl', 'r', encoding='utf-8') as file:
                for line in file:
                    self.test_dataset.append(json.loads(line.strip()))
        else:
            self.load_dataset()
            self.precess_dataset(sample_size=self.dataset_sample_size)
            self.train_test_split()

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def precess_dataset(self, sample_size):
        pass

    def train_test_split(self):
        train_dataset_size = round(len(self.dataset) * self.split_ratio)
        self.train_dataset, self.test_dataset = random_split(self.dataset, [train_dataset_size, len(self.dataset) - train_dataset_size])
        self.train_dataset = list(self.train_dataset)
        self.test_dataset = list(self.test_dataset)

    def generate_answer(self, instruction_name):
        with open(f'./instruction/{instruction_name}.txt', encoding='utf-8') as f:
            instruction = ''.join(f.readlines())
        queries = []
        for data in self.train_dataset:
            if 'choices' in data:
                for choice in data['choices']:
                    queries.append([{'role': 'user', 'content': instruction + data['query'] + choice}])
            else:
                queries += [[{'role': 'user', 'content': instruction + data['query']}] for _ in range(self.response_sample_size)]
        if self.model_name == 'gpt-4':
            log_probs, responses = batch_query_openai(queries, model_name='gpt-4o')
        elif self.model_name == 'gpt-3.5':
            log_probs, responses = batch_query_openai(queries, model_name='gpt-3.5-turbo')
        elif self.model_name == 'llama-3':
            log_probs, responses = batch_query_open_sourced_llm(queries, model_name='meta-llama/Meta-Llama-3-8B-Instruct')
        else:
            raise NotImplementedError
        for i, data in enumerate(self.train_dataset):
            data['log_probs'] = log_probs[i * self.response_sample_size: i * self.response_sample_size + self.response_sample_size]
            data['responses'] = responses[i * self.response_sample_size: i * self.response_sample_size + self.response_sample_size]

    def process_answer(self, instruction_name, extract_instruction_name):
        with open(f'./instruction/{instruction_name}.txt', encoding='utf-8') as f:
            instruction = ''.join(f.readlines())
        with open(f'./instruction/{extract_instruction_name}.txt', encoding='utf-8') as f:
            extract_instruction = ''.join(f.readlines())
        queries = []
        for data in self.train_dataset:
            if 'choices' in data:
                for choice, response in zip(data['choices'], data['responses']):
                    queries.append([
                        {'role': 'user', 'content': instruction + data['query'] + choice},
                        {'role': 'assistant', 'content': response},
                        {'role': 'user', 'content': extract_instruction}
                    ])
            else:
                for response in data['responses']:
                    queries.append([
                        {'role': 'user', 'content': instruction + data['query']},
                        {'role': 'assistant', 'content': response},
                        {'role': 'user', 'content': extract_instruction}
                    ])
        if self.model_name == 'gpt-4':
            _, responses = batch_query_openai(queries, model_name='gpt-4o', mode='extract')
        elif self.model_name == 'gpt-3.5':
            _, responses = batch_query_openai(queries, model_name='gpt-3.5-turbo', mode='extract')
        elif self.model_name == 'llama-3':
            _, responses = batch_query_open_sourced_llm(queries, model_name='meta-llama/Meta-Llama-3-8B-Instruct', mode='extract')
        else:
            raise NotImplementedError

        for i, data in enumerate(self.train_dataset):
            data['extracted answers'] = responses[i * self.response_sample_size: i * self.response_sample_size + self.response_sample_size]

    def save_dataset(self):
        os.makedirs(f'./output/{self.model_name}/', exist_ok=True)
        with open(f'./output/{self.model_name}/{self.output_name}.jsonl', 'w', encoding='utf-8') as file:
            for data in self.train_dataset:
                file.write(json.dumps(data) + '\n')
        with open(f'./output/{self.model_name}/{self.output_name}_test.jsonl', 'w', encoding='utf-8') as file:
            for data in self.test_dataset:
                file.write(json.dumps(data) + '\n')


def query_openai(prompt, index, model_name, mode):
    client = openai.OpenAI(api_key=OPENAI_KEY)
    if mode == 'generate':
        generate_kwargs = {
            "temperature": 1.0,
            "logprobs": True,
        }
    elif mode == 'evaluate':
        generate_kwargs = {
            "temperature": 0.0,
            # TODO
            # "logprobs": True,
        }
    else:
        generate_kwargs = {
            "temperature": 0.0,
            "max_tokens": 10,
        }
    retry_count = 100
    retry_interval = 10

    for _ in range(retry_count):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=prompt,
                **generate_kwargs
            )
            msg = response.choices[0].message.content
            log_prob = 0.0
            if mode == 'generate':
                log_prob = []
                for prob in response.choices[0].logprobs.content:
                    if -prob.logprob != 9999.0:
                        log_prob.append(-prob.logprob)
                log_prob = sum(log_prob) / len(log_prob)
            # TODO
            # elif mode == 'evaluate':
            #     log_prob = []
            #     for prob in response.choices[0].logprobs.content:
            #         log_prob.append((prob.token, prob.logprob))
            return index, log_prob, msg

        except Exception as e:
            print("Error info: ", e)
            print('Retrying....')
            retry_count += 1
            retry_interval *= 2
            time.sleep(retry_interval)
    print('Fail to get response for prompt: ', prompt)
    return index, 0.0, ''


def batch_query_openai(prompt_list, model_name="gpt-3.5-turbo", mode='generate'):
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(query_openai, prompt, index, model_name, mode) for index, prompt in
                   enumerate(prompt_list)]
        response_dict = collections.defaultdict(str)
        log_prob_dict = collections.defaultdict(str)
        for job in tqdm(as_completed(futures), total=len(futures), desc="querying openai..."):
            index, log_prob, res = job.result(timeout=None)
            response_dict[index] = res
            log_prob_dict[index] = log_prob

    return [log_prob_dict[i] for i in range(len(prompt_list))], [response_dict[i] for i in range(len(prompt_list))]


def batch_query_open_sourced_llm(prompt_list, model_name, mode='generate'):
    """
    :param mode:
    :param model_name:
    :param prompt_list:
    :return:
    """
    login(token='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')
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='balanced_low_0', torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize all instructions and answers
    if mode == 'generate':
        generate_kwargs = {
            "do_sample": True,
            "top_p": 1.0,
            "temperature": 1.0,
            "max_new_tokens": 1024,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "return_dict_in_generate": True,
            "output_logits": True,
        }
        batch_size = 5
    elif mode == 'evaluate':
        generate_kwargs = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 1024,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            # TODO
            # "return_dict_in_generate": True,
            # "output_logits": True,
        }
        batch_size = 1
    else:
        generate_kwargs = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 20,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        batch_size = 1
    # Get the log probabilities from the model_name
    log_probs = []
    responses = []
    with torch.no_grad():
        for i in tqdm(range(0, len(prompt_list), batch_size), desc="generating answers..."):
            begin = i
            end = min(i + batch_size, len(prompt_list))
            query_tokens = tokenizer.apply_chat_template(
                prompt_list[begin: end],
                add_generation_prompt=True,
                padding=True,
            )
            input_ids = torch.LongTensor(query_tokens).to("cuda")
            outputs = model.generate(input_ids=input_ids, **generate_kwargs)

            if mode == 'generate':
                sequences = outputs.sequences[:, input_ids.shape[-1]:].cpu()
                logits = [logit.cpu() for logit in outputs.logits]
                log_prob = -F.log_softmax(torch.stack(logits, dim=1), dim=-1)
                answer_log_prob = log_prob.gather(-1, sequences[:, :, None]).squeeze(-1)
                for j in range(end - begin):
                    log_probs.append(
                        answer_log_prob[j, :][sequences[j, :] != tokenizer.eos_token_id].mean().item()
                    )
            # TODO
            # elif mode == 'evaluate':
            #     sequences = outputs.sequences[:, input_ids.shape[-1]:].cpu()
            #     logits = [logit.cpu() for logit in outputs.logits]
            #     log_prob = -F.log_softmax(torch.stack(logits, dim=1), dim=-1)
            #     answer_log_prob = log_prob.gather(-1, sequences[:, :, None]).squeeze(-1)
            #     for j in range(end - begin):
            #         l = answer_log_prob[j, :][sequences[j, :] != tokenizer.eos_token_id].item()
            #         t = tokenizer.convert_ids_to_tokens(sequences[j, :][sequences[j, :] != tokenizer.eos_token_id])
            #         lt_pair = []
            #         for l_, t_ in zip(l, t):
            #             lt_pair.append((t_, l_))
            #         log_probs.append(lt_pair)
            else:
                sequences = outputs[:, input_ids.shape[-1]:].cpu()
            texts = tokenizer.batch_decode(sequences, skip_special_tokens=True)
            responses += texts

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
    return [choice for choice, _ in choices], [count / 3.0 for _, count in choices]


def calculate_similarity_by_ada(question, answers, correct_answer_index):
    client = openai.OpenAI(api_key=OPENAI_KEY)
    embeddings = []
    for answer in answers:
        # query = f'Question: {question}\nAnswer: {answer}\n'
        embeddings.append(client.embeddings.create(input=answer, model="text-embedding-3-small").data[0].embedding)
    embeddings = torch.Tensor(embeddings)
    correct_embedding = embeddings[correct_answer_index, None]
    cosine_similarity = F.cosine_similarity(embeddings, correct_embedding)
    return cosine_similarity.tolist()


def get_normalized_probabilities(instructions, answers, model_name="meta-llama/Meta-Llama-3-8B", batch_size=2):
    # Load the model_name and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto', load_in_8bit=True)

    tokenizer.pad_token = tokenizer.eos_token
    concat_queries = [instruction + tokenizer.bos_token + answer for instruction, answer in zip(instructions, answers)]

    # Tokenize all instructions and answers
    instruction_tokens = tokenizer(instructions, return_tensors="pt", padding=True)
    answer_tokens = tokenizer(answers, return_tensors="pt", padding=True)
    query_tokens = tokenizer(concat_queries, return_tensors="pt", padding=True)

    # Get the log probabilities from the model_name
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


def clean_extracted_answers(dataset, pattern=r'([A-Z])(\.|\. .+)?$'):
    pattern = re.compile(pattern)
    for data in dataset.train_dataset:
        new_extracted_answers = []
        for d in data['extracted answers']:
            match = pattern.search(d)
            if match:
                result = match.group(1)
                if result.isdigit():
                    result = int(result)
                else:
                    result = letter2idx[result]
            else:
                result = None
            new_extracted_answers.append(result)
        data['extracted answers'] = new_extracted_answers
