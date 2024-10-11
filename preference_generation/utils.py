import time
import collections
import openai
import random
import torch
import re
import torch.nn.functional as F
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM, T5EncoderModel
from peft import PeftModel


OPENAI_KEY = ''
HF_KEY = ''
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
        }
    else:
        generate_kwargs = {
            "temperature": 0.0,
            "max_tokens": 32,
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


def batch_query_open_sourced_llm(prompt_list, model_name, peft_dir=None, mode='generate'):
    """
    :param peft_dir:
    :param mode:
    :param model_name:
    :param prompt_list:
    :return:
    """
    if peft_dir is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map='auto',
            torch_dtype=torch.bfloat16,
            token=HF_KEY
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            peft_dir,
            device_map='balanced_low_0',
            torch_dtype=torch.bfloat16,
            token=HF_KEY,
        )
        model = PeftModel.from_pretrained(model, peft_dir)
        tokenizer = AutoTokenizer.from_pretrained(peft_dir)
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

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
    elif mode.find('evaluate') >= 0:
        _, max_new_tokens = mode.split('_')
        generate_kwargs = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": int(max_new_tokens),
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        batch_size = 5 if int(max_new_tokens) == 256 else 1
    else:  # mode = 'extract'
        generate_kwargs = {
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "max_new_tokens": 32,
            "pad_token_id": tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
        batch_size = 5
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
                return_tensors="pt",
                return_dict=True
            )
            input_ids = query_tokens['input_ids'].to('cuda')
            attention_mask = query_tokens['attention_mask'].to('cuda')
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, **generate_kwargs)

            if mode == 'generate':
                sequences = outputs.sequences[:, input_ids.shape[-1]:].cpu()
                logits = [logit.cpu() for logit in outputs.logits]
                log_prob = -F.log_softmax(torch.stack(logits, dim=1), dim=-1)
                answer_log_prob = log_prob.gather(-1, sequences[:, :, None]).squeeze(-1)
                for j in range(end - begin):
                    log_probs.append(
                        answer_log_prob[j, :][sequences[j, :] != tokenizer.eos_token_id].mean().item()
                    )
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


class Vera:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('liujch1998/vera')
        self.model = T5EncoderModel.from_pretrained(
            'liujch1998/vera',
            device_map='auto',
            torch_dtype=torch.bfloat16
        )
        self.model.D = self.model.shared.embedding_dim
        self.linear = torch.nn.Linear(self.model.D, 1, dtype=self.model.dtype)
        self.linear.weight = torch.nn.Parameter(self.model.shared.weight[32099, :].unsqueeze(0))
        self.linear.bias = torch.nn.Parameter(self.model.shared.weight[32098, 0].unsqueeze(0))
        self.model.eval()
        self.t = self.model.shared.weight[32097, 0].item()

    def get_scores(self, statements):
        tokens = self.tokenizer(statements, return_tensors='pt', padding=True)
        tokens.attention_mask = tokens.attention_mask.to(self.linear.weight.device)
        tokens.input_ids = tokens.input_ids.to(self.linear.weight.device)
        with torch.no_grad():
            output = self.model(**tokens)
            last_indices = tokens.attention_mask.sum(dim=1, keepdim=True) - 1
            last_indices = last_indices.unsqueeze(-1).expand(-1, -1, self.model.D)
            last_hidden_state = output.last_hidden_state.to(self.linear.weight.device)
            hidden = last_hidden_state.gather(dim=1, index=last_indices).squeeze(1)
            logits = self.linear(hidden).squeeze(-1)
            logits_calibrated = logits / self.t
            scores_calibrated = logits_calibrated.sigmoid()
        return scores_calibrated.cpu().tolist()


def get_vera_score_multihop(model: Vera, context, conditions, options):
    scores = [0.0 for _ in range(len(options))]
    for condition in conditions:
        statements = []
        for option in options:
            statements.append(context + ' ' + option + ' is ' + condition + '.')
        vera_scores = model.get_scores(statements)
        scores = [s + v for s, v in zip(scores, vera_scores)]
    return [s / len(conditions) for s in scores]


def get_vera_score(model: Vera, context, options):
    statements = []
    for option in options:
        statements.append(context + option)
    vera_scores = model.get_scores(statements)
    return vera_scores


def clean_extracted_answers(dataset: list, key, pattern, post_process=lambda x: x):
    pattern = re.compile(pattern)
    for data in dataset:
        new_extracted_answers = []
        for d in data[key]:
            match = pattern.search(d)
            if match:
                result = match.group(1)
                result = post_process(result)
            else:
                result = None
            new_extracted_answers.append(result)
        data[key] = new_extracted_answers
