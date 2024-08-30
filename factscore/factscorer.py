import os
import string

import numpy as np
from tqdm import tqdm

from factscore.abstain_detection import is_response_abstained
from factscore.atomic_facts import AtomicFactGenerator
from factscore.openai_lm import OpenAIModel
from factscore.retrieval import DocDB, Retrieval


class FactScorer(object):

    def __init__(self,
                 data_dir=".cache/factscore",
                 cache_dir=".cache/factscore",
                 openai_key="api.key",
                 abstain_detection_type=None,
                 batch_size=256):
        self.db = {}
        self.retrieval = {}
        self.npm = {}
        self.batch_size = batch_size  # batch size for retrieval
        self.openai_key = openai_key
        self.abstain_detection_type = abstain_detection_type

        self.data_dir = data_dir
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.af_generator = None
        self.lm = OpenAIModel("ChatGPT", key_path=openai_key)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    def save_cache(self):
        pass

    def register_knowledge_source(self, name="enwiki-20230401", db_path=None, data_path=None):
        assert name not in self.retrieval, f"{name} already registered"
        if db_path is None:
            db_path = os.path.join(self.data_dir, f"{name}.db")

        if data_path is None:
            data_path = os.path.join(self.data_dir, f"{name}.jsonl")

        cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.json")
        embed_cache_path = os.path.join(self.cache_dir, f"retrieval-{name}.pkl")

        self.db[name] = DocDB(db_path=db_path, data_path=data_path)
        self.retrieval[name] = Retrieval(self.db[name], cache_path, embed_cache_path, batch_size=self.batch_size)

    def get_score(self,
                  topics,
                  generations,
                  gamma=10,
                  knowledge_source=None,
                  verbose=False):
        if knowledge_source is None:
            # use the default knowledge source
            knowledge_source = "enwiki-20230401"

        if knowledge_source not in self.retrieval:
            self.register_knowledge_source(knowledge_source)

        if self.af_generator is None:
            self.af_generator = AtomicFactGenerator(key_path=self.openai_key,
                                                    demon_dir=os.path.join(self.data_dir, "demos"))

        if verbose:
            topics = tqdm(topics)

        atomic_facts = []
        for topic, gen in zip(topics, generations):
            # optionally, first detect if the response is abstained
            response_abstained = is_response_abstained(gen, self.abstain_detection_type)
            if response_abstained:
                atomic_facts.append(None)
                continue
            # continue only when the response is not abstained
            curr_afs, _ = self.af_generator.run(gen)
            curr_afs = [fact for _, facts in curr_afs for fact in facts]
            if len(curr_afs) == 0:
                atomic_facts.append(None)
            else:
                atomic_facts.append(curr_afs)

        assert len(atomic_facts) == len(topics)
        respond_ratio = np.mean([facts is not None for facts in atomic_facts])

        if verbose:
            topics = tqdm(topics)

        scores = []
        init_scores = []
        decisions = []
        for topic, generation, facts in zip(topics, generations, atomic_facts):
            if facts is None:
                decisions.append(None)
            else:
                decision = self._get_score(topic, generation, facts, knowledge_source)
                score = np.mean([d["is_supported"] for d in decision])

                if gamma:
                    init_scores.append(score)
                    penalty = 1.0 if len(facts) > gamma else np.exp(1 - gamma / len(facts))
                    score = penalty * score

                decisions.append(decision)
                scores.append(score)

        out = {"score": scores[0] if len(scores) == 1 else scores,
               "respond_ratio": respond_ratio,
               "decisions": decisions,
               "num_facts_per_response": np.mean([len(d) for d in decisions if d is not None])}

        if gamma:
            out["init_score"] = np.mean(init_scores)

        return out

    def _get_score(self, topic, generation, atomic_facts, knowledge_source, cost_estimate=None):
        decisions = []
        prompts = []
        for atom in atomic_facts:
            atom = atom.strip()
            passages = self.retrieval[knowledge_source].get_passages(topic, atom, k=5)
            definition = "Answer the question about {} based on the given context.\n\n".format(topic)
            context = ""
            for psg_idx, psg in enumerate(reversed(passages)):
                context += "Title: {}\nText: {}\n\n".format(psg["title"], psg["text"].replace("<s>", "").replace("</s>", ""))
            definition += context.strip()
            if not definition[-1] in string.punctuation:
                definition += "."
            prompts.append("{}\n\nInput: {} True or False?\nOutput:".format(definition.strip(), atom.strip()))

        outputs, _ = self.lm.generate(prompts)
        for output, atom in zip(outputs, atomic_facts):
            generated_answer = output.lower()
            if "true" in generated_answer or "false" in generated_answer:
                if "true" in generated_answer and "false" not in generated_answer:
                    is_supported = True
                elif "false" in generated_answer and "true" not in generated_answer:
                    is_supported = False
                else:
                    is_supported = generated_answer.index("true") > generated_answer.index("false")
            else:
                is_supported = all([keyword not in generated_answer.lower().translate(
                    str.maketrans("", "", string.punctuation)).split() for keyword in ["not", "cannot", "unknown", "information"]])

            decisions.append({"atom": atom, "is_supported": is_supported})

        return decisions


if __name__ == '__main__':
    pass
