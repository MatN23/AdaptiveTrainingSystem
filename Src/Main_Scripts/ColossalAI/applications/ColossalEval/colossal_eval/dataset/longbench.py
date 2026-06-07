import os
from copy import deepcopy
from typing import Dict, List

from colossal_eval.utils import get_json_list

from colossalai.logging import DistributedLogger

from .base import BaseDataset

dataset2prompt = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": 'You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write "unanswerable". If the question is a yes/no question, answer "yes", "no", or "unanswerable". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:',
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "\n\n{context}\n\n\n\n{input}\n",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "\n\n{context}\n\n\n\n{input}\n",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "\n\n{context}\n\n",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "\n\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": 'Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like "Paragraph 1", "Paragraph 2", etc.\n\nThe answer is: ',
    "passage_retrieval_zh": '\n\n{context}\n\n\n\n{input}\n\n"1""2"\n\n',
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n",
}

dataset2maxlen = {
    "narrativeqa": 128,
    "qasper": 128,
    "multifieldqa_en": 64,
    "multifieldqa_zh": 64,
    "hotpotqa": 32,
    "2wikimqa": 32,
    "musique": 32,
    "dureader": 128,
    "gov_report": 512,
    "qmsum": 512,
    "multi_news": 512,
    "vcsum": 512,
    "trec": 64,
    "triviaqa": 32,
    "samsum": 128,
    "lsht": 64,
    "passage_count": 32,
    "passage_retrieval_en": 32,
    "passage_retrieval_zh": 32,
    "lcc": 64,
    "repobench-p": 64,
}

default_inference_kwargs = {
    "calculate_loss": True,
    "all_classes": None,
    "language": "Chinese",
    "pretrain": False,
    "max_new_tokens": 32,
}


class LongBenchDataset(BaseDataset):
    """
    Dataset class for LongBench dataset.
    Data source: https://huggingface.co/datasets/THUDM/LongBench
    This dataset class will convert the original dataset into the inference dataset.

    Issue link: https://github.com/THUDM/LongBench/issues/15 (fixed)
    There are duplicate target answers in `nq.jsonl`, but this doesn't affect evaluation results.
    Also doesn't affect perplexity calculation (the program only need to select the minimum loss).
    """

    @staticmethod
    def load(path: str, logger: DistributedLogger) -> List[Dict]:
        dataset = {"test": {}}

        files = os.listdir(path)
        files.sort()

        for file in files:
            category = file[0:-6]

            if category.endswith("_e"):
                continue

            dataset["test"][category] = {"data": []}

            file_dir = os.path.join(path, file)

            loaded_jsonl = get_json_list(file_dir)

            # It's been tested that each data sample in one subcategory have same inference arguments.
            inference_kwargs = deepcopy(default_inference_kwargs)
            if loaded_jsonl[0]["all_classes"] is not None:
                inference_kwargs["all_classes"] = loaded_jsonl[0]["all_classes"]
            inference_kwargs["max_new_tokens"] = dataset2maxlen[category]
            dataset["test"][category]["inference_kwargs"] = inference_kwargs

            for sample in loaded_jsonl:
                prompt = dataset2prompt[category].format(**sample)

                data_sample = {
                    "dataset": "longbench",
                    "split": "test",
                    "category": category,
                    "instruction": prompt,
                    "input": "",
                    "output": "",
                    "target": sample["answers"],
                }

                dataset["test"][category]["data"].append(data_sample)

        return dataset