# GPT Evaluation
## Table of Contents
- [Overview](#overview)
- [GPT Evaluation](#gpt-evaluation)
  - [Evaluation Category](#evaluation-category)
  - [Evaluation Category Examples](#evaluation-category-examples)
  - [Evaluation Metrics](#evaluation-metrics)
- [Evaluation Process](#evaluation-process)
  - [Data Format](#data-format)
  - [Prompt](#prompt)
    - [Battle Prompt](#battle-prompt)
    - [Evaluation Prompt](#evaluation-prompt)
  - [Evaluation](#evaluation)
    - [Configuration](#configuration)
    - [Evaluate](#evaluate)
- [FAQ](#faq)
- [Citations](#citations)


## Overview

In this directory, we introduce how you can evaluate your model using GPTs. It is now available for evaluation of both Chinese and English capability and we provide the following functions:

* Compare the performance of two different models (battle).
* Rate the model according to pre-defined metrics using prompting design.
* Rate the model according to pre-defined metrics with additional reference answer using prompting design.

## GPT Evaluation

### Evaluation Category

Our evaluation pipeline can examine the model's capability using different categories of questions. The following table includes some example categories. You can add your own questions.

| Evaluation Category | Description                                                  |
| :-----------------: | :----------------------------------------------------------- |
|    Brainstorming    | Models are asked to generate a range of creative and diverse ideas according to the question. The capability of creativity is required. |
|        Chat         | Models are asked to continue a multi-round dialogue given the roles involved. The capability of understanding, memorizing previous rounds of the dialogue and answering according to the persona provided is required. |
|     Generation      | Models are asked to generate an email, letter, article, etc. The capability of generating texts in a high quality and human-written way is required. |
|       Open QA       | Models are asked to answer an open QA question(without context provided). The capability of answering questions with the models' own knowledge base is required. |
|       Roleplay      | Models are asked to play the role provided. The capability of engaging in the scenario and effectively interacting with the user is required. |


### Evaluation Category Examples
To better understand each evaluation category, here are some example questions provided. Example questions are in the `configs/gpt_evaluation/data` folder.


| Evaluation Category | Chinese Example                                              | English Example                                              |
| :-----------------: | :----------------------------------------------------------- | :----------------------------------------------------------- |
|    Brainstorming    |                              | How do you properly chop an onion without crying?            |
|        Chat         | <br/> <br/> <br/> <br/><br/> <br/> | Complete a dialogue based on the following character information. Alex: A novice writer who is struggling to find inspiration and develop his writing skills. Emma: A successful author with many published works, providing guidance and advice to Alex.<br/>Alex: Hi Emma, I have been writing for a while now but can't seem to make any progress. Can you give me any advice? <br/>Emma: Hi Alex, sure. What kind of writing are you doing?<br/>Alex: I'm trying to write a novel, but I just can't seem to find any inspiration.<br/>Emma: <br/> |
|     Generation      |          | Write a set of guidelines for first-time pet owners on how to properly care for a new puppy. |
|       Open QA       | RNADNA                                 | Explain the process of osmosis in biological systems.        |
|      Roleplay       |  {} {} | I want you to act as a rapper. You will come up with powerful and meaningful lyrics, beats and rhythm that can wow the audience. Your lyrics should have an intriguing meaning and message which people can relate too. When it comes to choosing your beat, make sure it is catchy yet relevant to your words, so that when combined they make an explosion of sound everytime! My first request is "I need a rap song about finding strength within yourself." |

### Evaluation Metrics

GPT evaluation uses GPT models to evaluate the prediction of different models and different pre-defined evaluation metrics are applied to different categories. The following table shows the 10 pre-defined evaluation metrics both in Chinese and English:

|   Evaluation Metric   | Prompt Words                                                 | CoT(Chain-of-Thought)                                        |
| :-------------------: | :----------------------------------------------------------- | :----------------------------------------------------------- |
| <br/>(Language organization) | (1-5)</br></br>Language organization (1-5): whether the answer language is fluent and coherent, uses correct grammar, has a certain logic, uses appropriate connecting words, transition words, etc. | 1. <br/> 2. <br/> 3. <br/> 4. <br/> 5. <br/> 6. 1551</br></br>1. Read the answers and check for grammatical errors, poor word choice, or other significant mistakes.<br>2. Check that the answer is logical, conveys the information in a logical order, and is self-explanatory.<br>3. Determine if the answer is relevant to the question or topic and conveys a clear message.<br>4. Check that the answer is coherent and that appropriate transitions and switches are used to maintain coherence between sentences and paragraphs.<br>5. Check that the answer is clearly structured and organized in such a way that the reader can easily understand the hierarchy and structure of the information.<br>6. Evaluate the linguistic organization of the answer based on a combination of the above factors and give a score of 1 to 5, where 5 indicates very good linguistic organization and 1 indicates very poor linguistic organization. |
|       <br/>(Relevance)       | (1-5)</br></br>Relevance (1-5): whether the content of the answer is relevant to the topic, does not answer the wrong question, and strictly follows the requirements of the topic. | 1. <br/> 2. <br/> 3. <br/> 4. 1551</br></br>1. Read the question to determine what the question asks and what aspects of the question need to be answered.<br>2. Read the answers to make sure that they directly answer the question asked.<br>3. Check that the answer follows the requirements of the question, including the way it is answered, the length of the answer, the format of the answer, etc.<br>4. Evaluate how relevant the answer is based on the above factors and give a score of 1 to 5, where 5 means the answer is very relevant and 1 means the answer is not relevant at all. |
|      <br/>(Creativity)       | (1-5)</br></br>Creativity (1-5): Some brainstorming questions may require answers that are creative and suggest new ideas. | 1. <br/> 2. <br/> 3. <br/> 4. 15</br></br>1. Read the provided brainstorming questions carefully to make sure you understand the gist and context of the questions.<br>2. Based on your knowledge and experience, determine if the answers provided are feasible. If the answer is not feasible, the creativity score may be affected.<br>3. Consider whether the answer contains novel ideas or unique thoughts. An answer may overlap with a known solution and still be considered creative, as long as it offers a new perspective or approach to the problem.<br>4. Give a score of 1 to 5 depending on the creativity of the answer. If the answer lacks creativity, a lower score should be given. If the answer is creative and provides a new idea, a higher score should be given. |
|     <br/>(Practicality)      | (1-5)</br></br>Practicality (1-5): Some brainstorming questions may require answers to suggest practical suggestions or solutions. | 1. <br/> 2. <br/> 3. <br/> 4. 15</br></br>1. Read the provided brainstorming questions carefully to make sure you understand the gist and context of the questions.<br>2. Based on your knowledge and experience, determine if the answers provided are feasible. If the answer is not feasible, the practicality score may be affected.<br>3. Consider whether the suggestions or solutions presented in the answer are practical and workable. The answer may look good, but if it cannot be implemented or applied, the practicality score may be affected.<br>4. Give a score of 1 to 5 depending on the practicality of the answer. If the answer lacks practicality, a lower score should be given. If the answer makes a practical suggestion or solution and solves the problem well, a higher score should be given. |
|      <br/>(Correctness)      | (1-5)(1-5)</br></br> Correctness (1-5): whether the answer is correct or not. | 1. <br/>2. 52341<br/><br/>1. Read the question carefully and try to answer the question yourself. <br/>2. Check the correctness of the answer. You can use known facts or research to verify that the answer is correct. If the answer is correct, you can give a score of 5 for correctness. If the answer is partially correct, an appropriate score, such as 2, 3, or 4, may be given. If the answer is completely incorrect, only 1 point is awarded. |
|      <br/>(Naturalness)      | (1-5)</br></br>Naturalness (1-5): whether the answer is natural and fits the identity given by the question. | 1. <br/> 2. <br/> 3. 1515</br></br>1. Read the question and determine the identity information provided in the question.<br>2. Check whether the content of the answer matches the identity given in the question.<br>3. Based on the above factors, score the naturalness of the response on a scale from 1 to 5, where 1 means unnatural and 5 means very natural and in accordance with the identity given in the question. |
|     <br/>(Engagingness)      | (1-5)</br></br>Engagingness (1-5): whether the answer responds appropriately to the content of the preceding conversation and whether it understands the context and background of the conversation. | 1. <br/> 2. <br/> 3. 1515</br></br>1. Read the questions to determine the context and background of the dialogue.<br>2. Check that the answer fully understands the context and background of the conversation and that it fits naturally into the conversation without seeming abrupt.<br>3. Based on the above factors, rate the response's engagement on a scale from 1 to 5, where 1 means not engaged and 5 means very engaged and appropriately understands the context and background of the conversation. |
|    <br/>(Reasonableness)     | (1-5)</br></br>Reasonableness (1-5): Whether the answer can form a logical connection with the content of the previous dialogue, whether it is consistent with common sense, and whether it can reasonably exist in this context. | 1. <br/> 2. <br/> 3. 1515</br></br>1. Read the question and determine the topic of the conversation and the direction the question expects the answer to go.<br>2. Determine whether the answer can be logically connected to the preceding conversation, whether it makes common sense, and whether it can reasonably exist in this context.<br>3. Based on the above factors, rate the reasonableness of the answer on a scale from 1 to 5, where 1 means unreasonable and 5 means very reasonable and able to form a logical connection with the preceding dialogue content and consistent with common sense. |
|       <br/>(Diversity)       | (1-5)</br></br>Diversity (1-5): Whether the answers use beautiful language and have some creativity and imagination. However, answers should also be kept reasonable and moderate, not overly exaggerated or off-topic. | 1. <br/> 2. <br/> 3. <br/> 4. 5. 1551</br></br>1. Read the entire response carefully to ensure that you fully understand the content and theme expressed in the response.<br>2. While reading the response, pay attention to the quality of the language, such as whether the wording is correct and the language is vivid.<br>3. Check the creativity and imagination of the response to see if the response is engaging to read on.<br>4. Check the reasonableness and appropriateness of the responses to see if the responses are exaggerated or off-topic.<br>5. Rate the diversity on a scale of 1 to 5, with a 5 indicating a good quality response that is engaging to read and a 1 indicating a raw response or a question that is off-topic. |
|       <br/>(Fidelity)        | (1-5)</br></br>Fidelity (1-5): whether the answer is able to answer the given request in strict compliance with the role setting. | 1. <br/> <br/> 3. <br/> 4. 1515</br></br>1. Read the question carefully to understand how the character is set up and represented in the question, including aspects such as occupation, background, point of view, and personality.<br>2. Read the question's request and confirm the details that need to be taken into account when answering the request.<br>3. Compare the provided answer with the setting of the role and assess whether the answer can strictly adhere to the setting of the role.<br>4. Combine the results of the above assessment to give a fidelity score ranging from 1 to 5, where a score of 1 means that the response does not match the persona at all, and a score of 5 means that the response fully complies with the persona and satisfies the given request. |

GPT models evaluate the quality of model predictions based on the given prompt words and gives a score between 1-5.

> **NOTE 1:**  You can find all the prompt words and CoT(Chain-of-Thought) in `configs/gpt_evaluation/prompt/evaluation_prompt`.

> **NOTE 2:** To add customized metrics, you can refer to [FAQ](#faq).

## Evaluation Process

### Data Format

A JSON file contains one list. Each element in the list is a target answer / prediction record for one instruction / question.
An element should have the following fields:

* `category` (str, compulsory): The category of the instruction / question.
* `instruction` (str, compulsory): The instruction / question for the LLM.
* `input` (str, optional): The additional context of the instruction / question.
* `output` (str, optional): The model output of the instruction, models will fill in this field during inference time.
* `target` (str, optional): The target answer for the instruction.
* `id` (int, compulsory): The ID of the instruction / question.

Example:

```json
[
    {
        "category": "brainstorming",
        "instruction": "",
        "input": "",
        "output": "",
        "target": "",
        "id": 1
    },
    {
        "category": "chat",
        "instruction": "",
        "input": "    ",
        "output": "",
        "target": "",
        "id": 2
    }
]
```

### Prompt

#### Battle Prompt

The following is the Chinese battle prompt. In the battle prompt, the question and answers from two different models are fed into the prompt template. You can find example battle prompt files for Chinese and English in `configs/gpt_evaluation/prompt/battle_prompt`.

```json
{
  "id": 1,
  "system_prompt": "",
  "prompt_template": "[]\n{question}\n\n[1AI]\n{answer_1}\n\n[1AI]\n\n[2AI  ]\n{answer_2}\n\n[2AI]\n\n[]\n{prompt}\n\n",
  "prompt": "AI\nAI110\n12AIAI"
}
```

#### Evaluation Prompt

The following is an example of a Chinese GPT evaluation prompt. In an evaluation prompt, you should define your metrics in `metrics` and provide CoT(Chain-of-Thought) in `CoT`.  You can find example evaluation prompt files for Chinese and English in `configs/gpt_evaluation/prompt/evaluation_prompt`.

```json
{
  "brainstorming": {
    "id": 1,
    "category": "brainstorming",
    "metrics": {
      "language organization": "(1-5)"
    },
    "CoT": {
      "language organization": "1. \n2. \n3. \n4. \n5. \n6. 1551\n\n"
    },
    "prompt": "\n\n\n\n{question}\n\n\n\n{answer}\n\n\n\n{metric}\n\n\n\n{steps}"
  }
}
```

`"metrics"`: the metrics that can be used in GPT evaluation. This field determines which metrics can be added to your config file.

`"CoT"`: evaluation steps you prompt to GPT models for each metric defined in `"metrics"`.

### Evaluation

#### Configuration

The following is an example of a Chinese config file. The configuration file can control how the pipeline evaluates the model. You need to specify GPT evaluation metrics in key `GPT`. You can find an example English config file in `configs/gpt_evaluation/config/config_en.json`.

```json
{
    "language": "cn",
    "category": {
        "brainstorming": {
            "GPT": [
                "language organization",
                "relevance",
                "creativity",
                "practicality",
                "reasonableness"
            ]
        }
    }
}
```

`"language"`: the language used to evaluate the model capability. We only support Chinese `"cn"` for now.

`"category"`: the category/categories needed to evaluate the model capability.

`"GPT"`: the metrics you want to use for GPT evaluation.


#### Evaluate

After setting the configuration file, you can evaluate the model using `examples/gpt_evaluation/eval.py`. If you want to make comparisons between answers of two different models, you should specify two answer files in the argument `answer_file_list` and two model names in the argument `model_name_list`. If you want to evaluate one answer file, the length of both `answer_file_list` and `model_name_list` should be 1 and the program will perform evaluation using automatic metrics and GPT models.

An example script is provided as follows:

```shell
python eval.py \
    --config_file "path to the config file" \
    --battle_prompt_file "path to the prompt file for battle" \
    --gpt_evaluation_prompt_file "path to the prompt file for gpt evaluation" \
    --target_file "path to the target answer file" \
    --answer_file_list "path to the answer files of at most 2 models" \
    --model_name_list "the names of at most 2 models" \
    --gpt_model "which GPT model to use for evaluation" \
    --save_path "path to save results" \
    --openai_key "your openai key" \
```

If you want GPT evaluation with reference, you can add an argument `--gpt_with_reference`, but make sure the reference file have target answers.

## FAQ

<details><summary><b>How can I add a new GPT evaluation metric?</b></summary>

For example, if you want to add a new metric `persuasiveness` into category `brainstorming`, you should add the metric definition and its corresponding CoT(Chain-of-thought) in the evaluation prompt file in `prompt/evaluation_promt`. The CoT can be generated using ChatGPT. You can prompt ChatGPT to generate evaluation steps for the new metric.

```json
{
  "brainstorming": {
    "id": 1,
    "category": "brainstorming",
    "metrics": {
      "persuasiveness": "persuasiveness(1-5)a short description for persuasiveness"
    },
    "CoT": {
      "persuasiveness": "CoT for persuasiveness\n\npersuasiveness"
    },
    "prompt": "You are a good assistant. Please rate the given answer to the \"brainstorming\" question below.\n\nThe question is as follows:\n\n{question}\n\nThe answer is as follows:\n\n{answer}\n\nThe metric for evaluation is as follows:\n\n{metric}\n\nYou should follow the following evaluation steps:\n\n{steps}"
  }
}
```

</details>

## Citations

```bibtex
@misc{vicuna2023,
    title = {Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90\%* ChatGPT Quality},
    url = {https://vicuna.lmsys.org},
    author = {Chiang, Wei-Lin and Li, Zhuohan and Lin, Zi and Sheng, Ying and Wu, Zhanghao and Zhang, Hao and Zheng, Lianmin and Zhuang, Siyuan and Zhuang, Yonghao and Gonzalez, Joseph E. and Stoica, Ion and Xing, Eric P.},
    month = {March},
    year = {2023}
}

@misc{liu2023geval,
      title={G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment},
      author={Yang Liu and Dan Iter and Yichong Xu and Shuohang Wang and Ruochen Xu and Chenguang Zhu},
      year={2023},
      eprint={2303.16634},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
