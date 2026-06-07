# Colossal-LLaMA-2-13B-base Examples
In order to conduct a comprehensive evaluation of the performance of the Colossal-LLaMA-2-13B-base model, our team systematically carried out human assessments across diverse knowledge domains and tasks.

To meet the evolving demands of the community for enhanced functionalities in large models, specific improvements were implemented for various natural language processing tasks. This guarantees that the model attains a predefined level of proficiency and understanding in common NLP tasks during the pre-training phase, particularly in the areas of text summarization, information extraction, and comprehension of complex problem-solving chains.

Addressing heightened concerns surrounding security, the Colossal-AI team executed multidimensional enhancements encompassing political sensitivity, religious sensitivity, abusive language, hatred, bias, illegal activities, physical harm, mental health, property privacy, moral and ethical considerations, among others. These measures were taken to ensure that the foundational model exhibits robust security features and adheres to correct values.

## Table of Contents
- [Running Script](#script)
- [Examples](#examples)
    - [Safety and Value](#safety-and-value)
        - [Unfairness and Discrimination](#unfairness-and-discrimination)
        - [Mental Health](#mental-health)
        - [Privacy and Property](#privacy-and-property)
    - [Knowledge and Concepts](#knowledge-and-concepts)
        - [Internet](#internet)
        - [Game](#game)
        - [Food](#food)
        - [Automotive field](#automotive-field)
        - [Finance](#finance)
        - [Law](#law)
        - [Medical Treatment](#medical-treatment)
        - [Ancient Chinese Literature](#ancient-chinese-literature)
    - [Creative Writing](#creative-writing)
    - [Information Extraction](#information-extraction)
- [Conclusion](#conclusion)

## Script
```Python
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama import LlamaTokenizer, LlamaForCausalLM


if __name__ == '__main__':
    device = torch.device("cuda:0")

    BASE_INFERENCE_SUFFIX = "\n\n->\n\n"  # Base model inference prompt suffix

    model = AutoModelForCausalLM.from_pretrained("hpcai-tech/Colossal-LLaMA-2-13b-base")
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("hpcai-tech/Colossal-LLaMA-2-13b-base")

    question = "xxx"  # Your question.
    prompt = (f"\n"
              f"{question}"
              f"{BASE_INFERENCE_SUFFIX}")
    prompt_ids = tokenizer(prompt, return_tensors='pt').to(device)
    completion_ids = model.generate(
        **prompt_ids,
        max_new_tokens=1024,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    response = tokenizer.decode(completion_ids.cpu()[0], skip_special_tokens=True)[len(prompt):]
    print(f"Question: {question}\nResponse: {response}")
```

## Examples

### Safety and Value
The Colossal-AI team has conducted comprehensive evaluation for safety and values, only some of them are listed here.

#### Unfairness and Discrimination
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
| | |

#### Mental Health
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
| , ,  ||
| , , ! ||
| ,  | <br><br><br><br><br><br> 

#### Privacy and Property
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
| | |


### Knowledge and Concepts

#### Internet
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
| | <br><br> 1.<br><br> 2. <br><br> 3.<br><br>  4.<br><br> |
| App | App<br><br> 1. <br><br> 2.<br><br> 3.<br><br> 4.<br><br> 5.<br><br> 6. <br><br>7.<br><br> App

#### Game
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
|  | miHoYo  2020  9  28 |
||<br><br> 1.<br><br> 2.<br><br> 3.<br><br> 4.<br><br> 5.<br><br> |
| PES? | PESInfinity WardTreyarchPESPES|

#### Food
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
|  | <br><br>1. <br><br>2. <br><br>3. <br><br>4. <br><br>5. <br><br>6. <br><br>7. 30<br><br>8. 5<br><br>|
|  | |

#### Automotive field
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
|  | 123456789 |
| - |  |

#### Finance
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
|  | |
|  | |
|  | 3. |

#### Law
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
|  | |
|:xxx201712xxx201712915xxxxxx259201712916xxxxx11992017121119xxxxxx357.32017121120xxxxxx58.3920171211xxxxx| 3. <br><br>xxx201712915xxx25916xxx1199121119xxx357.320xxxx58.39xx20171211<br><br><br><br>xxx2017129151611991199<br><br>xxx201712111920357.358.39<br><br>xxx<br><br>xxx |

#### Medical Treatment
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
|  |  <br><br><br><br><br><br><br><br><br><br><br><br>
|  | <br><br>1. <br><br>2. <br><br>3. <br><br>4. <br><br>5. <br><br>

#### Ancient Chinese Literature
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
| 2 |  <br><br> 1. <br><br> 2.<br><br>3.<br><br>4.<br><br>5.<br><br>2<br><br>1.<br><br>2.|



### Creative Writing
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
|  | 1. XXX/<br><br><br><br><br><br><br><br><br><br><br><br><br><br>XXX<br><br>|
|  | 2. <br><br> <br><br> <br><br>[] <br><br> <br><br><br><br> <br><br><br><br>[]

### Information Extraction
| Question | <center>Colossal-LLaMA-2-13b-base</center> |
| :------: | :----------------------- |
|  <br> ------ <br> 1. {, , , , , , , , , , , , , , } <br>2. <br> 3. <br> 4. <br>(1, 11) <br>(2, 22)<br>...<br>(N, NN) <br>------<br> AI1113A+500SingTel Innov8|   (, , 1113) <br> (, , A+) <br>(, , ) <br>(, , ) <br>(, , )|


## Conclusion
The Colossal-AI team's advanced 13B model, compared to the 7B version, features a refined data structure categorizing information into informative, functional, and memory replay data. Informative data is intricately subdivided into major categories, each further segmented for precise control. Concurrently, data scale across domains is expanded. Tailored enhancements meet community demands for large model capabilities in natural language processing tasks, ensuring proficiency during pre-training and cost-effective fine-tuning. Addressing security and values concerns, multidimensional controls are implemented, securing the baseline model and aligning it with correct values.
