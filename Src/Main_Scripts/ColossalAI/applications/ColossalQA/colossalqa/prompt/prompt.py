"""
All custom prompt templates are defined here.
"""

from langchain.prompts.prompt import PromptTemplate

# Below are Chinese retrieval qa prompts

_CUSTOM_SUMMARIZER_TEMPLATE_ZH = """

1:
:
Assistant

:
: ?
Assistant: 

:
Assistant


:
{summary}

:
{new_lines}

:"""


_ZH_RETRIEVAL_QA_PROMPT = """<>  </>

{context}

<>
{chat_history}
</>

<>{question}</>
"""

ZH_RETRIEVAL_QA_TRIGGER_KEYWORDS = [""]
ZH_RETRIEVAL_QA_REJECTION_ANSWER = ""


_ZH_RETRIEVAL_CLASSIFICATION_USE_CASE = """

:
{context}

:
{question}
"""

_ZH_DISAMBIGUATION_PROMPT = """
(:)

:
:
: ?
Assistant: 

: ?
: ?

:
{chat_history}

: {input}
:"""


# Below are English retrieval qa prompts

_EN_RETRIEVAL_QA_PROMPT = """[INST] <<SYS>>Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist content.
If the answer cannot be inferred based on the given context, please say "I cannot answer the question based on the information given.".<</SYS>>
Use the context and chat history to answer the question.

context:
{context}

chat history
{chat_history}

question: {question}
answer:"""
EN_RETRIEVAL_QA_TRIGGER_KEYWORDS = ["cannot answer the question"]
EN_RETRIEVAL_QA_REJECTION_ANSWER = "Sorry, this question cannot be answered based on the information provided."

_EN_DISAMBIGUATION_PROMPT = """[INST] <<SYS>>You are a helpful, respectful and honest assistant. You always follow the instruction.<</SYS>>
Please replace any ambiguous references in the given sentence with the specific names or entities mentioned in the chat history or just output the original sentence if no chat history is provided or if the sentence doesn't contain ambiguous references. Your output should be the disambiguated sentence itself (in the same line as "disambiguated sentence:") and contain nothing else.

Here is an example:
Chat history:
Human: I have a friend, Mike. Do you know him?
Assistant: Yes, I know a person named Mike

sentence: What's his favorite food?
disambiguated sentence: What's Mike's favorite food?
[/INST]
Chat history:
{chat_history}

sentence: {input}
disambiguated sentence:"""


# Prompt templates

# English retrieval prompt, the model generates answer based on this prompt
PROMPT_RETRIEVAL_QA_EN = PromptTemplate(
    template=_EN_RETRIEVAL_QA_PROMPT, input_variables=["question", "chat_history", "context"]
)
# English disambigate prompt, which replace any ambiguous references in the user's input with the specific names or entities mentioned in the chat history
PROMPT_DISAMBIGUATE_EN = PromptTemplate(template=_EN_DISAMBIGUATION_PROMPT, input_variables=["chat_history", "input"])

# Chinese summary prompt, which summarize the chat history
SUMMARY_PROMPT_ZH = PromptTemplate(input_variables=["summary", "new_lines"], template=_CUSTOM_SUMMARIZER_TEMPLATE_ZH)
# Chinese disambigate prompt, which replace any ambiguous references in the user's input with the specific names or entities mentioned in the chat history
PROMPT_DISAMBIGUATE_ZH = PromptTemplate(template=_ZH_DISAMBIGUATION_PROMPT, input_variables=["chat_history", "input"])
# Chinese retrieval prompt, the model generates answer based on this prompt
PROMPT_RETRIEVAL_QA_ZH = PromptTemplate(
    template=_ZH_RETRIEVAL_QA_PROMPT, input_variables=["question", "chat_history", "context"]
)
# Chinese retrieval prompt for a use case to analyze fault causes
PROMPT_RETRIEVAL_CLASSIFICATION_USE_CASE_ZH = PromptTemplate(
    template=_ZH_RETRIEVAL_CLASSIFICATION_USE_CASE, input_variables=["question", "context"]
)
