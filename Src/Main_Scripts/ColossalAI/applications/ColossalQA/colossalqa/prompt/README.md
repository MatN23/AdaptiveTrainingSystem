# Prompt Design Guide

For the retriever conversation system, users can customize three prompts.

## The Retrieval QA Prompt
This is the prompt for retrieval QA, the input is user's inputs, the retrieved documents, the historical conversation.

### Chinese
```
AI



:
[retrieved documents]

:
[historical conversation, overlength chat history will be summarized]

: [question]
Assistant:
```

### English
```
[INST] <<SYS>>Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If the answer cannot be inferred based on the given context, please don't share false information.<</SYS>>
Use the context and chat history to respond to the human's input at the end or carry on the conversation. You should generate one response only. No following up is needed.

context:
[retrieved documents]

chat history
[historical conversation, overlength chat history will be summarized]

Human: {question}
Assistant:
```

## Summarization Prompt
This prompt is used by the memory module to recursively summarize overlength conversation to shrink the length of the prompt.

## Disambiguity Prompt
This prompt is used to perform zero-shot reference resolution to disambiguate entity references within user's questions.

## Final Prompt Examples
Assume k=3 for the retriever.

### English
Note that the "[INST] <<SYS>>...<</SYS>>" template is the specific prompt format used in LLaMA2.
#### Normal Length
```
[INST] <<SYS>>Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If the answer cannot be inferred based on the given context, please don't share false information.<</SYS>>
Use the context and chat history to respond to the human's input at the end or carry on the conversation. You should generate one response only. No following up is needed.

context:
[document 1]

[document 2]

[document 3]

chat history
Human: XXX
Assistant: XXX
...

Human: {question}
Assistant:
```

#### Overlength
```
[INST] <<SYS>>Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.
If the answer cannot be inferred based on the given context, please don't share false information.<</SYS>>
Use the context and chat history to respond to the human's input at the end or carry on the conversation. You should generate one response only. No following up is needed.

context:
[document 1]

[document 2]

[document 3]

chat history
A summarization of historical conversation:
[one line summary of historical conversation]
Most recent conversation:
Human: XXX
Assistant: XXX
...

Human: {question}
Assistant:
```

### Chinese
#### Normal Length
```
AI



:
[document 1]

[document 2]

[document 3]

:
: XXX
Assistant: XXX
...

: [question]
Assistant:
```

#### Overlength
```
AI



:
[document 1]

[document 2]

[document 3]

:
:
[one line summary of historical conversation]
:
: XXX
Assistant: XXX
...

: [question]
Assistant:
```
