---
sidebar_position: 2
---
# How to use few shot examples in chat models

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Prompt templates](/docs/concepts/#prompt-templates)
- [Example selectors](/docs/concepts/#example-selectors)
- [Chat models](/docs/concepts/#chat-model)
- [Vectorstores](/docs/concepts/#vectorstores)

:::

This guide covers how to prompt a chat model with example inputs and outputs. Providing the model with a few such examples is called few-shotting, and is a simple yet powerful way to guide generation and in some cases drastically improve model performance.

There does not appear to be solid consensus on how best to do few-shot prompting, and the optimal prompt compilation will likely vary by model. Because of this, we provide few-shot prompt templates like the [FewShotChatMessagePromptTemplate](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate.html?highlight=fewshot#langchain_core.prompts.few_shot.FewShotChatMessagePromptTemplate) as a flexible starting point, and you can modify or replace them as you see fit.

The goal of few-shot prompt templates are to dynamically select examples based on an input, and then format the examples in a final prompt to provide for the model.

**Note:** The following code examples are for chat models only, since `FewShotChatMessagePromptTemplates` are designed to output formatted [chat messages](/docs/concepts/#message-types) rather than pure strings. For similar few-shot prompt examples for pure string templates compatible with completion models (LLMs), see the [few-shot prompt templates](/docs/how_to/few_shot_examples/) guide.

## Fixed Examples

The most basic (and common) few-shot prompting technique is to use fixed prompt examples. This way you can select a chain, evaluate it, and avoid worrying about additional moving parts in production.

The basic components of the template are:
- `examples`: A list of dictionary examples to include in the final prompt.
- `example_prompt`: converts each example into 1 or more messages through its [`format_messages`](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html?highlight=format_messages#langchain_core.prompts.chat.ChatPromptTemplate.format_messages) method. A common example would be to convert each example into one human message and one AI message response, or a human message followed by a function call message.

Below is a simple demonstration. First, define the examples you'd like to include:


```python
%pip install -qU langchain langchain-openai langchain-chroma

import os
from getpass import getpass

os.environ["OPENAI_API_KEY"] = getpass()
```

    [33mWARNING: You are using pip version 22.0.4; however, version 24.0 is available.
    You should consider upgrading via the '/Users/jacoblee/.pyenv/versions/3.10.5/bin/python -m pip install --upgrade pip' command.[0m[33m
    [0mNote: you may need to restart the kernel to use updated packages.



```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
]
```

Next, assemble them into the few-shot prompt template.


```python
# This is a prompt template used to format each individual example.
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

print(few_shot_prompt.invoke({}).to_messages())
```

    [HumanMessage(content='2+2'), AIMessage(content='4'), HumanMessage(content='2+3'), AIMessage(content='5')]


Finally, we assemble the final prompt as shown below, passing `few_shot_prompt` directly into the `from_messages` factory method, and use it with a model:


```python
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)
```


```python
from langchain_openai import ChatOpenAI

chain = final_prompt | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)

chain.invoke({"input": "What's the square of a triangle?"})
```




    AIMessage(content='A triangle does not have a square. The square of a number is the result of multiplying the number by itself.', response_metadata={'token_usage': {'completion_tokens': 23, 'prompt_tokens': 52, 'total_tokens': 75}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'stop', 'logprobs': None}, id='run-3456c4ef-7b4d-4adb-9e02-8079de82a47a-0')



## Dynamic few-shot prompting

Sometimes you may want to select only a few examples from your overall set to show based on the input. For this, you can replace the `examples` passed into `FewShotChatMessagePromptTemplate` with an `example_selector`. The other components remain the same as above! Our dynamic few-shot prompt template would look like:

- `example_selector`: responsible for selecting few-shot examples (and the order in which they are returned) for a given input. These implement the [BaseExampleSelector](https://api.python.langchain.com/en/latest/example_selectors/langchain_core.example_selectors.base.BaseExampleSelector.html?highlight=baseexampleselector#langchain_core.example_selectors.base.BaseExampleSelector) interface. A common example is the vectorstore-backed [SemanticSimilarityExampleSelector](https://api.python.langchain.com/en/latest/example_selectors/langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector.html?highlight=semanticsimilarityexampleselector#langchain_core.example_selectors.semantic_similarity.SemanticSimilarityExampleSelector)
- `example_prompt`: convert each example into 1 or more messages through its [`format_messages`](https://api.python.langchain.com/en/latest/prompts/langchain_core.prompts.chat.ChatPromptTemplate.html?highlight=chatprompttemplate#langchain_core.prompts.chat.ChatPromptTemplate.format_messages) method. A common example would be to convert each example into one human message and one AI message response, or a human message followed by a function call message.

These once again can be composed with other messages and chat templates to assemble your final prompt.

Let's walk through an example with the `SemanticSimilarityExampleSelector`. Since this implementation uses a vectorstore to select examples based on semantic similarity, we will want to first populate the store. Since the basic idea here is that we want to search for and return examples most similar to the text input, we embed the `values` of our prompt examples rather than considering the keys:


```python
from langchain_chroma import Chroma
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

examples = [
    {"input": "2+2", "output": "4"},
    {"input": "2+3", "output": "5"},
    {"input": "2+4", "output": "6"},
    {"input": "What did the cow say to the moon?", "output": "nothing at all"},
    {
        "input": "Write me a poem about the moon",
        "output": "One for the moon, and one for me, who are we to talk about the moon?",
    },
]

to_vectorize = [" ".join(example.values()) for example in examples]
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_texts(to_vectorize, embeddings, metadatas=examples)
```

### Create the `example_selector`

With a vectorstore created, we can create the `example_selector`. Here we will call it in isolation, and set `k` on it to only fetch the two example closest to the input.


```python
example_selector = SemanticSimilarityExampleSelector(
    vectorstore=vectorstore,
    k=2,
)

# The prompt template will load examples by passing the input do the `select_examples` method
example_selector.select_examples({"input": "horse"})
```




    [{'input': 'What did the cow say to the moon?', 'output': 'nothing at all'},
     {'input': '2+4', 'output': '6'}]



### Create prompt template

We now assemble the prompt template, using the `example_selector` created above.


```python
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate

# Define the few-shot prompt.
few_shot_prompt = FewShotChatMessagePromptTemplate(
    # The input variables select the values to pass to the example_selector
    input_variables=["input"],
    example_selector=example_selector,
    # Define how each example will be formatted.
    # In this case, each example will become 2 messages:
    # 1 human, and 1 AI
    example_prompt=ChatPromptTemplate.from_messages(
        [("human", "{input}"), ("ai", "{output}")]
    ),
)

print(few_shot_prompt.invoke(input="What's 3+3?").to_messages())
```

    [HumanMessage(content='2+3'), AIMessage(content='5'), HumanMessage(content='2+2'), AIMessage(content='4')]


And we can pass this few-shot chat message prompt template into another chat prompt template:


```python
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a wondrous wizard of math."),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

print(few_shot_prompt.invoke(input="What's 3+3?"))
```

    messages=[HumanMessage(content='2+3'), AIMessage(content='5'), HumanMessage(content='2+2'), AIMessage(content='4')]


### Use with an chat model

Finally, you can connect your model to the few-shot prompt.


```python
chain = final_prompt | ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0.0)

chain.invoke({"input": "What's 3+3?"})
```




    AIMessage(content='6', response_metadata={'token_usage': {'completion_tokens': 1, 'prompt_tokens': 51, 'total_tokens': 52}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'stop', 'logprobs': None}, id='run-6bcbe158-a8e3-4a85-a754-1ba274a9f147-0')



## Next steps

You've now learned how to add few-shot examples to your chat prompts.

Next, check out the other how-to guides on prompt templates in this section, the related how-to guide on [few shotting with text completion models](/docs/how_to/few_shot_examples), or the other [example selector how-to guides](/docs/how_to/example_selectors/).


```python

```
