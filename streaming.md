---
sidebar_position: 1.5
---
# How to stream runnables

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/#chat-models)
- [LangChain Expression Language](/docs/concepts/#langchain-expression-language)
- [Output parsers](/docs/concepts/#output-parsers)

:::

Streaming is critical in making applications based on LLMs feel responsive to end-users.

Important LangChain primitives like [chat models](/docs/concepts/#chat-models), [output parsers](/docs/concepts/#output-parsers), [prompts](/docs/concepts/#prompt-templates), [retrievers](/docs/concepts/#retrievers), and [agents](/docs/concepts/#agents) implement the LangChain [Runnable Interface](/docs/concepts#interface).

This interface provides two general approaches to stream content:

1. sync `stream` and async `astream`: a **default implementation** of streaming that streams the **final output** from the chain.
2. async `astream_events` and async `astream_log`: these provide a way to stream both **intermediate steps** and **final output** from the chain.

Let's take a look at both approaches, and try to understand how to use them.

## Using Stream

All `Runnable` objects implement a sync method called `stream` and an async variant called `astream`. 

These methods are designed to stream the final output in chunks, yielding each chunk as soon as it is available.

Streaming is only possible if all steps in the program know how to process an **input stream**; i.e., process an input chunk one at a time, and yield a corresponding output chunk.

The complexity of this processing can vary, from straightforward tasks like emitting tokens produced by an LLM, to more challenging ones like streaming parts of JSON results before the entire JSON is complete.

The best place to start exploring streaming is with the single most important components in LLMs apps-- the LLMs themselves!

### LLMs and Chat Models

Large language models and their chat variants are the primary bottleneck in LLM based apps.

Large language models can take **several seconds** to generate a complete response to a query. This is far slower than the **~200-300 ms** threshold at which an application feels responsive to an end user.

The key strategy to make the application feel more responsive is to show intermediate progress; viz., to stream the output from the model **token by token**.

We will show examples of streaming using a chat model. Choose one from the options below:

```{=mdx}
import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs
  customVarName="model"
/>
```


```python
# | output: false
# | echo: false

%pip install -qU langchain langchain_anthropic

import os
from getpass import getpass

keys = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
]

for key in keys:
    if key not in os.environ:
        os.environ[key] = getpass(f"Enter API Key for {key}=?")


from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
```

    
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m A new release of pip is available: [0m[31;49m23.2.1[0m[39;49m -> [0m[32;49m24.0[0m
    [1m[[0m[34;49mnotice[0m[1;39;49m][0m[39;49m To update, run: [0m[32;49mpython -m pip install --upgrade pip[0m
    Note: you may need to restart the kernel to use updated packages.


Let's start with the sync `stream` API:


```python
chunks = []
for chunk in model.stream("what color is the sky?"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)
```

    The| sky| appears| blue| during| the| day|.|

Alternatively, if you're working in an async environment, you may consider using the async `astream` API:


```python
chunks = []
async for chunk in model.astream("what color is the sky?"):
    chunks.append(chunk)
    print(chunk.content, end="|", flush=True)
```

    The| sky| appears| blue| during| the| day|.|

Let's inspect one of the chunks


```python
chunks[0]
```




    AIMessageChunk(content='The', id='run-b36bea64-5511-4d7a-b6a3-a07b3db0c8e7')



We got back something called an `AIMessageChunk`. This chunk represents a part of an `AIMessage`.

Message chunks are additive by design -- one can simply add them up to get the state of the response so far!


```python
chunks[0] + chunks[1] + chunks[2] + chunks[3] + chunks[4]
```




    AIMessageChunk(content='The sky appears blue during', id='run-b36bea64-5511-4d7a-b6a3-a07b3db0c8e7')



### Chains

Virtually all LLM applications involve more steps than just a call to a language model.

Let's build a simple chain using `LangChain Expression Language` (`LCEL`) that combines a prompt, model and a parser and verify that streaming works.

We will use [`StrOutputParser`](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html) to parse the output from the model. This is a simple parser that extracts the `content` field from an `AIMessageChunk`, giving us the `token` returned by the model.

:::{.callout-tip}
LCEL is a *declarative* way to specify a "program" by chainining together different LangChain primitives. Chains created using LCEL benefit from an automatic implementation of `stream` and `astream` allowing streaming of the final output. In fact, chains created with LCEL implement the entire standard Runnable interface.
:::


```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_template("tell me a joke about {topic}")
parser = StrOutputParser()
chain = prompt | model | parser

async for chunk in chain.astream({"topic": "parrot"}):
    print(chunk, end="|", flush=True)
```

    Here|'s| a| joke| about| a| par|rot|:|
    
    A man| goes| to| a| pet| shop| to| buy| a| par|rot|.| The| shop| owner| shows| him| two| stunning| pa|rr|ots| with| beautiful| pl|um|age|.|
    
    "|There|'s| a| talking| par|rot| an|d a| non|-|talking| par|rot|,"| the| owner| says|.| "|The| talking| par|rot| costs| $|100|,| an|d the| non|-|talking| par|rot| is| $|20|."|
    
    The| man| says|,| "|I|'ll| take| the| non|-|talking| par|rot| at| $|20|."|
    
    He| pays| an|d leaves| with| the| par|rot|.| As| he|'s| walking| down| the| street|,| the| par|rot| looks| up| at| him| an|d says|,| "|You| know|,| you| really| are| a| stupi|d man|!"|
    
    The| man| is| stun|ne|d an|d looks| at| the| par|rot| in| dis|bel|ief|.| The| par|rot| continues|,| "|Yes|,| you| got| r|ippe|d off| big| time|!| I| can| talk| just| as| well| as| that| other| par|rot|,| an|d you| only| pai|d $|20| |for| me|!"|

Note that we're getting streaming output even though we're using `parser` at the end of the chain above. The `parser` operates on each streaming chunk individidually. Many of the [LCEL primitives](/docs/how_to#langchain-expression-language-lcel) also support this kind of transform-style passthrough streaming, which can be very convenient when constructing apps. 

Custom functions can be [designed to return generators](/docs/how_to/functions#streaming), which are able to operate on streams.

Certain runnables, like [prompt templates](/docs/how_to#prompt-templates) and [chat models](/docs/how_to#chat-models), cannot process individual chunks and instead aggregate all previous steps. Such runnables can interrupt the streaming process.

:::{.callout-note}
The LangChain Expression language allows you to separate the construction of a chain from the mode in which it is used (e.g., sync/async, batch/streaming etc.). If this is not relevant to what you're building, you can also rely on a standard **imperative** programming approach by
caling `invoke`, `batch` or `stream` on each component individually, assigning the results to variables and then using them downstream as you see fit.

:::

### Working with Input Streams

What if you wanted to stream JSON from the output as it was being generated?

If you were to rely on `json.loads` to parse the partial json, the parsing would fail as the partial json wouldn't be valid json.

You'd likely be at a complete loss of what to do and claim that it wasn't possible to stream JSON.

Well, turns out there is a way to do it -- the parser needs to operate on the **input stream**, and attempt to "auto-complete" the partial json into a valid state.

Let's see such a parser in action to understand what this means.


```python
from langchain_core.output_parsers import JsonOutputParser

chain = (
    model | JsonOutputParser()
)  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models
async for text in chain.astream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`"
):
    print(text, flush=True)
```

    {}
    {'countries': []}
    {'countries': [{}]}
    {'countries': [{'name': ''}]}
    {'countries': [{'name': 'France'}]}
    {'countries': [{'name': 'France', 'population': 67}]}
    {'countries': [{'name': 'France', 'population': 67413}]}
    {'countries': [{'name': 'France', 'population': 67413000}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': ''}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain'}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain', 'population': 47}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain', 'population': 47351}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain', 'population': 47351567}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain', 'population': 47351567}, {}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain', 'population': 47351567}, {'name': ''}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain', 'population': 47351567}, {'name': 'Japan'}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain', 'population': 47351567}, {'name': 'Japan', 'population': 125}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain', 'population': 47351567}, {'name': 'Japan', 'population': 125584}]}
    {'countries': [{'name': 'France', 'population': 67413000}, {'name': 'Spain', 'population': 47351567}, {'name': 'Japan', 'population': 125584000}]}


Now, let's **break** streaming. We'll use the previous example and append an extraction function at the end that extracts the country names from the finalized JSON.

:::{.callout-warning}
Any steps in the chain that operate on **finalized inputs** rather than on **input streams** can break streaming functionality via `stream` or `astream`.
:::

:::{.callout-tip}
Later, we will discuss the `astream_events` API which streams results from intermediate steps. This API will stream results from intermediate steps even if the chain contains steps that only operate on **finalized inputs**.
:::


```python
from langchain_core.output_parsers import (
    JsonOutputParser,
)


# A function that operates on finalized inputs
# rather than on an input_stream
def _extract_country_names(inputs):
    """A function that does not operates on input streams and breaks streaming."""
    if not isinstance(inputs, dict):
        return ""

    if "countries" not in inputs:
        return ""

    countries = inputs["countries"]

    if not isinstance(countries, list):
        return ""

    country_names = [
        country.get("name") for country in countries if isinstance(country, dict)
    ]
    return country_names


chain = model | JsonOutputParser() | _extract_country_names

async for text in chain.astream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`"
):
    print(text, end="|", flush=True)
```

    ['France', 'Spain', 'Japan']|

#### Generator Functions

Le'ts fix the streaming using a generator function that can operate on the **input stream**.

:::{.callout-tip}
A generator function (a function that uses `yield`) allows writing code that operates on **input streams**
:::


```python
from langchain_core.output_parsers import JsonOutputParser


async def _extract_country_names_streaming(input_stream):
    """A function that operates on input streams."""
    country_names_so_far = set()

    async for input in input_stream:
        if not isinstance(input, dict):
            continue

        if "countries" not in input:
            continue

        countries = input["countries"]

        if not isinstance(countries, list):
            continue

        for country in countries:
            name = country.get("name")
            if not name:
                continue
            if name not in country_names_so_far:
                yield name
                country_names_so_far.add(name)


chain = model | JsonOutputParser() | _extract_country_names_streaming

async for text in chain.astream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
):
    print(text, end="|", flush=True)
```

    France|Spain|Japan|

:::{.callout-note}
Because the code above is relying on JSON auto-completion, you may see partial names of countries (e.g., `Sp` and `Spain`), which is not what one would want for an extraction result!

We're focusing on streaming concepts, not necessarily the results of the chains.
:::

### Non-streaming components

Some built-in components like Retrievers do not offer any `streaming`. What happens if we try to `stream` them? 🤨


```python
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

vectorstore = FAISS.from_texts(
    ["harrison worked at kensho", "harrison likes spicy food"],
    embedding=OpenAIEmbeddings(),
)
retriever = vectorstore.as_retriever()

chunks = [chunk for chunk in retriever.stream("where did harrison work?")]
chunks
```




    [[Document(page_content='harrison worked at kensho'),
      Document(page_content='harrison likes spicy food')]]



Stream just yielded the final result from that component.

This is OK 🥹! Not all components have to implement streaming -- in some cases streaming is either unnecessary, difficult or just doesn't make sense.

:::{.callout-tip}
An LCEL chain constructed using non-streaming components, will still be able to stream in a lot of cases, with streaming of partial output starting after the last non-streaming step in the chain.
:::


```python
retrieval_chain = (
    {
        "context": retriever.with_config(run_name="Docs"),
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)
```


```python
for chunk in retrieval_chain.stream(
    "Where did harrison work? " "Write 3 made up sentences about this place."
):
    print(chunk, end="|", flush=True)
```

    Base|d on| the| given| context|,| Harrison| worke|d at| K|ens|ho|.|
    
    Here| are| |3| |made| up| sentences| about| this| place|:|
    
    1|.| K|ens|ho| was| a| cutting|-|edge| technology| company| known| for| its| innovative| solutions| in| artificial| intelligence| an|d data| analytics|.|
    
    2|.| The| modern| office| space| at| K|ens|ho| feature|d open| floor| plans|,| collaborative| work|sp|aces|,| an|d a| vib|rant| atmosphere| that| fos|tere|d creativity| an|d team|work|.|
    
    3|.| With| its| prime| location| in| the| heart| of| the| city|,| K|ens|ho| attracte|d top| talent| from| aroun|d the| worl|d,| creating| a| diverse| an|d dynamic| work| environment|.|

Now that we've seen how `stream` and `astream` work, let's venture into the world of streaming events. 🏞️

## Using Stream Events

Event Streaming is a **beta** API. This API may change a bit based on feedback.

:::{.callout-note}

This guide demonstrates the `V2` API and requires langchain-core >= 0.2. For the `V1` API compatible with older versions of LangChain, see [here](https://python.langchain.com/v0.1/docs/expression_language/streaming/#using-stream-events).
:::


```python
import langchain_core

langchain_core.__version__
```

For the `astream_events` API to work properly:

* Use `async` throughout the code to the extent possible (e.g., async tools etc)
* Propagate callbacks if defining custom functions / runnables
* Whenever using runnables without LCEL, make sure to call `.astream()` on LLMs rather than `.ainvoke` to force the LLM to stream tokens.
* Let us know if anything doesn't work as expected! :)

### Event Reference

Below is a reference table that shows some events that might be emitted by the various Runnable objects.


:::{.callout-note}
When streaming is implemented properly, the inputs to a runnable will not be known until after the input stream has been entirely consumed. This means that `inputs` will often be included only for `end` events and rather than for `start` events.
:::

| event                | name             | chunk                           | input                                         | output                                          |
|----------------------|------------------|---------------------------------|-----------------------------------------------|-------------------------------------------------|
| on_chat_model_start  | [model name]     |                                 | {"messages": [[SystemMessage, HumanMessage]]} |                                                 |
| on_chat_model_stream | [model name]     | AIMessageChunk(content="hello") |                                               |                                                 |
| on_chat_model_end    | [model name]     |                                 | {"messages": [[SystemMessage, HumanMessage]]} | AIMessageChunk(content="hello world")           |
| on_llm_start         | [model name]     |                                 | {'input': 'hello'}                            |                                                 |
| on_llm_stream        | [model name]     | 'Hello'                         |                                               |                                                 |
| on_llm_end           | [model name]     |                                 | 'Hello human!'                                |                                                 |
| on_chain_start       | format_docs      |                                 |                                               |                                                 |
| on_chain_stream      | format_docs      | "hello world!, goodbye world!"  |                                               |                                                 |
| on_chain_end         | format_docs      |                                 | [Document(...)]                               | "hello world!, goodbye world!"                  |
| on_tool_start        | some_tool        |                                 | {"x": 1, "y": "2"}                            |                                                 |
| on_tool_end          | some_tool        |                                 |                                               | {"x": 1, "y": "2"}                              |
| on_retriever_start   | [retriever name] |                                 | {"query": "hello"}                            |                                                 |
| on_retriever_end     | [retriever name] |                                 | {"query": "hello"}                            | [Document(...), ..]                             |
| on_prompt_start      | [template_name]  |                                 | {"question": "hello"}                         |                                                 |
| on_prompt_end        | [template_name]  |                                 | {"question": "hello"}                         | ChatPromptValue(messages: [SystemMessage, ...]) |

### Chat Model

Let's start off by looking at the events produced by a chat model.


```python
events = []
async for event in model.astream_events("hello", version="v2"):
    events.append(event)
```

    /home/eugene/src/langchain/libs/core/langchain_core/_api/beta_decorator.py:87: LangChainBetaWarning: This API is in beta and may change in the future.
      warn_beta(


:::{.callout-note}

Hey what's that funny version="v2" parameter in the API?! 😾

This is a **beta API**, and we're almost certainly going to make some changes to it (in fact, we already have!)

This version parameter will allow us to minimize such breaking changes to your code. 

In short, we are annoying you now, so we don't have to annoy you later.

`v2` is only available for langchain-core>=0.2.0.

:::

Let's take a look at the few of the start event and a few of the end events.


```python
events[:3]
```




    [{'event': 'on_chat_model_start',
      'data': {'input': 'hello'},
      'name': 'ChatAnthropic',
      'tags': [],
      'run_id': 'a81e4c0f-fc36-4d33-93bc-1ac25b9bb2c3',
      'metadata': {}},
     {'event': 'on_chat_model_stream',
      'data': {'chunk': AIMessageChunk(content='Hello', id='run-a81e4c0f-fc36-4d33-93bc-1ac25b9bb2c3')},
      'run_id': 'a81e4c0f-fc36-4d33-93bc-1ac25b9bb2c3',
      'name': 'ChatAnthropic',
      'tags': [],
      'metadata': {}},
     {'event': 'on_chat_model_stream',
      'data': {'chunk': AIMessageChunk(content='!', id='run-a81e4c0f-fc36-4d33-93bc-1ac25b9bb2c3')},
      'run_id': 'a81e4c0f-fc36-4d33-93bc-1ac25b9bb2c3',
      'name': 'ChatAnthropic',
      'tags': [],
      'metadata': {}}]




```python
events[-2:]
```




    [{'event': 'on_chat_model_stream',
      'data': {'chunk': AIMessageChunk(content='?', id='run-a81e4c0f-fc36-4d33-93bc-1ac25b9bb2c3')},
      'run_id': 'a81e4c0f-fc36-4d33-93bc-1ac25b9bb2c3',
      'name': 'ChatAnthropic',
      'tags': [],
      'metadata': {}},
     {'event': 'on_chat_model_end',
      'data': {'output': AIMessageChunk(content='Hello! How can I assist you today?', id='run-a81e4c0f-fc36-4d33-93bc-1ac25b9bb2c3')},
      'run_id': 'a81e4c0f-fc36-4d33-93bc-1ac25b9bb2c3',
      'name': 'ChatAnthropic',
      'tags': [],
      'metadata': {}}]



### Chain

Let's revisit the example chain that parsed streaming JSON to explore the streaming events API.


```python
chain = (
    model | JsonOutputParser()
)  # Due to a bug in older versions of Langchain, JsonOutputParser did not stream results from some models

events = [
    event
    async for event in chain.astream_events(
        "output a list of the countries france, spain and japan and their populations in JSON format. "
        'Use a dict with an outer key of "countries" which contains a list of countries. '
        "Each country should have the key `name` and `population`",
        version="v2",
    )
]
```

If you examine at the first few events, you'll notice that there are **3** different start events rather than **2** start events.

The three start events correspond to:

1. The chain (model + parser)
2. The model
3. The parser


```python
events[:3]
```




    [{'event': 'on_chain_start',
      'data': {'input': 'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'},
      'name': 'RunnableSequence',
      'tags': [],
      'run_id': '4765006b-16e2-4b1d-a523-edd9fd64cb92',
      'metadata': {}},
     {'event': 'on_chat_model_start',
      'data': {'input': {'messages': [[HumanMessage(content='output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`')]]}},
      'name': 'ChatAnthropic',
      'tags': ['seq:step:1'],
      'run_id': '0320c234-7b52-4a14-ae4e-5f100949e589',
      'metadata': {}},
     {'event': 'on_chat_model_stream',
      'data': {'chunk': AIMessageChunk(content='{', id='run-0320c234-7b52-4a14-ae4e-5f100949e589')},
      'run_id': '0320c234-7b52-4a14-ae4e-5f100949e589',
      'name': 'ChatAnthropic',
      'tags': ['seq:step:1'],
      'metadata': {}}]



What do you think you'd see if you looked at the last 3 events? what about the middle?

Let's use this API to take output the stream events from the model and the parser. We're ignoring start events, end events and events from the chain.


```python
num_events = 0

async for event in chain.astream_events(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
    version="v2",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(
            f"Chat model chunk: {repr(event['data']['chunk'].content)}",
            flush=True,
        )
    if kind == "on_parser_stream":
        print(f"Parser chunk: {event['data']['chunk']}", flush=True)
    num_events += 1
    if num_events > 30:
        # Truncate the output
        print("...")
        break
```

    Chat model chunk: '{'
    Parser chunk: {}
    Chat model chunk: '\n  '
    Chat model chunk: '"'
    Chat model chunk: 'countries'
    Chat model chunk: '":'
    Chat model chunk: ' ['
    Parser chunk: {'countries': []}
    Chat model chunk: '\n    '
    Chat model chunk: '{'
    Parser chunk: {'countries': [{}]}
    Chat model chunk: '\n      '
    Chat model chunk: '"'
    Chat model chunk: 'name'
    Chat model chunk: '":'
    Chat model chunk: ' "'
    Parser chunk: {'countries': [{'name': ''}]}
    Chat model chunk: 'France'
    Parser chunk: {'countries': [{'name': 'France'}]}
    Chat model chunk: '",'
    Chat model chunk: '\n      '
    Chat model chunk: '"'
    Chat model chunk: 'population'
    ...


Because both the model and the parser support streaming, we see sreaming events from both components in real time! Kind of cool isn't it? 🦜

### Filtering Events

Because this API produces so many events, it is useful to be able to filter on events.

You can filter by either component `name`, component `tags` or component `type`.

#### By Name


```python
chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
    {"run_name": "my_parser"}
)

max_events = 0
async for event in chain.astream_events(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
    version="v2",
    include_names=["my_parser"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break
```

    {'event': 'on_parser_start', 'data': {'input': 'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'}, 'name': 'my_parser', 'tags': ['seq:step:2'], 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {'countries': []}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {'countries': [{}]}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {'countries': [{'name': ''}]}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {'countries': [{'name': 'France'}]}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {'countries': [{'name': 'France', 'population': 67}]}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {'countries': [{'name': 'France', 'population': 67413}]}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {'countries': [{'name': 'France', 'population': 67413000}]}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {'countries': [{'name': 'France', 'population': 67413000}, {}]}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {'countries': [{'name': 'France', 'population': 67413000}, {'name': ''}]}}, 'run_id': 'e058d750-f2c2-40f6-aa61-10f84cd671a9', 'name': 'my_parser', 'tags': ['seq:step:2'], 'metadata': {}}
    ...


#### By Type


```python
chain = model.with_config({"run_name": "model"}) | JsonOutputParser().with_config(
    {"run_name": "my_parser"}
)

max_events = 0
async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    version="v2",
    include_types=["chat_model"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break
```

    {'event': 'on_chat_model_start', 'data': {'input': 'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'}, 'name': 'model', 'tags': ['seq:step:1'], 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='{', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='\n  ', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='"', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='countries', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='":', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content=' [', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='\n    ', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='{', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='\n      ', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='"', id='run-db246792-2a91-4eb3-a14b-29658947065d')}, 'run_id': 'db246792-2a91-4eb3-a14b-29658947065d', 'name': 'model', 'tags': ['seq:step:1'], 'metadata': {}}
    ...


#### By Tags

:::{.callout-caution}

Tags are inherited by child components of a given runnable. 

If you're using tags to filter, make sure that this is what you want.
:::


```python
chain = (model | JsonOutputParser()).with_config({"tags": ["my_chain"]})

max_events = 0
async for event in chain.astream_events(
    'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`',
    version="v2",
    include_tags=["my_chain"],
):
    print(event)
    max_events += 1
    if max_events > 10:
        # Truncate output
        print("...")
        break
```

    {'event': 'on_chain_start', 'data': {'input': 'output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`'}, 'name': 'RunnableSequence', 'tags': ['my_chain'], 'run_id': 'fd68dd64-7a4d-4bdb-a0c2-ee592db0d024', 'metadata': {}}
    {'event': 'on_chat_model_start', 'data': {'input': {'messages': [[HumanMessage(content='output a list of the countries france, spain and japan and their populations in JSON format. Use a dict with an outer key of "countries" which contains a list of countries. Each country should have the key `name` and `population`')]]}}, 'name': 'ChatAnthropic', 'tags': ['seq:step:1', 'my_chain'], 'run_id': 'efd3c8af-4be5-4f6c-9327-e3f9865dd1cd', 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='{', id='run-efd3c8af-4be5-4f6c-9327-e3f9865dd1cd')}, 'run_id': 'efd3c8af-4be5-4f6c-9327-e3f9865dd1cd', 'name': 'ChatAnthropic', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}}
    {'event': 'on_parser_start', 'data': {}, 'name': 'JsonOutputParser', 'tags': ['seq:step:2', 'my_chain'], 'run_id': 'afde30b9-beac-4b36-b4c7-dbbe423ddcdb', 'metadata': {}}
    {'event': 'on_parser_stream', 'data': {'chunk': {}}, 'run_id': 'afde30b9-beac-4b36-b4c7-dbbe423ddcdb', 'name': 'JsonOutputParser', 'tags': ['seq:step:2', 'my_chain'], 'metadata': {}}
    {'event': 'on_chain_stream', 'data': {'chunk': {}}, 'run_id': 'fd68dd64-7a4d-4bdb-a0c2-ee592db0d024', 'name': 'RunnableSequence', 'tags': ['my_chain'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='\n  ', id='run-efd3c8af-4be5-4f6c-9327-e3f9865dd1cd')}, 'run_id': 'efd3c8af-4be5-4f6c-9327-e3f9865dd1cd', 'name': 'ChatAnthropic', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='"', id='run-efd3c8af-4be5-4f6c-9327-e3f9865dd1cd')}, 'run_id': 'efd3c8af-4be5-4f6c-9327-e3f9865dd1cd', 'name': 'ChatAnthropic', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='countries', id='run-efd3c8af-4be5-4f6c-9327-e3f9865dd1cd')}, 'run_id': 'efd3c8af-4be5-4f6c-9327-e3f9865dd1cd', 'name': 'ChatAnthropic', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content='":', id='run-efd3c8af-4be5-4f6c-9327-e3f9865dd1cd')}, 'run_id': 'efd3c8af-4be5-4f6c-9327-e3f9865dd1cd', 'name': 'ChatAnthropic', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}}
    {'event': 'on_chat_model_stream', 'data': {'chunk': AIMessageChunk(content=' [', id='run-efd3c8af-4be5-4f6c-9327-e3f9865dd1cd')}, 'run_id': 'efd3c8af-4be5-4f6c-9327-e3f9865dd1cd', 'name': 'ChatAnthropic', 'tags': ['seq:step:1', 'my_chain'], 'metadata': {}}
    ...


### Non-streaming components

Remember how some components don't stream well because they don't operate on **input streams**?

While such components can break streaming of the final output when using `astream`, `astream_events` will still yield streaming events from intermediate steps that support streaming!


```python
# Function that does not support streaming.
# It operates on the finalizes inputs rather than
# operating on the input stream.
def _extract_country_names(inputs):
    """A function that does not operates on input streams and breaks streaming."""
    if not isinstance(inputs, dict):
        return ""

    if "countries" not in inputs:
        return ""

    countries = inputs["countries"]

    if not isinstance(countries, list):
        return ""

    country_names = [
        country.get("name") for country in countries if isinstance(country, dict)
    ]
    return country_names


chain = (
    model | JsonOutputParser() | _extract_country_names
)  # This parser only works with OpenAI right now
```

As expected, the `astream` API doesn't work correctly because `_extract_country_names` doesn't operate on streams.


```python
async for chunk in chain.astream(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
):
    print(chunk, flush=True)
```

    ['France', 'Spain', 'Japan']


Now, let's confirm that with astream_events we're still seeing streaming output from the model and the parser.


```python
num_events = 0

async for event in chain.astream_events(
    "output a list of the countries france, spain and japan and their populations in JSON format. "
    'Use a dict with an outer key of "countries" which contains a list of countries. '
    "Each country should have the key `name` and `population`",
    version="v2",
):
    kind = event["event"]
    if kind == "on_chat_model_stream":
        print(
            f"Chat model chunk: {repr(event['data']['chunk'].content)}",
            flush=True,
        )
    if kind == "on_parser_stream":
        print(f"Parser chunk: {event['data']['chunk']}", flush=True)
    num_events += 1
    if num_events > 30:
        # Truncate the output
        print("...")
        break
```

    Chat model chunk: '{'
    Parser chunk: {}
    Chat model chunk: '\n  '
    Chat model chunk: '"'
    Chat model chunk: 'countries'
    Chat model chunk: '":'
    Chat model chunk: ' ['
    Parser chunk: {'countries': []}
    Chat model chunk: '\n    '
    Chat model chunk: '{'
    Parser chunk: {'countries': [{}]}
    Chat model chunk: '\n      '
    Chat model chunk: '"'
    Chat model chunk: 'name'
    Chat model chunk: '":'
    Chat model chunk: ' "'
    Parser chunk: {'countries': [{'name': ''}]}
    Chat model chunk: 'France'
    Parser chunk: {'countries': [{'name': 'France'}]}
    Chat model chunk: '",'
    Chat model chunk: '\n      '
    Chat model chunk: '"'
    Chat model chunk: 'population'
    Chat model chunk: '":'
    Chat model chunk: ' '
    Chat model chunk: '67'
    Parser chunk: {'countries': [{'name': 'France', 'population': 67}]}
    ...


### Propagating Callbacks

:::{.callout-caution}
If you're using invoking runnables inside your tools, you need to propagate callbacks to the runnable; otherwise, no stream events will be generated.
:::

:::{.callout-note}
When using `RunnableLambdas` or `@chain` decorator, callbacks are propagated automatically behind the scenes.
:::


```python
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool


def reverse_word(word: str):
    return word[::-1]


reverse_word = RunnableLambda(reverse_word)


@tool
def bad_tool(word: str):
    """Custom tool that doesn't propagate callbacks."""
    return reverse_word.invoke(word)


async for event in bad_tool.astream_events("hello", version="v2"):
    print(event)
```

    {'event': 'on_tool_start', 'data': {'input': 'hello'}, 'name': 'bad_tool', 'tags': [], 'run_id': 'ea900472-a8f7-425d-b627-facdef936ee8', 'metadata': {}}
    {'event': 'on_chain_start', 'data': {'input': 'hello'}, 'name': 'reverse_word', 'tags': [], 'run_id': '77b01284-0515-48f4-8d7c-eb27c1882f86', 'metadata': {}}
    {'event': 'on_chain_end', 'data': {'output': 'olleh', 'input': 'hello'}, 'run_id': '77b01284-0515-48f4-8d7c-eb27c1882f86', 'name': 'reverse_word', 'tags': [], 'metadata': {}}
    {'event': 'on_tool_end', 'data': {'output': 'olleh'}, 'run_id': 'ea900472-a8f7-425d-b627-facdef936ee8', 'name': 'bad_tool', 'tags': [], 'metadata': {}}


Here's a re-implementation that does propagate callbacks correctly. You'll notice that now we're getting events from the `reverse_word` runnable as well.


```python
@tool
def correct_tool(word: str, callbacks):
    """A tool that correctly propagates callbacks."""
    return reverse_word.invoke(word, {"callbacks": callbacks})


async for event in correct_tool.astream_events("hello", version="v2"):
    print(event)
```

    {'event': 'on_tool_start', 'data': {'input': 'hello'}, 'name': 'correct_tool', 'tags': [], 'run_id': 'd5ea83b9-9278-49cc-9f1d-aa302d671040', 'metadata': {}}
    {'event': 'on_chain_start', 'data': {'input': 'hello'}, 'name': 'reverse_word', 'tags': [], 'run_id': '44dafbf4-2f87-412b-ae0e-9f71713810df', 'metadata': {}}
    {'event': 'on_chain_end', 'data': {'output': 'olleh', 'input': 'hello'}, 'run_id': '44dafbf4-2f87-412b-ae0e-9f71713810df', 'name': 'reverse_word', 'tags': [], 'metadata': {}}
    {'event': 'on_tool_end', 'data': {'output': 'olleh'}, 'run_id': 'd5ea83b9-9278-49cc-9f1d-aa302d671040', 'name': 'correct_tool', 'tags': [], 'metadata': {}}


If you're invoking runnables from within Runnable Lambdas or `@chains`, then callbacks will be passed automatically on your behalf.


```python
from langchain_core.runnables import RunnableLambda


async def reverse_and_double(word: str):
    return await reverse_word.ainvoke(word) * 2


reverse_and_double = RunnableLambda(reverse_and_double)

await reverse_and_double.ainvoke("1234")

async for event in reverse_and_double.astream_events("1234", version="v2"):
    print(event)
```

    {'event': 'on_chain_start', 'data': {'input': '1234'}, 'name': 'reverse_and_double', 'tags': [], 'run_id': '03b0e6a1-3e60-42fc-8373-1e7829198d80', 'metadata': {}}
    {'event': 'on_chain_start', 'data': {'input': '1234'}, 'name': 'reverse_word', 'tags': [], 'run_id': '5cf26fc8-840b-4642-98ed-623dda28707a', 'metadata': {}}
    {'event': 'on_chain_end', 'data': {'output': '4321', 'input': '1234'}, 'run_id': '5cf26fc8-840b-4642-98ed-623dda28707a', 'name': 'reverse_word', 'tags': [], 'metadata': {}}
    {'event': 'on_chain_stream', 'data': {'chunk': '43214321'}, 'run_id': '03b0e6a1-3e60-42fc-8373-1e7829198d80', 'name': 'reverse_and_double', 'tags': [], 'metadata': {}}
    {'event': 'on_chain_end', 'data': {'output': '43214321'}, 'run_id': '03b0e6a1-3e60-42fc-8373-1e7829198d80', 'name': 'reverse_and_double', 'tags': [], 'metadata': {}}


And with the `@chain` decorator:


```python
from langchain_core.runnables import chain


@chain
async def reverse_and_double(word: str):
    return await reverse_word.ainvoke(word) * 2


await reverse_and_double.ainvoke("1234")

async for event in reverse_and_double.astream_events("1234", version="v2"):
    print(event)
```

    {'event': 'on_chain_start', 'data': {'input': '1234'}, 'name': 'reverse_and_double', 'tags': [], 'run_id': '1bfcaedc-f4aa-4d8e-beee-9bba6ef17008', 'metadata': {}}
    {'event': 'on_chain_start', 'data': {'input': '1234'}, 'name': 'reverse_word', 'tags': [], 'run_id': '64fc99f0-5d7d-442b-b4f5-4537129f67d1', 'metadata': {}}
    {'event': 'on_chain_end', 'data': {'output': '4321', 'input': '1234'}, 'run_id': '64fc99f0-5d7d-442b-b4f5-4537129f67d1', 'name': 'reverse_word', 'tags': [], 'metadata': {}}
    {'event': 'on_chain_stream', 'data': {'chunk': '43214321'}, 'run_id': '1bfcaedc-f4aa-4d8e-beee-9bba6ef17008', 'name': 'reverse_and_double', 'tags': [], 'metadata': {}}
    {'event': 'on_chain_end', 'data': {'output': '43214321'}, 'run_id': '1bfcaedc-f4aa-4d8e-beee-9bba6ef17008', 'name': 'reverse_and_double', 'tags': [], 'metadata': {}}


## Next steps

Now you've learned some ways to stream both final outputs and internal steps with LangChain.

To learn more, check out the other how-to guides in this section, or the [conceptual guide on Langchain Expression Language](/docs/concepts/#langchain-expression-language/).
