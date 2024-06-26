# How to stream responses from an LLM

All `LLM`s implement the [Runnable interface](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable), which comes with **default** implementations of standard runnable methods (i.e. `ainvoke`, `batch`, `abatch`, `stream`, `astream`, `astream_events`).

The **default** streaming implementations provide an`Iterator` (or `AsyncIterator` for asynchronous streaming) that yields a single value: the final output from the underlying chat model provider.

The ability to stream the output token-by-token depends on whether the provider has implemented proper streaming support.

See which [integrations support token-by-token streaming here](/docs/integrations/llms/).



:::{.callout-note}

The **default** implementation does **not** provide support for token-by-token streaming, but it ensures that the model can be swapped in for any other model as it supports the same standard interface.

:::

## Sync stream

Below we use a `|` to help visualize the delimiter between tokens.


```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)
for chunk in llm.stream("Write me a 1 verse song about sparkling water."):
    print(chunk, end="|", flush=True)
```

    
    
    |Spark|ling| water|,| oh| so clear|
    |Bubbles dancing|,| without| fear|
    |Refreshing| taste|,| a| pure| delight|
    |Spark|ling| water|,| my| thirst|'s| delight||

## Async streaming

Let's see how to stream in an async setting using `astream`.


```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)
async for chunk in llm.astream("Write me a 1 verse song about sparkling water."):
    print(chunk, end="|", flush=True)
```

    
    
    |Spark|ling| water|,| oh| so clear|
    |Bubbles dancing|,| without| fear|
    |Refreshing| taste|,| a| pure| delight|
    |Spark|ling| water|,| my| thirst|'s| delight||

## Async event streaming


LLMs also support the standard [astream events](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events) method.

:::{.callout-tip}

`astream_events` is most useful when implementing streaming in a larger LLM application that contains multiple steps (e.g., an application that involves an `agent`).
:::


```python
from langchain_openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0, max_tokens=512)

idx = 0

async for event in llm.astream_events(
    "Write me a 1 verse song about goldfish on the moon", version="v1"
):
    idx += 1
    if idx >= 5:  # Truncate the output
        print("...Truncated")
        break
    print(event)
```
