# How to use a model to call tools

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [Chat models](/docs/concepts/#chat-models)
- [LangChain Tools](/docs/concepts/#tools)

:::

```{=mdx}
:::info
We use the term tool calling interchangeably with function calling. Although
function calling is sometimes meant to refer to invocations of a single function,
we treat all models as though they can return multiple tool or function calls in 
each message.
:::
```

Tool calling allows a chat model to respond to a given prompt by "calling a tool".
While the name implies that the model is performing 
some action, this is actually not the case! The model generates the 
arguments to a tool, and actually running the tool (or not) is up to the user.
For example, if you want to [extract output matching some schema](/docs/how_to/structured_output/) 
from unstructured text, you could give the model an "extraction" tool that takes 
parameters matching the desired schema, then treat the generated output as your final 
result.

However, tool calling goes beyond [structured output](/docs/how_to/structured_output/)
since you can pass responses from called tools back to the model to create longer interactions.
For instance, given a search engine tool, an LLM might handle a 
query by first issuing a call to the search engine with arguments. The system calling the LLM can 
receive the tool call, execute it, and return the output to the LLM to inform its 
response. LangChain includes a suite of [built-in tools](/docs/integrations/tools/) 
and supports several methods for defining your own [custom tools](/docs/how_to/custom_tools). 

Tool calling is not universal, but many popular LLM providers, including [Anthropic](https://www.anthropic.com/), 
[Cohere](https://cohere.com/), [Google](https://cloud.google.com/vertex-ai), 
[Mistral](https://mistral.ai/), [OpenAI](https://openai.com/), and others, 
support variants of a tool calling feature.

LangChain implements standard interfaces for defining tools, passing them to LLMs, 
and representing tool calls. This guide will show you how to use them.

## Passing tools to chat models

Chat models that support tool calling features implement a `.bind_tools` method, which 
receives a list of LangChain [tool objects](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.BaseTool.html#langchain_core.tools.BaseTool) 
and binds them to the chat model in its expected format. Subsequent invocations of the 
chat model will include tool schemas in its calls to the LLM.

For example, we can define the schema for custom tools using the `@tool` decorator 
on Python functions:


```python
from langchain_core.tools import tool


@tool
def add(a: int, b: int) -> int:
    """Adds a and b."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b."""
    return a * b


tools = [add, multiply]
```

Or below, we define the schema using [Pydantic](https://docs.pydantic.dev):


```python
from langchain_core.pydantic_v1 import BaseModel, Field


# Note that the docstrings here are crucial, as they will be passed along
# to the model along with the class name.
class Add(BaseModel):
    """Add two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


class Multiply(BaseModel):
    """Multiply two integers together."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")


tools = [Add, Multiply]
```

We can bind them to chat models as follows:

```{=mdx}
import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs
  customVarName="llm"
  fireworksParams={`model="accounts/fireworks/models/firefunction-v1", temperature=0`}
/>
```

We'll use the `.bind_tools()` method to handle converting
`Multiply` to the proper format for the model, then and bind it (i.e.,
passing it in each time the model is invoked).


```python
# | output: false
# | echo: false

%pip install -qU langchain langchain_openai

import os
from getpass import getpass

from langchain_openai import ChatOpenAI

os.environ["OPENAI_API_KEY"] = getpass()

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
```


```python
llm_with_tools = llm.bind_tools(tools)
```

## Tool calls

If tool calls are included in a LLM response, they are attached to the corresponding 
[message](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessage.html#langchain_core.messages.ai.AIMessage) 
or [message chunk](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) 
as a list of [tool call](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolCall.html#langchain_core.messages.tool.ToolCall) 
objects in the `.tool_calls` attribute.

Note that chat models can call multiple tools at once.

A `ToolCall` is a typed dict that includes a 
tool name, dict of argument values, and (optionally) an identifier. Messages with no 
tool calls default to an empty list for this attribute.


```python
query = "What is 3 * 12? Also, what is 11 + 49?"

llm_with_tools.invoke(query).tool_calls
```




    [{'name': 'Multiply',
      'args': {'a': 3, 'b': 12},
      'id': 'call_KquHA7mSbgtAkpkmRPaFnJKa'},
     {'name': 'Add',
      'args': {'a': 11, 'b': 49},
      'id': 'call_Fl0hQi4IBTzlpaJYlM5kPQhE'}]



The `.tool_calls` attribute should contain valid tool calls. Note that on occasion, 
model providers may output malformed tool calls (e.g., arguments that are not 
valid JSON). When parsing fails in these cases, instances 
of [InvalidToolCall](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.InvalidToolCall.html#langchain_core.messages.tool.InvalidToolCall) 
are populated in the `.invalid_tool_calls` attribute. An `InvalidToolCall` can have 
a name, string arguments, identifier, and error message.

If desired, [output parsers](/docs/how_to#output-parsers) can further 
process the output. For example, we can convert back to the original Pydantic class:


```python
from langchain_core.output_parsers.openai_tools import PydanticToolsParser

chain = llm_with_tools | PydanticToolsParser(tools=[Multiply, Add])
chain.invoke(query)
```




    [Multiply(a=3, b=12), Add(a=11, b=49)]



### Streaming

When tools are called in a streaming context, 
[message chunks](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) 
will be populated with [tool call chunk](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.tool.ToolCallChunk.html#langchain_core.messages.tool.ToolCallChunk) 
objects in a list via the `.tool_call_chunks` attribute. A `ToolCallChunk` includes 
optional string fields for the tool `name`, `args`, and `id`, and includes an optional 
integer field `index` that can be used to join chunks together. Fields are optional 
because portions of a tool call may be streamed across different chunks (e.g., a chunk 
that includes a substring of the arguments may have null values for the tool name and id).

Because message chunks inherit from their parent message class, an 
[AIMessageChunk](https://api.python.langchain.com/en/latest/messages/langchain_core.messages.ai.AIMessageChunk.html#langchain_core.messages.ai.AIMessageChunk) 
with tool call chunks will also include `.tool_calls` and `.invalid_tool_calls` fields. 
These fields are parsed best-effort from the message's tool call chunks.

Note that not all providers currently support streaming for tool calls:


```python
async for chunk in llm_with_tools.astream(query):
    print(chunk.tool_call_chunks)
```

    []
    [{'name': 'Multiply', 'args': '', 'id': 'call_3aQwTP9CYlFxwOvQZPHDu6wL', 'index': 0}]
    [{'name': None, 'args': '{"a"', 'id': None, 'index': 0}]
    [{'name': None, 'args': ': 3, ', 'id': None, 'index': 0}]
    [{'name': None, 'args': '"b": 1', 'id': None, 'index': 0}]
    [{'name': None, 'args': '2}', 'id': None, 'index': 0}]
    [{'name': 'Add', 'args': '', 'id': 'call_SQUoSsJz2p9Kx2x73GOgN1ja', 'index': 1}]
    [{'name': None, 'args': '{"a"', 'id': None, 'index': 1}]
    [{'name': None, 'args': ': 11,', 'id': None, 'index': 1}]
    [{'name': None, 'args': ' "b": ', 'id': None, 'index': 1}]
    [{'name': None, 'args': '49}', 'id': None, 'index': 1}]
    []


Note that adding message chunks will merge their corresponding tool call chunks. This is the principle by which LangChain's various [tool output parsers](/docs/how_to/output_parser_structured) support streaming.

For example, below we accumulate tool call chunks:


```python
first = True
async for chunk in llm_with_tools.astream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    print(gathered.tool_call_chunks)
```

    []
    [{'name': 'Multiply', 'args': '', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}]
    [{'name': 'Multiply', 'args': '{"a"', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}]
    [{'name': 'Multiply', 'args': '{"a": 3, ', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}]
    [{'name': 'Multiply', 'args': '{"a": 3, "b": 1', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}]
    [{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}]
    [{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}, {'name': 'Add', 'args': '', 'id': 'call_b4iMiB3chGNGqbt5SjqqD2Wh', 'index': 1}]
    [{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}, {'name': 'Add', 'args': '{"a"', 'id': 'call_b4iMiB3chGNGqbt5SjqqD2Wh', 'index': 1}]
    [{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}, {'name': 'Add', 'args': '{"a": 11,', 'id': 'call_b4iMiB3chGNGqbt5SjqqD2Wh', 'index': 1}]
    [{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}, {'name': 'Add', 'args': '{"a": 11, "b": ', 'id': 'call_b4iMiB3chGNGqbt5SjqqD2Wh', 'index': 1}]
    [{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}, {'name': 'Add', 'args': '{"a": 11, "b": 49}', 'id': 'call_b4iMiB3chGNGqbt5SjqqD2Wh', 'index': 1}]
    [{'name': 'Multiply', 'args': '{"a": 3, "b": 12}', 'id': 'call_AkL3dVeCjjiqvjv8ckLxL3gP', 'index': 0}, {'name': 'Add', 'args': '{"a": 11, "b": 49}', 'id': 'call_b4iMiB3chGNGqbt5SjqqD2Wh', 'index': 1}]



```python
print(type(gathered.tool_call_chunks[0]["args"]))
```

    <class 'str'>


And below we accumulate tool calls to demonstrate partial parsing:


```python
first = True
async for chunk in llm_with_tools.astream(query):
    if first:
        gathered = chunk
        first = False
    else:
        gathered = gathered + chunk

    print(gathered.tool_calls)
```

    []
    []
    [{'name': 'Multiply', 'args': {}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}]
    [{'name': 'Multiply', 'args': {'a': 3}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}]
    [{'name': 'Multiply', 'args': {'a': 3, 'b': 1}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}]
    [{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}]
    [{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}]
    [{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}, {'name': 'Add', 'args': {}, 'id': 'call_54Hx3DGjZitFlEjgMe1DYonh'}]
    [{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}, {'name': 'Add', 'args': {'a': 11}, 'id': 'call_54Hx3DGjZitFlEjgMe1DYonh'}]
    [{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}, {'name': 'Add', 'args': {'a': 11}, 'id': 'call_54Hx3DGjZitFlEjgMe1DYonh'}]
    [{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}, {'name': 'Add', 'args': {'a': 11, 'b': 49}, 'id': 'call_54Hx3DGjZitFlEjgMe1DYonh'}]
    [{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_4p0D4tHVXSiae9Mu0e8jlI1m'}, {'name': 'Add', 'args': {'a': 11, 'b': 49}, 'id': 'call_54Hx3DGjZitFlEjgMe1DYonh'}]



```python
print(type(gathered.tool_calls[0]["args"]))
```

    <class 'dict'>


## Passing tool outputs to the model

If we're using the model-generated tool invocations to actually call tools and want to pass the tool results back to the model, we can do so using `ToolMessage`s.


```python
from langchain_core.messages import HumanMessage, ToolMessage

messages = [HumanMessage(query)]
ai_msg = llm_with_tools.invoke(messages)
messages.append(ai_msg)
for tool_call in ai_msg.tool_calls:
    selected_tool = {"add": add, "multiply": multiply}[tool_call["name"].lower()]
    tool_output = selected_tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
messages
```




    [HumanMessage(content='What is 3 * 12? Also, what is 11 + 49?'),
     AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_svc2GLSxNFALbaCAbSjMI9J8', 'function': {'arguments': '{"a": 3, "b": 12}', 'name': 'Multiply'}, 'type': 'function'}, {'id': 'call_r8jxte3zW6h3MEGV3zH2qzFh', 'function': {'arguments': '{"a": 11, "b": 49}', 'name': 'Add'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 50, 'prompt_tokens': 105, 'total_tokens': 155}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_d9767fc5b9', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-a79ad1dd-95f1-4a46-b688-4c83f327a7b3-0', tool_calls=[{'name': 'Multiply', 'args': {'a': 3, 'b': 12}, 'id': 'call_svc2GLSxNFALbaCAbSjMI9J8'}, {'name': 'Add', 'args': {'a': 11, 'b': 49}, 'id': 'call_r8jxte3zW6h3MEGV3zH2qzFh'}]),
     ToolMessage(content='36', tool_call_id='call_svc2GLSxNFALbaCAbSjMI9J8'),
     ToolMessage(content='60', tool_call_id='call_r8jxte3zW6h3MEGV3zH2qzFh')]




```python
llm_with_tools.invoke(messages)
```




    AIMessage(content='3 * 12 is 36 and 11 + 49 is 60.', response_metadata={'token_usage': {'completion_tokens': 18, 'prompt_tokens': 171, 'total_tokens': 189}, 'model_name': 'gpt-3.5-turbo-0125', 'system_fingerprint': 'fp_d9767fc5b9', 'finish_reason': 'stop', 'logprobs': None}, id='run-20b52149-e00d-48ea-97cf-f8de7a255f8c-0')



Note that we pass back the same `id` in the `ToolMessage` as the what we receive from the model in order to help the model match tool responses with tool calls.

## Few-shot prompting

For more complex tool use it's very useful to add few-shot examples to the prompt. We can do this by adding `AIMessage`s with `ToolCall`s and corresponding `ToolMessage`s to our prompt.

For example, even with some special instructions our model can get tripped up by order of operations:


```python
llm_with_tools.invoke(
    "Whats 119 times 8 minus 20. Don't do any math yourself, only use tools for math. Respect order of operations"
).tool_calls
```




    [{'name': 'Multiply',
      'args': {'a': 119, 'b': 8},
      'id': 'call_T88XN6ECucTgbXXkyDeC2CQj'},
     {'name': 'Add',
      'args': {'a': 952, 'b': -20},
      'id': 'call_licdlmGsRqzup8rhqJSb1yZ4'}]



The model shouldn't be trying to add anything yet, since it technically can't know the results of 119 * 8 yet.

By adding a prompt with some examples we can correct this behavior:


```python
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

examples = [
    HumanMessage(
        "What's the product of 317253 and 128472 plus four", name="example_user"
    ),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[
            {"name": "Multiply", "args": {"x": 317253, "y": 128472}, "id": "1"}
        ],
    ),
    ToolMessage("16505054784", tool_call_id="1"),
    AIMessage(
        "",
        name="example_assistant",
        tool_calls=[{"name": "Add", "args": {"x": 16505054784, "y": 4}, "id": "2"}],
    ),
    ToolMessage("16505054788", tool_call_id="2"),
    AIMessage(
        "The product of 317253 and 128472 plus four is 16505054788",
        name="example_assistant",
    ),
]

system = """You are bad at math but are an expert at using a calculator. 

Use past tool usage as an example of how to correctly use the tools."""
few_shot_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{query}"),
    ]
)

chain = {"query": RunnablePassthrough()} | few_shot_prompt | llm_with_tools
chain.invoke("Whats 119 times 8 minus 20").tool_calls
```




    [{'name': 'Multiply',
      'args': {'a': 119, 'b': 8},
      'id': 'call_9MvuwQqg7dlJupJcoTWiEsDo'}]



And we get the correct output this time.

Here's what the [LangSmith trace](https://smith.langchain.com/public/f70550a1-585f-4c9d-a643-13148ab1616f/r) looks like.

## Binding model-specific formats (advanced)

Providers adopt different conventions for formatting tool schemas. 
For instance, OpenAI uses a format like this:

- `type`: The type of the tool. At the time of writing, this is always `"function"`.
- `function`: An object containing tool parameters.
- `function.name`: The name of the schema to output.
- `function.description`: A high level description of the schema to output.
- `function.parameters`: The nested details of the schema you want to extract, formatted as a [JSON schema](https://json-schema.org/) dict.

We can bind this model-specific format directly to the model as well if preferred. Here's an example:


```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI()

model_with_tools = model.bind(
    tools=[
        {
            "type": "function",
            "function": {
                "name": "multiply",
                "description": "Multiply two integers together.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {"type": "number", "description": "First integer"},
                        "b": {"type": "number", "description": "Second integer"},
                    },
                    "required": ["a", "b"],
                },
            },
        }
    ]
)

model_with_tools.invoke("Whats 119 times 8?")
```




    AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_mn4ELw1NbuE0DFYhIeK0GrPe', 'function': {'arguments': '{"a":119,"b":8}', 'name': 'multiply'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 17, 'prompt_tokens': 62, 'total_tokens': 79}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_c2295e73ad', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-353e8a9a-7125-4f94-8c68-4f3da4c21120-0', tool_calls=[{'name': 'multiply', 'args': {'a': 119, 'b': 8}, 'id': 'call_mn4ELw1NbuE0DFYhIeK0GrPe'}])



This is functionally equivalent to the `bind_tools()` calls above.

## Next steps

Now you've learned how to bind tool schemas to a chat model and to call those tools. Next, check out some more specific uses of tool calling:

- Building [tool-using chains and agents](/docs/how_to#tools)
- Getting [structured outputs](/docs/how_to/structured_output/) from models
