# How to handle tool errors

Using a model to invoke a tool has some obvious potential failure modes. Firstly, the model needs to return a output that can be parsed at all. Secondly, the model needs to return tool arguments that are valid.

We can build error handling into our chains to mitigate these failure modes.

## Setup

We'll need to install the following packages:


```python
%pip install --upgrade --quiet langchain-core langchain-openai
```

If you'd like to trace your runs in [LangSmith](https://docs.smith.langchain.com/) uncomment and set the following environment variables:


```python
import getpass
import os

# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

## Chain

Suppose we have the following (dummy) tool and tool-calling chain. We'll make our tool intentionally convoluted to try and trip up the model.

```{=mdx}
import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs customVarName="llm"/>
```


```python
# | echo: false
# | output: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
```


```python
# Define tool
from langchain_core.tools import tool


@tool
def complex_tool(int_arg: int, float_arg: float, dict_arg: dict) -> int:
    """Do something complex with a complex tool."""
    return int_arg * float_arg
```


```python
llm_with_tools = llm.bind_tools(
    [complex_tool],
)
```


```python
# Define chain
chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool
```

We can see that when we try to invoke this chain with even a fairly explicit input, the model fails to correctly call the tool (it forgets the `dict_arg` argument).


```python
chain.invoke(
    "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
)
```


    ---------------------------------------------------------------------------

    ValidationError                           Traceback (most recent call last)

    Cell In[12], line 1
    ----> 1 chain.invoke(
          2     "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
          3 )


    File ~/langchain/libs/core/langchain_core/runnables/base.py:2499, in RunnableSequence.invoke(self, input, config)
       2497 try:
       2498     for i, step in enumerate(self.steps):
    -> 2499         input = step.invoke(
       2500             input,
       2501             # mark each step as a child run
       2502             patch_config(
       2503                 config, callbacks=run_manager.get_child(f"seq:step:{i+1}")
       2504             ),
       2505         )
       2506 # finish the root run
       2507 except BaseException as e:


    File ~/langchain/libs/core/langchain_core/tools.py:241, in BaseTool.invoke(self, input, config, **kwargs)
        234 def invoke(
        235     self,
        236     input: Union[str, Dict],
        237     config: Optional[RunnableConfig] = None,
        238     **kwargs: Any,
        239 ) -> Any:
        240     config = ensure_config(config)
    --> 241     return self.run(
        242         input,
        243         callbacks=config.get("callbacks"),
        244         tags=config.get("tags"),
        245         metadata=config.get("metadata"),
        246         run_name=config.get("run_name"),
        247         run_id=config.pop("run_id", None),
        248         **kwargs,
        249     )


    File ~/langchain/libs/core/langchain_core/tools.py:387, in BaseTool.run(self, tool_input, verbose, start_color, color, callbacks, tags, metadata, run_name, run_id, **kwargs)
        385 except ValidationError as e:
        386     if not self.handle_validation_error:
    --> 387         raise e
        388     elif isinstance(self.handle_validation_error, bool):
        389         observation = "Tool input validation error"


    File ~/langchain/libs/core/langchain_core/tools.py:378, in BaseTool.run(self, tool_input, verbose, start_color, color, callbacks, tags, metadata, run_name, run_id, **kwargs)
        364 run_manager = callback_manager.on_tool_start(
        365     {"name": self.name, "description": self.description},
        366     tool_input if isinstance(tool_input, str) else str(tool_input),
       (...)
        375     **kwargs,
        376 )
        377 try:
    --> 378     parsed_input = self._parse_input(tool_input)
        379     tool_args, tool_kwargs = self._to_args_and_kwargs(parsed_input)
        380     observation = (
        381         self._run(*tool_args, run_manager=run_manager, **tool_kwargs)
        382         if new_arg_supported
        383         else self._run(*tool_args, **tool_kwargs)
        384     )


    File ~/langchain/libs/core/langchain_core/tools.py:283, in BaseTool._parse_input(self, tool_input)
        281 else:
        282     if input_args is not None:
    --> 283         result = input_args.parse_obj(tool_input)
        284         return {
        285             k: getattr(result, k)
        286             for k, v in result.dict().items()
        287             if k in tool_input
        288         }
        289 return tool_input


    File ~/langchain/.venv/lib/python3.9/site-packages/pydantic/v1/main.py:526, in BaseModel.parse_obj(cls, obj)
        524         exc = TypeError(f'{cls.__name__} expected dict not {obj.__class__.__name__}')
        525         raise ValidationError([ErrorWrapper(exc, loc=ROOT_KEY)], cls) from e
    --> 526 return cls(**obj)


    File ~/langchain/.venv/lib/python3.9/site-packages/pydantic/v1/main.py:341, in BaseModel.__init__(__pydantic_self__, **data)
        339 values, fields_set, validation_error = validate_model(__pydantic_self__.__class__, data)
        340 if validation_error:
    --> 341     raise validation_error
        342 try:
        343     object_setattr(__pydantic_self__, '__dict__', values)


    ValidationError: 1 validation error for complex_toolSchema
    dict_arg
      field required (type=value_error.missing)


## Try/except tool call

The simplest way to more gracefully handle errors is to try/except the tool-calling step and return a helpful message on errors:


```python
from typing import Any

from langchain_core.runnables import Runnable, RunnableConfig


def try_except_tool(tool_args: dict, config: RunnableConfig) -> Runnable:
    try:
        complex_tool.invoke(tool_args, config=config)
    except Exception as e:
        return f"Calling tool with arguments:\n\n{tool_args}\n\nraised the following error:\n\n{type(e)}: {e}"


chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | try_except_tool
```


```python
print(
    chain.invoke(
        "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
    )
)
```

    Calling tool with arguments:
    
    {'int_arg': 5, 'float_arg': 2.1}
    
    raised the following error:
    
    <class 'pydantic.v1.error_wrappers.ValidationError'>: 1 validation error for complex_toolSchema
    dict_arg
      field required (type=value_error.missing)


## Fallbacks

We can also try to fallback to a better model in the event of a tool invocation error. In this case we'll fall back to an identical chain that uses `gpt-4-1106-preview` instead of `gpt-3.5-turbo`.


```python
chain = llm_with_tools | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool
better_model = ChatOpenAI(model="gpt-4-1106-preview", temperature=0).bind_tools(
    [complex_tool], tool_choice="complex_tool"
)
better_chain = better_model | (lambda msg: msg.tool_calls[0]["args"]) | complex_tool

chain_with_fallback = chain.with_fallbacks([better_chain])
chain_with_fallback.invoke(
    "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
)
```




    10.5



Looking at the [Langsmith trace](https://smith.langchain.com/public/00e91fc2-e1a4-4b0f-a82e-e6b3119d196c/r) for this chain run, we can see that the first chain call fails as expected and it's the fallback that succeeds.

## Retry with exception

To take things one step further, we can try to automatically re-run the chain with the exception passed in, so that the model may be able to correct its behavior:


```python
import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough


class CustomToolException(Exception):
    """Custom LangChain tool exception."""

    def __init__(self, tool_call: ToolCall, exception: Exception) -> None:
        super().__init__()
        self.tool_call = tool_call
        self.exception = exception


def tool_custom_exception(msg: AIMessage, config: RunnableConfig) -> Runnable:
    try:
        return complex_tool.invoke(msg.tool_calls[0]["args"], config=config)
    except Exception as e:
        raise CustomToolException(msg.tool_calls[0], e)


def exception_to_messages(inputs: dict) -> dict:
    exception = inputs.pop("exception")

    # Add historical messages to the original input, so the model knows that it made a mistake with the last tool call.
    messages = [
        AIMessage(content="", tool_calls=[exception.tool_call]),
        ToolMessage(
            tool_call_id=exception.tool_call["id"], content=str(exception.exception)
        ),
        HumanMessage(
            content="The last tool call raised an exception. Try calling the tool again with corrected arguments. Do not repeat mistakes."
        ),
    ]
    inputs["last_output"] = messages
    return inputs


# We add a last_output MessagesPlaceholder to our prompt which if not passed in doesn't
# affect the prompt at all, but gives us the option to insert an arbitrary list of Messages
# into the prompt if needed. We'll use this on retries to insert the error message.
prompt = ChatPromptTemplate.from_messages(
    [("human", "{input}"), MessagesPlaceholder("last_output", optional=True)]
)
chain = prompt | llm_with_tools | tool_custom_exception

# If the initial chain call fails, we rerun it withe the exception passed in as a message.
self_correcting_chain = chain.with_fallbacks(
    [exception_to_messages | chain], exception_key="exception"
)
```


```python
self_correcting_chain.invoke(
    {
        "input": "use complex tool. the args are 5, 2.1, empty dictionary. don't forget dict_arg"
    }
)
```




    10.5



And our chain succeeds! Looking at the [LangSmith trace](https://smith.langchain.com/public/c11e804c-e14f-4059-bd09-64766f999c14/r), we can see that indeed our initial chain still fails, and it's only on retrying that the chain succeeds.
