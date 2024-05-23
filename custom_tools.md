# How to create custom tools

When constructing an agent, you will need to provide it with a list of `Tool`s that it can use. Besides the actual function that is called, the Tool consists of several components:

| Attribute       | Type                      | Description                                                                                                      |
|-----------------|---------------------------|------------------------------------------------------------------------------------------------------------------|
| name          | str                     | Must be unique within a set of tools provided to an LLM or agent.                                           |
| description   | str                     | Describes what the tool does. Used as context by the LLM or agent.                                       |
| args_schema   | Pydantic BaseModel      | Optional but recommended, can be used to provide more information (e.g., few-shot examples) or validation for expected parameters |
| return_direct   | boolean      | Only relevant for agents. When True, after invoking the given tool, the agent will stop and return the result direcly to the user.  |

LangChain provides 3 ways to create tools:

1. Using [@tool decorator](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.tool.html#langchain_core.tools.tool) -- the simplest way to define a custom tool.
2. Using [StructuredTool.from_function](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.StructuredTool.html#langchain_core.tools.StructuredTool.from_function) class method -- this is similar to the `@tool` decorator, but allows more configuration and specification of both sync and async implementations.
3. By sub-classing from [BaseTool](https://api.python.langchain.com/en/latest/tools/langchain_core.tools.BaseTool.html) -- This is the most flexible method, it provides the largest degree of control, at the expense of more effort and code.

The `@tool` or the `StructuredTool.from_function` class method should be sufficient for most use cases.

:::{.callout-tip}

Models will perform better if the tools have well chosen names, descriptions and JSON schemas.
:::

## @tool decorator

This `@tool` decorator is the simplest way to define a custom tool. The decorator uses the function name as the tool name by default, but this can be overridden by passing a string as the first argument. Additionally, the decorator will use the function's docstring as the tool's description - so a docstring MUST be provided. 


```python
from langchain_core.tools import tool


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Let's inspect some of the attributes associated with the tool.
print(multiply.name)
print(multiply.description)
print(multiply.args)
```

    multiply
    multiply(a: int, b: int) -> int - Multiply two numbers.
    {'a': {'title': 'A', 'type': 'integer'}, 'b': {'title': 'B', 'type': 'integer'}}


Or create an **async** implementation, like this:


```python
from langchain_core.tools import tool


@tool
async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b
```

You can also customize the tool name and JSON args by passing them into the tool decorator.


```python
from langchain.pydantic_v1 import BaseModel, Field


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


@tool("multiplication-tool", args_schema=CalculatorInput, return_direct=True)
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


# Let's inspect some of the attributes associated with the tool.
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.return_direct)
```

    multiplication-tool
    multiplication-tool(a: int, b: int) -> int - Multiply two numbers.
    {'a': {'title': 'A', 'description': 'first number', 'type': 'integer'}, 'b': {'title': 'B', 'description': 'second number', 'type': 'integer'}}
    True


## StructuredTool

The `StrurcturedTool.from_function` class method provides a bit more configurability than the `@tool` decorator, without requiring much additional code.


```python
from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

print(calculator.invoke({"a": 2, "b": 3}))
print(await calculator.ainvoke({"a": 2, "b": 5}))
```

    6
    10


To configure it:


```python
class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(
    func=multiply,
    name="Calculator",
    description="multiply numbers",
    args_schema=CalculatorInput,
    return_direct=True,
    # coroutine= ... <- you can specify an async method if desired as well
)

print(calculator.invoke({"a": 2, "b": 3}))
print(calculator.name)
print(calculator.description)
print(calculator.args)
```

    6
    Calculator
    Calculator(a: int, b: int) -> int - multiply numbers
    {'a': {'title': 'A', 'description': 'first number', 'type': 'integer'}, 'b': {'title': 'B', 'description': 'second number', 'type': 'integer'}}


## Subclass BaseTool

You can define a custom tool by sub-classing from `BaseTool`. This provides maximal control over the tool definition, but requires writing more code.


```python
from typing import Optional, Type

from langchain.pydantic_v1 import BaseModel
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool


class CalculatorInput(BaseModel):
    a: int = Field(description="first number")
    b: int = Field(description="second number")


class CustomCalculatorTool(BaseTool):
    name = "Calculator"
    description = "useful for when you need to answer questions about math"
    args_schema: Type[BaseModel] = CalculatorInput
    return_direct: bool = True

    def _run(
        self, a: int, b: int, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        return a * b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the tool asynchronously."""
        # If the calculation is cheap, you can just delegate to the sync implementation
        # as shown below.
        # If the sync calculation is expensive, you should delete the entire _arun method.
        # LangChain will automatically provide a better implementation that will
        # kick off the task in a thread to make sure it doesn't block other async code.
        return self._run(a, b, run_manager=run_manager.get_sync())
```


```python
multiply = CustomCalculatorTool()
print(multiply.name)
print(multiply.description)
print(multiply.args)
print(multiply.return_direct)

print(multiply.invoke({"a": 2, "b": 3}))
print(await multiply.ainvoke({"a": 2, "b": 3}))
```

    Calculator
    useful for when you need to answer questions about math
    {'a': {'title': 'A', 'description': 'first number', 'type': 'integer'}, 'b': {'title': 'B', 'description': 'second number', 'type': 'integer'}}
    True
    6
    6


## How to create async tools

LangChain Tools implement the [Runnable interface 🏃](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html).

All Runnables expose the `invoke` and `ainvoke` methods (as well as other methods like `batch`, `abatch`, `astream` etc).

So even if you only provide an `sync` implementation of a tool, you could still use the `ainvoke` interface, but there
are some important things to know:

* LangChain's by default provides an async implementation that assumes that the function is expensive to compute, so it'll delegate execution to another thread.
* If you're working in an async codebase, you should create async tools rather than sync tools, to avoid incuring a small overhead due to that thread.
* If you need both sync and async implementations, use `StructuredTool.from_function` or sub-class from `BaseTool`.
* If implementing both sync and async, and the sync code is fast to run, override the default LangChain async implementation and simply call the sync code.
* You CANNOT and SHOULD NOT use the sync `invoke` with an `async` tool.


```python
from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply)

print(calculator.invoke({"a": 2, "b": 3}))
print(
    await calculator.ainvoke({"a": 2, "b": 5})
)  # Uses default LangChain async implementation incurs small overhead
```

    6
    10



```python
from langchain_core.tools import StructuredTool


def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


async def amultiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(func=multiply, coroutine=amultiply)

print(calculator.invoke({"a": 2, "b": 3}))
print(
    await calculator.ainvoke({"a": 2, "b": 5})
)  # Uses use provided amultiply without additional overhead
```

    6
    10


You should not and cannot use `.invoke` when providing only an async definition.


```python
@tool
async def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


try:
    multiply.invoke({"a": 2, "b": 3})
except NotImplementedError:
    print("Raised not implemented error. You should not be doing this.")
```

    Raised not implemented error. You should not be doing this.


## Handling Tool Errors 

If you're using tools with agents, you will likely need an error handling strategy, so the agent can recover from the error and continue execution.

A simple strategy is to throw a `ToolException` from inside the tool and specify an error handler using `handle_tool_error`. 

When the error handler is specified, the exception will be caught and the error handler will decide which output to return from the tool.

You can set `handle_tool_error` to `True`, a string value, or a function. If it's a function, the function should take a `ToolException` as a parameter and return a value.

Please note that only raising a `ToolException` won't be effective. You need to first set the `handle_tool_error` of the tool because its default value is `False`.


```python
from langchain_core.tools import ToolException


def get_weather(city: str) -> int:
    """Get weather for the given city."""
    raise ToolException(f"Error: There is no city by the name of {city}.")
```

Here's an example with the default `handle_tool_error=True` behavior.


```python
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=True,
)

get_weather_tool.invoke({"city": "foobar"})
```




    'Error: There is no city by the name of foobar.'



We can set `handle_tool_error` to a string that will always be returned.


```python
get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error="There is no such city, but it's probably above 0K there!",
)

get_weather_tool.invoke({"city": "foobar"})
```




    "There is no such city, but it's probably above 0K there!"



Handling the error using a function:


```python
def _handle_error(error: ToolException) -> str:
    return f"The following errors occurred during tool execution: `{error.args[0]}`"


get_weather_tool = StructuredTool.from_function(
    func=get_weather,
    handle_tool_error=_handle_error,
)

get_weather_tool.invoke({"city": "foobar"})
```




    'The following errors occurred during tool execution: `Error: There is no city by the name of foobar.`'


