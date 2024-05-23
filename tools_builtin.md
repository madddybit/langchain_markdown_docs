---
sidebar_position: 4
sidebar_class_name: hidden
---
# How to use built-in tools and toolkits

:::info Prerequisites

This guide assumes familiarity with the following concepts:

- [LangChain Tools](/docs/concepts/#tools)
- [LangChain Toolkits](/docs/concepts/#tools)

:::

## Tools

LangChain has a large collection of 3rd party tools. Please visit [Tool Integrations](/docs/integrations/tools/) for a list of the available tools.

:::{.callout-important}

When using 3rd party tools, make sure that you understand how the tool works, what permissions
it has. Read over its documentation and check if anything is required from you
from a security point of view. Please see our [security](https://python.langchain.com/v0.1/docs/security/) 
guidelines for more information.

:::

Let's try out the [Wikipedia integration](/docs/integrations/tools/wikipedia/).


```python
!pip install -qU wikipedia
```


```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=100)
tool = WikipediaQueryRun(api_wrapper=api_wrapper)

print(tool.invoke({"query": "langchain"}))
```

    Page: LangChain
    Summary: LangChain is a framework designed to simplify the creation of applications 


The tool has the following defaults associated with it:


```python
print(f"Name: {tool.name}")
print(f"Description: {tool.description}")
print(f"args schema: {tool.args}")
print(f"returns directly?: {tool.return_direct}")
```

    Name: wiki-tool
    Description: look up things in wikipedia
    args schema: {'query': {'title': 'Query', 'description': 'query to look up in Wikipedia, should be 3 or less words', 'type': 'string'}}
    returns directly?: True


## Customizing Default Tools
We can also modify the built in name, description, and JSON schema of the arguments.

When defining the JSON schema of the arguments, it is important that the inputs remain the same as the function, so you shouldn't change that. But you can define custom descriptions for each input easily.


```python
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.pydantic_v1 import BaseModel, Field


class WikiInputs(BaseModel):
    """Inputs to the wikipedia tool."""

    query: str = Field(
        description="query to look up in Wikipedia, should be 3 or less words"
    )


tool = WikipediaQueryRun(
    name="wiki-tool",
    description="look up things in wikipedia",
    args_schema=WikiInputs,
    api_wrapper=api_wrapper,
    return_direct=True,
)

print(tool.run("langchain"))
```

    Page: LangChain
    Summary: LangChain is a framework designed to simplify the creation of applications 



```python
print(f"Name: {tool.name}")
print(f"Description: {tool.description}")
print(f"args schema: {tool.args}")
print(f"returns directly?: {tool.return_direct}")
```

    Name: wiki-tool
    Description: look up things in wikipedia
    args schema: {'query': {'title': 'Query', 'description': 'query to look up in Wikipedia, should be 3 or less words', 'type': 'string'}}
    returns directly?: True


## How to use built-in toolkits

Toolkits are collections of tools that are designed to be used together for specific tasks. They have convenient loading methods.

For a complete list of available ready-made toolkits, visit [Integrations](/docs/integrations/toolkits/).

All Toolkits expose a `get_tools` method which returns a list of tools.

You're usually meant to use them this way:

```python
# Initialize a toolkit
toolkit = ExampleTookit(...)

# Get list of tools
tools = toolkit.get_tools()
```
