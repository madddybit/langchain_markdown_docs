# How to add a human-in-the-loop for tools

There are certain tools that we don't trust a model to execute on its own. One thing we can do in such situations is require human approval before the tool is invoked.

:::{.callout-info}

This how-to guide shows a simple way to add human-in-the-loop for code running in a jupyter notebook or in a terminal.

To build a production application, you will need to do more work to keep track of application state appropriately.

We recommend using `langgraph` for powering such a capability. For more details, please see this [guide](https://langchain-ai.github.io/langgraph/how-tos/human-in-the-loop/).
:::


## Setup

We'll need to install the following packages:


```python
%pip install --upgrade --quiet langchain
```

And set these environment variables:


```python
import getpass
import os

# If you'd like to use LangSmith, uncomment the below:
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

## Chain

Let's create a few simple (dummy) tools and a tool-calling chain:

```{=mdx}
import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs customVarName="llm"/>
```


```python
# | output: false
# | echo: false

from langchain_anthropic import ChatAnthropic

llm = ChatAnthropic(model="claude-3-sonnet-20240229", temperature=0)
```


```python
from typing import Dict, List

from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import tool


@tool
def count_emails(last_n_days: int) -> int:
    """Multiply two integers together."""
    return last_n_days * 2


@tool
def send_email(message: str, recipient: str) -> str:
    "Add two integers."
    return f"Successfully sent email to {recipient}."


tools = [count_emails, send_email]
llm_with_tools = llm.bind_tools(tools)


def call_tools(msg: AIMessage) -> List[Dict]:
    """Simple sequential tool calling helper."""
    tool_map = {tool.name: tool for tool in tools}
    tool_calls = msg.tool_calls.copy()
    for tool_call in tool_calls:
        tool_call["output"] = tool_map[tool_call["name"]].invoke(tool_call["args"])
    return tool_calls


chain = llm_with_tools | call_tools
chain.invoke("how many emails did i get in the last 5 days?")
```




    [{'name': 'count_emails',
      'args': {'last_n_days': 5},
      'id': 'toolu_01QYZdJ4yPiqsdeENWHqioFW',
      'output': 10}]



## Adding human approval

Let's add a step in the chain that will ask a person to approve or reject the tall call request.

On rejection, the step will raise an exception which will stop execution of the rest of the chain.


```python
import json


class NotApproved(Exception):
    """Custom exception."""


def human_approval(msg: AIMessage) -> AIMessage:
    """Responsible for passing through its input or raising an exception.

    Args:
        msg: output from the chat model

    Returns:
        msg: original output from the msg
    """
    tool_strs = "\n\n".join(
        json.dumps(tool_call, indent=2) for tool_call in msg.tool_calls
    )
    input_msg = (
        f"Do you approve of the following tool invocations\n\n{tool_strs}\n\n"
        "Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.\n >>>"
    )
    resp = input(input_msg)
    if resp.lower() not in ("yes", "y"):
        raise NotApproved(f"Tool invocations not approved:\n\n{tool_strs}")
    return msg
```


```python
chain = llm_with_tools | human_approval | call_tools
chain.invoke("how many emails did i get in the last 5 days?")
```

    Do you approve of the following tool invocations
    
    {
      "name": "count_emails",
      "args": {
        "last_n_days": 5
      },
      "id": "toolu_01WbD8XeMoQaRFtsZezfsHor"
    }
    
    Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.
     >>> yes





    [{'name': 'count_emails',
      'args': {'last_n_days': 5},
      'id': 'toolu_01WbD8XeMoQaRFtsZezfsHor',
      'output': 10}]




```python
try:
    chain.invoke("Send sally@gmail.com an email saying 'What's up homie'")
except NotApproved as e:
    print()
    print(e)
```

    Do you approve of the following tool invocations
    
    {
      "name": "send_email",
      "args": {
        "recipient": "sally@gmail.com",
        "message": "What's up homie"
      },
      "id": "toolu_014XccHFzBiVcc9GV1harV9U"
    }
    
    Anything except 'Y'/'Yes' (case-insensitive) will be treated as a no.
     >>> no


    
    Tool invocations not approved:
    
    {
      "name": "send_email",
      "args": {
        "recipient": "sally@gmail.com",
        "message": "What's up homie"
      },
      "id": "toolu_014XccHFzBiVcc9GV1harV9U"
    }

