# How to call tools with multi-modal data

Here we demonstrate how to call tools with multi-modal data, such as images.

Some multi-modal models, such as those that can reason over images or audio, support [tool calling](/docs/concepts/#functiontool-calling) features as well.

To call tools using such models, simply bind tools to them in the [usual way](/docs/how_to/tool_calling), and invoke the model using content blocks of the desired type (e.g., containing image data).

Below, we demonstrate examples using [OpenAI](/docs/integrations/platforms/openai) and [Anthropic](/docs/integrations/platforms/anthropic). We will use the same image and tool in all cases. Let's first select an image, and build a placeholder tool that expects as input the string "sunny", "cloudy", or "rainy". We will ask the models to describe the weather in the image.


```python
from typing import Literal

from langchain_core.tools import tool

image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"


@tool
def weather_tool(weather: Literal["sunny", "cloudy", "rainy"]) -> None:
    """Describe the weather"""
    pass
```

## OpenAI

For OpenAI, we can feed the image URL directly in a content block of type "image_url":


```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o").bind_tools([weather_tool])

message = HumanMessage(
    content=[
        {"type": "text", "text": "describe the weather in this image"},
        {"type": "image_url", "image_url": {"url": image_url}},
    ],
)
response = model.invoke([message])
print(response.tool_calls)
```

    [{'name': 'weather_tool', 'args': {'weather': 'sunny'}, 'id': 'call_mRYL50MtHdeNuNIjSCm5UPmB'}]


Note that we recover tool calls with parsed arguments in LangChain's [standard format](/docs/how_to/tool_calling) in the model response.

## Anthropic

For Anthropic, we can format a base64-encoded image into a content block of type "image", as below:


```python
import base64

import httpx
from langchain_anthropic import ChatAnthropic

image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")

model = ChatAnthropic(model="claude-3-sonnet-20240229").bind_tools([weather_tool])

message = HumanMessage(
    content=[
        {"type": "text", "text": "describe the weather in this image"},
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_data,
            },
        },
    ],
)
response = model.invoke([message])
print(response.tool_calls)
```

    [{'name': 'weather_tool', 'args': {'weather': 'sunny'}, 'id': 'toolu_016m9KfknJqx5fVRYk4tkF6s'}]

