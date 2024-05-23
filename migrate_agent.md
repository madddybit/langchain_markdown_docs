# How to migrate from legacy LangChain agents to LangGraph

Here we focus on how to move from legacy LangChain agents to LangGraph agents.
LangChain agents (the [AgentExecutor](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor) in particular) have multiple configuration parameters.
In this notebook we will show how those parameters map to the LangGraph [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent).

#### Prerequisites

This how-to guide uses OpenAI as the LLM. Install the dependencies to run.


```python
%%capture --no-stderr
%pip install -U langgraph langchain langchain-openai
```

## Basic Usage

For basic creation and usage of a tool-calling ReAct-style agent, the functionality is the same. First, let's define a model and tool(s), then we'll use those to create an agent.


```python
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


query = "what is the value of magic_function(3)?"
```

For the LangChain [AgentExecutor](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor), we define a prompt with a placeholder for the agent's scratchpad. The agent can be invoked as follows:


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant"),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke({"input": query})
```




    {'input': 'what is the value of magic_function(3)?',
     'output': 'The value of `magic_function(3)` is 5.'}



LangGraph's [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent) manages a state that is defined by a list of messages. It will continue to process the list until there are no tool calls in the agent's output. To kick it off, we input a list of messages. The output will contain the entire state of the graph-- in this case, the conversation history.




```python
from langgraph.prebuilt import create_react_agent

app = create_react_agent(model, tools)


messages = app.invoke({"messages": [("human", query)]})
{
    "input": query,
    "output": messages["messages"][-1].content,
}
```




    {'input': 'what is the value of magic_function(3)?',
     'output': 'The value of `magic_function(3)` is 5.'}




```python
message_history = messages["messages"]

new_query = "Pardon?"

messages = app.invoke({"messages": message_history + [("human", new_query)]})
{
    "input": new_query,
    "output": messages["messages"][-1].content,
}
```




    {'input': 'Pardon?',
     'output': 'The result of applying `magic_function` to the input 3 is 5.'}



## Prompt Templates

With legacy LangChain agents you have to pass in a prompt template. You can use this to control the agent.

With LangGraph [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent), by default there is no prompt. You can achieve similar control over the agent in a few ways:

1. Pass in a system message as input
2. Initialize the agent with a system message
3. Initialize the agent with a function to transform messages before passing to the model.

Let's take a look at all of these below. We will pass in custom instructions to get the agent to respond in Spanish.

First up, using AgentExecutor:


```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_executor.invoke({"input": query})
```




    {'input': 'what is the value of magic_function(3)?',
     'output': 'El valor de `magic_function(3)` es 5.'}



Now, let's pass a custom system message to [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent). This can either be a string or a LangChain SystemMessage.


```python
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

system_message = "You are a helpful assistant. Respond only in Spanish."
# This could also be a SystemMessage object
# system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")

app = create_react_agent(model, tools, messages_modifier=system_message)


messages = app.invoke({"messages": [("user", query)]})
```

We can also pass in an arbitrary function. This function should take in a list of messages and output a list of messages.
We can do all types of arbitrary formatting of messages here. In this cases, let's just add a SystemMessage to the start of the list of messages.


```python
from langchain_core.messages import AnyMessage
from langgraph.prebuilt import create_react_agent

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("placeholder", "{messages}"),
    ]
)


def _modify_messages(messages: list[AnyMessage]):
    return prompt.invoke({"messages": messages}).to_messages() + [
        ("user", "Also say 'Pandamonium!' after the answer.")
    ]


app = create_react_agent(model, tools, messages_modifier=_modify_messages)


messages = app.invoke({"messages": [("human", query)]})
print(
    {
        "input": query,
        "output": messages["messages"][-1].content,
    }
)
```

    {'input': 'what is the value of magic_function(3)?', 'output': 'El valor de magic_function(3) es 5. ¡Pandamonium!'}


## Memory

With LangChain's [AgentExecutor](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter), you could add chat [Memory](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.memory) so it can engage in a multi-turn conversation.


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")
memory = ChatMessageHistory(session_id="test-session")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        # First put the history
        ("placeholder", "{chat_history}"),
        # Then the new input
        ("human", "{input}"),
        # Finally the scratchpad
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]


agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

agent_with_chat_history = RunnableWithMessageHistory(
    agent_executor,
    # This is needed because in most real world scenarios, a session id is needed
    # It isn't really used here because we are using a simple in memory ChatMessageHistory
    lambda session_id: memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

config = {"configurable": {"session_id": "test-session"}}
print(
    agent_with_chat_history.invoke(
        {"input": "Hi, I'm polly! What's the output of magic_function of 3?"}, config
    )["output"]
)
print("---")
print(agent_with_chat_history.invoke({"input": "Remember my name?"}, config)["output"])
print("---")
print(
    agent_with_chat_history.invoke({"input": "what was that output again?"}, config)[
        "output"
    ]
)
```

    Hi Polly! The output of the magic function for the input 3 is 5.
    ---
    Yes, I remember your name, Polly! How can I assist you further?
    ---
    The output of the magic function for the input 3 is 5.


#### In LangGraph

Memory is just [persistence](https://langchain-ai.github.io/langgraph/how-tos/persistence/), aka [checkpointing](https://langchain-ai.github.io/langgraph/reference/checkpoints/).

Add a `checkpointer` to the agent and you get chat memory for free.


```python
from langchain_core.messages import SystemMessage
from langgraph.checkpoint import MemorySaver  # an in-memory checkpointer
from langgraph.prebuilt import create_react_agent

system_message = "You are a helpful assistant."
# This could also be a SystemMessage object
# system_message = SystemMessage(content="You are a helpful assistant. Respond only in Spanish.")

memory = MemorySaver()
app = create_react_agent(
    model, tools, messages_modifier=system_message, checkpointer=memory
)

config = {"configurable": {"thread_id": "test-thread"}}
print(
    app.invoke(
        {
            "messages": [
                ("user", "Hi, I'm polly! What's the output of magic_function of 3?")
            ]
        },
        config,
    )["messages"][-1].content
)
print("---")
print(
    app.invoke({"messages": [("user", "Remember my name?")]}, config)["messages"][
        -1
    ].content
)
print("---")
print(
    app.invoke({"messages": [("user", "what was that output again?")]}, config)[
        "messages"
    ][-1].content
)
```

    Hi Polly! The output of the magic_function for the input 3 is 5.
    ---
    Yes, your name is Polly!
    ---
    The output of the magic_function for the input 3 was 5.


## Iterating through steps

With LangChain's [AgentExecutor](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter), you could iterate over the steps using the [stream](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.stream) (or async `astream`) methods or the [iter](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter) method. LangGraph supports stepwise iteration using [stream](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.stream) 


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return input + 2


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)

for step in agent_executor.stream({"input": query}):
    print(step)
```

    {'actions': [ToolAgentAction(tool='magic_function', tool_input={'input': 3}, log="\nInvoking: `magic_function` with `{'input': 3}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_q9MgGFjqJbV2xSUX93WqxmOt', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls'}, id='run-c68fd76f-a3c3-4c3c-bfd7-748c171ed4b8', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_q9MgGFjqJbV2xSUX93WqxmOt'}], tool_call_chunks=[{'name': 'magic_function', 'args': '{"input":3}', 'id': 'call_q9MgGFjqJbV2xSUX93WqxmOt', 'index': 0}])], tool_call_id='call_q9MgGFjqJbV2xSUX93WqxmOt')], 'messages': [AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_q9MgGFjqJbV2xSUX93WqxmOt', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls'}, id='run-c68fd76f-a3c3-4c3c-bfd7-748c171ed4b8', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_q9MgGFjqJbV2xSUX93WqxmOt'}], tool_call_chunks=[{'name': 'magic_function', 'args': '{"input":3}', 'id': 'call_q9MgGFjqJbV2xSUX93WqxmOt', 'index': 0}])]}
    {'steps': [AgentStep(action=ToolAgentAction(tool='magic_function', tool_input={'input': 3}, log="\nInvoking: `magic_function` with `{'input': 3}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_q9MgGFjqJbV2xSUX93WqxmOt', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls'}, id='run-c68fd76f-a3c3-4c3c-bfd7-748c171ed4b8', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_q9MgGFjqJbV2xSUX93WqxmOt'}], tool_call_chunks=[{'name': 'magic_function', 'args': '{"input":3}', 'id': 'call_q9MgGFjqJbV2xSUX93WqxmOt', 'index': 0}])], tool_call_id='call_q9MgGFjqJbV2xSUX93WqxmOt'), observation=5)], 'messages': [FunctionMessage(content='5', name='magic_function')]}
    {'output': 'The value of `magic_function(3)` is 5.', 'messages': [AIMessage(content='The value of `magic_function(3)` is 5.')]}


#### In LangGraph

In LangGraph, things are handled natively using [stream](https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.graph.CompiledGraph.stream) or the asynchronous `astream` method.


```python
from langchain_core.messages import AnyMessage
from langgraph.prebuilt import create_react_agent

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("placeholder", "{messages}"),
    ]
)


def _modify_messages(messages: list[AnyMessage]):
    return prompt.invoke({"messages": messages}).to_messages()


app = create_react_agent(model, tools, messages_modifier=_modify_messages)


for step in app.stream({"messages": [("human", query)]}, stream_mode="updates"):
    print(step)
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_yTjXXibj76tyFyPRa1soLo0S', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 70, 'total_tokens': 84}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-b275f314-c42e-4e77-9dec-5c23f7dbd53b-0', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_yTjXXibj76tyFyPRa1soLo0S'}])]}}
    {'tools': {'messages': [ToolMessage(content='5', name='magic_function', id='41c5f227-528d-4483-a313-b03b23b1d327', tool_call_id='call_yTjXXibj76tyFyPRa1soLo0S')]}}
    {'agent': {'messages': [AIMessage(content='The value of `magic_function(3)` is 5.', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 93, 'total_tokens': 107}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'stop', 'logprobs': None}, id='run-0ef12b6e-415d-4758-9b62-5e5e1b350072-0')]}}


## `return_intermediate_steps`

Setting this parameter on AgentExecutor allows users to access intermediate_steps, which pairs agent actions (e.g., tool invocations) with their outcomes.



```python
agent_executor = AgentExecutor(agent=agent, tools=tools, return_intermediate_steps=True)
result = agent_executor.invoke({"input": query})
print(result["intermediate_steps"])
```

    [(ToolAgentAction(tool='magic_function', tool_input={'input': 3}, log="\nInvoking: `magic_function` with `{'input': 3}`\n\n\n", message_log=[AIMessageChunk(content='', additional_kwargs={'tool_calls': [{'index': 0, 'id': 'call_ABI4hftfEdnVgKyfF6OzZbca', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls'}, id='run-837e794f-cfd8-40e0-8abc-4d98ced11b75', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_ABI4hftfEdnVgKyfF6OzZbca'}], tool_call_chunks=[{'name': 'magic_function', 'args': '{"input":3}', 'id': 'call_ABI4hftfEdnVgKyfF6OzZbca', 'index': 0}])], tool_call_id='call_ABI4hftfEdnVgKyfF6OzZbca'), 5)]


By default the [react agent executor](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent) in LangGraph appends all messages to the central state. Therefore, it is easy to see any intermediate steps by just looking at the full state.


```python
from langgraph.prebuilt import create_react_agent

app = create_react_agent(model, tools=tools)

messages = app.invoke({"messages": [("human", query)]})

messages
```




    {'messages': [HumanMessage(content='what is the value of magic_function(3)?', id='0f63e437-c4d8-4da9-b6f5-b293ebfe4a64'),
      AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_S96v28LlI6hNkQrNnIio0JPh', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 64, 'total_tokens': 78}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-ffef7898-14b1-4537-ad90-7c000a8a5d25-0', tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_S96v28LlI6hNkQrNnIio0JPh'}]),
      ToolMessage(content='5', name='magic_function', id='fbd9df4e-1dda-4d3e-9044-b001f7875476', tool_call_id='call_S96v28LlI6hNkQrNnIio0JPh'),
      AIMessage(content='The value of `magic_function(3)` is 5.', response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 87, 'total_tokens': 101}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'stop', 'logprobs': None}, id='run-e5d94c54-d9f4-45cd-be8e-a9101a8d88d6-0')]}



## `max_iterations`

`AgentExecutor` implements a `max_iterations` parameter, whereas this is controlled via `recursion_limit` in LangGraph.

Note that in AgentExecutor, an "iteration" includes a full turn of tool invocation and execution. In LangGraph, each step contributes to the recursion limit, so we will need to multiply by two (and add one) to get equivalent results.

If the recursion limit is reached, LangGraph raises a specific exception type, that we can catch and manage similarly to AgentExecutor.


```python
@tool
def magic_function(input: str) -> str:
    """Applies a magic function to an input."""
    return "Sorry, there was an error. Please try again."


tools = [magic_function]
```


```python
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Respond only in Spanish."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    max_iterations=3,
)

agent_executor.invoke({"input": query})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `magic_function` with `{'input': '3'}`
    
    
    [0m[36;1m[1;3mSorry, there was an error. Please try again.[0m[32;1m[1;3m
    Invoking: `magic_function` with `{'input': '3'}`
    responded: Parece que hubo un error al intentar obtener el valor de `magic_function(3)`. Permíteme intentarlo de nuevo.
    
    [0m[36;1m[1;3mSorry, there was an error. Please try again.[0m[32;1m[1;3mAún no puedo obtener el valor de `magic_function(3)`. ¿Hay algo más en lo que pueda ayudarte?[0m
    
    [1m> Finished chain.[0m





    {'input': 'what is the value of magic_function(3)?',
     'output': 'Aún no puedo obtener el valor de `magic_function(3)`. ¿Hay algo más en lo que pueda ayudarte?'}




```python
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

RECURSION_LIMIT = 2 * 3 + 1

app = create_react_agent(model, tools=tools)

try:
    for chunk in app.stream(
        {"messages": [("human", query)]},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="values",
    ):
        print(chunk["messages"][-1])
except GraphRecursionError:
    print({"input": query, "output": "Agent stopped due to max iterations."})
```

    ('human', 'what is the value of magic_function(3)?')
    content='' additional_kwargs={'tool_calls': [{'id': 'call_pFdKcCu5taDTtOOfX14vEDRp', 'function': {'arguments': '{"input":"3"}', 'name': 'magic_function'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 64, 'total_tokens': 78}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-25836468-ba7e-43be-a7cf-76bba06a2a08-0' tool_calls=[{'name': 'magic_function', 'args': {'input': '3'}, 'id': 'call_pFdKcCu5taDTtOOfX14vEDRp'}]
    content='Sorry, there was an error. Please try again.' name='magic_function' id='1a08b883-9c7b-4969-9e9b-67ce64cdcb5f' tool_call_id='call_pFdKcCu5taDTtOOfX14vEDRp'
    content='It seems there was an error when trying to apply the magic function. Let me try again.' additional_kwargs={'tool_calls': [{'id': 'call_DA0lpDIkBFg2GHy4WsEcZG4K', 'function': {'arguments': '{"input":"3"}', 'name': 'magic_function'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 34, 'prompt_tokens': 97, 'total_tokens': 131}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-d571b774-0ea3-4e35-8b7d-f32932c3f3cc-0' tool_calls=[{'name': 'magic_function', 'args': {'input': '3'}, 'id': 'call_DA0lpDIkBFg2GHy4WsEcZG4K'}]
    content='Sorry, there was an error. Please try again.' name='magic_function' id='0b45787b-c82a-487f-9a5a-de129c30460f' tool_call_id='call_DA0lpDIkBFg2GHy4WsEcZG4K'
    content='It appears that there is a consistent issue when trying to apply the magic function to the input "3." This could be due to various reasons, such as the input not being in the correct format or an internal error.\n\nIf you have any other questions or if there\'s something else you\'d like to try, please let me know!' response_metadata={'token_usage': {'completion_tokens': 66, 'prompt_tokens': 153, 'total_tokens': 219}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'stop', 'logprobs': None} id='run-50a962e6-21b7-4327-8dea-8e2304062627-0'


## `max_execution_time`

`AgentExecutor` implements a `max_execution_time` parameter, allowing users to abort a run that exceeds a total time limit.


```python
import time


@tool
def magic_function(input: str) -> str:
    """Applies a magic function to an input."""
    time.sleep(2.5)
    return "Sorry, there was an error. Please try again."


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_execution_time=2,
    verbose=True,
)

agent_executor.invoke({"input": query})
```

    
    
    [1m> Entering new AgentExecutor chain...[0m
    [32;1m[1;3m
    Invoking: `magic_function` with `{'input': '3'}`
    
    
    [0m[36;1m[1;3mSorry, there was an error. Please try again.[0m[32;1m[1;3m[0m
    
    [1m> Finished chain.[0m





    {'input': 'what is the value of magic_function(3)?',
     'output': 'Agent stopped due to max iterations.'}



With LangGraph's react agent, you can control timeouts on two levels. 

You can set a `step_timeout` to bound each **step**:


```python
from langgraph.prebuilt import create_react_agent

app = create_react_agent(model, tools=tools)
# Set the max timeout for each step here
app.step_timeout = 2

try:
    for chunk in app.stream({"messages": [("human", query)]}):
        print(chunk)
        print("------")
except TimeoutError:
    print({"input": query, "output": "Agent stopped due to max iterations."})
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_HaQkeCwD5QskzJzFixCBacZ4', 'function': {'arguments': '{"input":"3"}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 64, 'total_tokens': 78}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-596c9200-771f-436d-8576-72fcb81620f1-0', tool_calls=[{'name': 'magic_function', 'args': {'input': '3'}, 'id': 'call_HaQkeCwD5QskzJzFixCBacZ4'}])]}}
    ------
    {'input': 'what is the value of magic_function(3)?', 'output': 'Agent stopped due to max iterations.'}


The other way to set a single max timeout for an entire run is to directly use the python stdlib [asyncio](https://docs.python.org/3/library/asyncio.html) library.


```python
import asyncio

from langgraph.prebuilt import create_react_agent

app = create_react_agent(model, tools=tools)


async def stream(app, inputs):
    async for chunk in app.astream({"messages": [("human", query)]}):
        print(chunk)
        print("------")


try:
    task = asyncio.create_task(stream(app, {"messages": [("human", query)]}))
    await asyncio.wait_for(task, timeout=3)
except TimeoutError:
    print("Task Cancelled.")
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_4agJXUHtmHrOOMogjF6ZuzAv', 'function': {'arguments': '{"input":"3"}', 'name': 'magic_function'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 64, 'total_tokens': 78}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-a1c77db7-405f-43d9-8d57-751f2ca1a58c-0', tool_calls=[{'name': 'magic_function', 'args': {'input': '3'}, 'id': 'call_4agJXUHtmHrOOMogjF6ZuzAv'}])]}}
    ------
    Task Cancelled.


## `early_stopping_method`

With LangChain's [AgentExecutor](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.iter), you could configure an [early_stopping_method](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.early_stopping_method) to either return a string saying "Agent stopped due to iteration limit or time limit." (`"force"`) or prompt the LLM a final time to respond (`"generate"`).


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    return "Sorry there was an error, please try again."


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)
agent_executor = AgentExecutor(
    agent=agent, tools=tools, early_stopping_method="force", max_iterations=1
)

result = agent_executor.invoke({"input": query})
print("Output with early_stopping_method='force':")
print(result["output"])
```

    Output with early_stopping_method='force':
    Agent stopped due to max iterations.


#### In LangGraph

In LangGraph, you can explicitly handle the response behavior outside the agent, since the full state can be accessed.


```python
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

RECURSION_LIMIT = 2 * 1 + 1

app = create_react_agent(model, tools=tools)

try:
    for chunk in app.stream(
        {"messages": [("human", query)]},
        {"recursion_limit": RECURSION_LIMIT},
        stream_mode="values",
    ):
        print(chunk["messages"][-1])
except GraphRecursionError:
    print({"input": query, "output": "Agent stopped due to max iterations."})
```

    ('human', 'what is the value of magic_function(3)?')
    content='' additional_kwargs={'tool_calls': [{'id': 'call_bTURmOn9C8zslmn0kMFeykIn', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 64, 'total_tokens': 78}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-0844a504-7e6b-4ea6-a069-7017e38121ee-0' tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_bTURmOn9C8zslmn0kMFeykIn'}]
    content='Sorry there was an error, please try again.' name='magic_function' id='00d5386f-eb23-4628-9a29-d9ce6a7098cc' tool_call_id='call_bTURmOn9C8zslmn0kMFeykIn'
    content='' additional_kwargs={'tool_calls': [{'id': 'call_JYqvvvWmXow2u012DuPoDHFV', 'function': {'arguments': '{"input":3}', 'name': 'magic_function'}, 'type': 'function'}]} response_metadata={'token_usage': {'completion_tokens': 14, 'prompt_tokens': 96, 'total_tokens': 110}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_729ea513f7', 'finish_reason': 'tool_calls', 'logprobs': None} id='run-b73b1b1c-c829-4348-98cd-60b315c85448-0' tool_calls=[{'name': 'magic_function', 'args': {'input': 3}, 'id': 'call_JYqvvvWmXow2u012DuPoDHFV'}]
    {'input': 'what is the value of magic_function(3)?', 'output': 'Agent stopped due to max iterations.'}


## `trim_intermediate_steps`

With LangChain's [AgentExecutor](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor), you could trim the intermediate steps of long-running agents using [trim_intermediate_steps](https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html#langchain.agents.agent.AgentExecutor.trim_intermediate_steps), which is either an integer (indicating the agent should keep the last N steps) or a custom function.

For instance, we could trim the value so the agent only sees the most recent intermediate step.


```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4o")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "{input}"),
        # Placeholders fill up a **list** of messages
        ("placeholder", "{agent_scratchpad}"),
    ]
)


magic_step_num = 1


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    global magic_step_num
    print(f"Call number: {magic_step_num}")
    magic_step_num += 1
    return input + magic_step_num


tools = [magic_function]

agent = create_tool_calling_agent(model, tools, prompt=prompt)


def trim_steps(steps: list):
    # Let's give the agent amnesia
    return []


agent_executor = AgentExecutor(
    agent=agent, tools=tools, trim_intermediate_steps=trim_steps
)


query = "Call the magic function 4 times in sequence with the value 3. You cannot call it multiple times at once."

for step in agent_executor.stream({"input": query}):
    pass
```

    Call number: 1
    Call number: 2
    Call number: 3
    Call number: 4
    Call number: 5
    Call number: 6
    Call number: 7
    Call number: 8
    Call number: 9
    Call number: 10
    Call number: 11
    Call number: 12
    Call number: 13
    Call number: 14


    Stopping agent prematurely due to triggering stop condition


    Call number: 15


#### In LangGraph

We can use the [`messages_modifier`](https://langchain-ai.github.io/langgraph/reference/prebuilt/#create_react_agent) just as before when passing in [prompt templates](#prompt-templates).


```python
from langchain_core.messages import AnyMessage
from langgraph.errors import GraphRecursionError
from langgraph.prebuilt import create_react_agent

magic_step_num = 1


@tool
def magic_function(input: int) -> int:
    """Applies a magic function to an input."""
    global magic_step_num
    print(f"Call number: {magic_step_num}")
    magic_step_num += 1
    return input + magic_step_num


tools = [magic_function]


def _modify_messages(messages: list[AnyMessage]):
    # Give the agent amnesia, only keeping the original user query
    return [("system", "You are a helpful assistant"), messages[0]]


app = create_react_agent(model, tools, messages_modifier=_modify_messages)

try:
    for step in app.stream({"messages": [("human", query)]}, stream_mode="updates"):
        pass
except GraphRecursionError as e:
    print("Stopping agent prematurely due to triggering stop condition")
```

    Call number: 1
    Call number: 2
    Call number: 3
    Call number: 4
    Call number: 5
    Call number: 6
    Call number: 7
    Call number: 8
    Call number: 9
    Call number: 10
    Call number: 11
    Call number: 12
    Stopping agent prematurely due to triggering stop condition

