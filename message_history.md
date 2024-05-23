# How to add message history

:::info Prerequisites

This guide assumes familiarity with the following concepts:
- [LangChain Expression Language (LCEL)](/docs/concepts/#langchain-expression-language)
- [Chaining runnables](/docs/how_to/sequence/)
- [Configuring chain parameters at runtime](/docs/how_to/configure)
- [Prompt templates](/docs/concepts/#prompt-templates)
- [Chat Messages](/docs/concepts/#message-types)

:::

Passing conversation state into and out a chain is vital when building a chatbot. The [`RunnableWithMessageHistory`](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html#langchain_core.runnables.history.RunnableWithMessageHistory) class lets us add message history to certain types of chains. It wraps another Runnable and manages the chat message history for it.

Specifically, it can be used for any Runnable that takes as input one of:

* a sequence of [`BaseMessages`](/docs/concepts/#message-types)
* a dict with a key that takes a sequence of `BaseMessages`
* a dict with a key that takes the latest message(s) as a string or sequence of `BaseMessages`, and a separate key that takes historical messages

And returns as output one of

* a string that can be treated as the contents of an `AIMessage`
* a sequence of `BaseMessage`
* a dict with a key that contains a sequence of `BaseMessage`

Let's take a look at some examples to see how it works. First we construct a runnable (which here accepts a dict as input and returns a message as output):

```{=mdx}
import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs
  customVarName="llm"
/>
```


```python
# | output: false
# | echo: false

%pip install -qU langchain langchain_anthropic

import os
from getpass import getpass

from langchain_anthropic import ChatAnthropic

os.environ["ANTHROPIC_API_KEY"] = getpass()

model = ChatAnthropic(model="claude-3-haiku-20240307", temperature=0)
```


```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai.chat_models import ChatOpenAI

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You're an assistant who's good at {ability}. Respond in 20 words or fewer",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)
runnable = prompt | model
```

To manage the message history, we will need:
1. This runnable;
2. A callable that returns an instance of `BaseChatMessageHistory`.

Check out the [memory integrations](https://integrations.langchain.com/memory) page for implementations of chat message histories using Redis and other providers. Here we demonstrate using an in-memory `ChatMessageHistory` as well as more persistent storage using `RedisChatMessageHistory`.

## In-memory

Below we show a simple example in which the chat history lives in memory, in this case via a global Python dict.

We construct a callable `get_session_history` that references this dict to return an instance of `ChatMessageHistory`. The arguments to the callable can be specified by passing a configuration to the `RunnableWithMessageHistory` at runtime. By default, the configuration parameter is expected to be a single string `session_id`. This can be adjusted via the `history_factory_config` kwarg.

Using the single-parameter default:


```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

Note that we've specified `input_messages_key` (the key to be treated as the latest input message) and `history_messages_key` (the key to add historical messages to).

When invoking this new runnable, we specify the corresponding chat history via a configuration parameter:


```python
with_message_history.invoke(
    {"ability": "math", "input": "What does cosine mean?"},
    config={"configurable": {"session_id": "abc123"}},
)
```




    AIMessage(content='Cosine is a trigonometric function that represents the ratio of the adjacent side to the hypotenuse of a right triangle.', response_metadata={'id': 'msg_017rAM9qrBTSdJ5i1rwhB7bT', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 32, 'output_tokens': 31}}, id='run-65e94a5e-a804-40de-ba88-d01b6cd06864-0')




```python
# Remembers
with_message_history.invoke(
    {"ability": "math", "input": "What?"},
    config={"configurable": {"session_id": "abc123"}},
)
```




    AIMessage(content='Cosine is a trigonometric function that represents the ratio of the adjacent side to the hypotenuse of a right triangle.', response_metadata={'id': 'msg_017hK1Q63ganeQZ9wdeqruLP', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 68, 'output_tokens': 31}}, id='run-a42177ef-b04a-4968-8606-446fb465b943-0')




```python
# New session_id --> does not remember.
with_message_history.invoke(
    {"ability": "math", "input": "What?"},
    config={"configurable": {"session_id": "def234"}},
)
```




    AIMessage(content="I'm an AI assistant skilled in mathematics. How can I help you with a math-related task?", response_metadata={'id': 'msg_01AYwfQ6SH5qz8ZQMW3nYtGU', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 28, 'output_tokens': 24}}, id='run-c57d93e3-305f-4c0e-bdb9-ef82f5b49f61-0')



The configuration parameters by which we track message histories can be customized by passing in a list of ``ConfigurableFieldSpec`` objects to the ``history_factory_config`` parameter. Below, we use two parameters: a `user_id` and `conversation_id`.


```python
from langchain_core.runnables import ConfigurableFieldSpec

store = {}


def get_session_history(user_id: str, conversation_id: str) -> BaseChatMessageHistory:
    if (user_id, conversation_id) not in store:
        store[(user_id, conversation_id)] = ChatMessageHistory()
    return store[(user_id, conversation_id)]


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

with_message_history.invoke(
    {"ability": "math", "input": "Hello"},
    config={"configurable": {"user_id": "123", "conversation_id": "1"}},
)
```




    AIMessage(content='Hello! How can I assist you with math today?', response_metadata={'id': 'msg_01UdhnwghuSE7oRM57STFhHL', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 27, 'output_tokens': 14}}, id='run-3d53f67a-4ea7-4d78-8e67-37db43d4af5d-0')



### Examples with runnables of different signatures

The above runnable takes a dict as input and returns a BaseMessage. Below we show some alternatives.

#### Messages input, dict output


```python
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableParallel

chain = RunnableParallel({"output_message": model})


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


with_message_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    output_messages_key="output_message",
)

with_message_history.invoke(
    [HumanMessage(content="What did Simone de Beauvoir believe about free will")],
    config={"configurable": {"session_id": "baz"}},
)
```




    {'output_message': AIMessage(content='Simone de Beauvoir was a prominent French existentialist philosopher who had some key beliefs about free will:\n\n1. Radical Freedom: De Beauvoir believed that humans have radical freedom - the ability to choose and define themselves through their actions. She rejected determinism and believed that we are not simply products of our biology, upbringing, or social circumstances.\n\n2. Ambiguity of the Human Condition: However, de Beauvoir also recognized the ambiguity of the human condition. While we have radical freedom, we are also situated beings constrained by our facticity (our given circumstances and limitations). This creates a tension and anguish in the human experience.\n\n3. Responsibility and Bad Faith: With this radical freedom comes great responsibility. De Beauvoir criticized "bad faith" - the tendency of people to deny their freedom and responsibility by making excuses or hiding behind social roles and norms.\n\n4. Ethical Engagement: For de Beauvoir, true freedom and authenticity required ethical engagement with the world and with others. We must take responsibility for our choices and their impact on others.\n\nOverall, de Beauvoir saw free will as a core aspect of the human condition, but one that is fraught with difficulty and ambiguity. Her philosophy emphasized the importance of owning our freedom and using it to ethically shape our lives and world.', response_metadata={'id': 'msg_01A78LdxxsCm6uR8vcAdMQBt', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 20, 'output_tokens': 293}}, id='run-9447a229-5d17-4b20-a48b-7507b78b225a-0')}




```python
with_message_history.invoke(
    [HumanMessage(content="How did this compare to Sartre")],
    config={"configurable": {"session_id": "baz"}},
)
```




    {'output_message': AIMessage(content="Simone de Beauvoir's views on free will were quite similar, but not identical, to those of her long-time partner Jean-Paul Sartre, another prominent existentialist philosopher.\n\nKey similarities:\n\n1. Radical Freedom: Both de Beauvoir and Sartre believed that humans have radical, unconditioned freedom to choose and define themselves.\n\n2. Rejection of Determinism: They both rejected deterministic views that see humans as products of their circumstances or biology.\n\n3. Emphasis on Responsibility: They agreed that with radical freedom comes great responsibility for one's choices and their consequences.\n\nKey differences:\n\n1. Ambiguity of the Human Condition: While Sartre emphasized the pure, unconditioned nature of human freedom, de Beauvoir recognized the ambiguity of the human condition - our freedom is constrained by our facticity (circumstances).\n\n2. Ethical Engagement: De Beauvoir placed more emphasis on the importance of ethical engagement with the world and others, whereas Sartre's focus was more on the individual's freedom.\n\n3. Gendered Perspectives: As a woman, de Beauvoir's perspective was more attuned to issues of gender and the lived experience of women, which shaped her views on freedom and ethics.\n\nSo in summary, while Sartre and de Beauvoir shared a core existentialist philosophy centered on radical human freedom, de Beauvoir's thought incorporated a greater recognition of the ambiguity and ethical dimensions of the human condition. This reflected her distinct feminist and phenomenological approach.", response_metadata={'id': 'msg_01U6X3KNPufVg3zFvnx24eKq', 'model': 'claude-3-haiku-20240307', 'stop_reason': 'end_turn', 'stop_sequence': None, 'usage': {'input_tokens': 324, 'output_tokens': 338}}, id='run-c4a984bd-33c6-4e26-a4d1-d58b666d065c-0')}



#### Messages input, messages output


```python
RunnableWithMessageHistory(
    model,
    get_session_history,
)
```




    RunnableWithMessageHistory(bound=RunnableBinding(bound=RunnableBinding(bound=RunnableLambda(_enter_history), config={'run_name': 'load_history'})
    | RunnableBinding(bound=ChatAnthropic(model='claude-3-haiku-20240307', temperature=0.0, anthropic_api_url='https://api.anthropic.com', anthropic_api_key=SecretStr('**********'), _client=<anthropic.Anthropic object at 0x1077ff5b0>, _async_client=<anthropic.AsyncAnthropic object at 0x1321c71f0>), config_factories=[<function Runnable.with_listeners.<locals>.<lambda> at 0x1473dd000>]), config={'run_name': 'RunnableWithMessageHistory'}), get_session_history=<function get_session_history at 0x1374c7be0>, history_factory_config=[ConfigurableFieldSpec(id='session_id', annotation=<class 'str'>, name='Session ID', description='Unique identifier for a session.', default='', is_shared=True, dependencies=None)])



#### Dict with single key for all messages input, messages output


```python
from operator import itemgetter

RunnableWithMessageHistory(
    itemgetter("input_messages") | model,
    get_session_history,
    input_messages_key="input_messages",
)
```




    RunnableWithMessageHistory(bound=RunnableBinding(bound=RunnableBinding(bound=RunnableAssign(mapper={
      input_messages: RunnableBinding(bound=RunnableLambda(_enter_history), config={'run_name': 'load_history'})
    }), config={'run_name': 'insert_history'})
    | RunnableBinding(bound=RunnableLambda(itemgetter('input_messages'))
      | ChatAnthropic(model='claude-3-haiku-20240307', temperature=0.0, anthropic_api_url='https://api.anthropic.com', anthropic_api_key=SecretStr('**********'), _client=<anthropic.Anthropic object at 0x1077ff5b0>, _async_client=<anthropic.AsyncAnthropic object at 0x1321c71f0>), config_factories=[<function Runnable.with_listeners.<locals>.<lambda> at 0x1473df6d0>]), config={'run_name': 'RunnableWithMessageHistory'}), get_session_history=<function get_session_history at 0x1374c7be0>, input_messages_key='input_messages', history_factory_config=[ConfigurableFieldSpec(id='session_id', annotation=<class 'str'>, name='Session ID', description='Unique identifier for a session.', default='', is_shared=True, dependencies=None)])



## Persistent storage

In many cases it is preferable to persist conversation histories. `RunnableWithMessageHistory` is agnostic as to how the `get_session_history` callable retrieves its chat message histories. See [here](https://github.com/langchain-ai/langserve/blob/main/examples/chat_with_persistence_and_user/server.py) for an example using a local filesystem. Below we demonstrate how one could use Redis. Check out the [memory integrations](https://integrations.langchain.com/memory) page for implementations of chat message histories using other providers.

### Setup

We'll need to install Redis if it's not installed already:


```python
%pip install --upgrade --quiet redis
```

Start a local Redis Stack server if we don't have an existing Redis deployment to connect to:
```bash
docker run -d -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
```


```python
REDIS_URL = "redis://localhost:6379/0"
```

### [LangSmith](https://docs.smith.langchain.com)

LangSmith is especially useful for something like message history injection, where it can be hard to otherwise understand what the inputs are to various parts of the chain.

Note that LangSmith is not needed, but it is helpful.
If you do want to use LangSmith, after you sign up at the link above, make sure to uncoment the below and set your environment variables to start logging traces:


```python
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

Updating the message history implementation just requires us to define a new callable, this time returning an instance of `RedisChatMessageHistory`:


```python
from langchain_community.chat_message_histories import RedisChatMessageHistory


def get_message_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id, url=REDIS_URL)


with_message_history = RunnableWithMessageHistory(
    runnable,
    get_message_history,
    input_messages_key="input",
    history_messages_key="history",
)
```

We can invoke as before:


```python
with_message_history.invoke(
    {"ability": "math", "input": "What does cosine mean?"},
    config={"configurable": {"session_id": "foobar"}},
)
```




    AIMessage(content='Cosine is a trigonometric function that represents the ratio of the adjacent side to the hypotenuse in a right triangle.')




```python
with_message_history.invoke(
    {"ability": "math", "input": "What's its inverse"},
    config={"configurable": {"session_id": "foobar"}},
)
```




    AIMessage(content='The inverse of cosine is the arccosine function, denoted as acos or cos^-1, which gives the angle corresponding to a given cosine value.')



:::{.callout-tip}

[Langsmith trace](https://smith.langchain.com/public/bd73e122-6ec1-48b2-82df-e6483dc9cb63/r)

:::

Looking at the Langsmith trace for the second call, we can see that when constructing the prompt, a "history" variable has been injected which is a list of two messages (our first input and first output).

## Next steps

You have now learned one way to manage message history for a runnable.

To learn more, see the other how-to guides on runnables in this section.
