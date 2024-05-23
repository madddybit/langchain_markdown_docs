# How to add chat history

In many Q&A applications we want to allow the user to have a back-and-forth conversation, meaning the application needs some sort of "memory" of past questions and answers, and some logic for incorporating those into its current thinking.

In this guide we focus on **adding logic for incorporating historical messages.**

This is largely a condensed version of the [Conversational RAG tutorial](/docs/tutorials/qa_chat_history).

We will cover two approaches:
1. [Chains](/docs/how_to/qa_chat_history_how_to#chains), in which we always execute a retrieval step;
2. [Agents](/docs/how_to/qa_chat_history_how_to#agents), in which we give an LLM discretion over whether and how to execute a retrieval step (or multiple steps).

For the external knowledge source, we will use the same [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng from the [RAG tutorial](/docs/tutorials/rag).

## Setup

### Dependencies

We'll use OpenAI embeddings and a Chroma vector store in this walkthrough, but everything shown here works with any [Embeddings](/docs/concepts#embedding-models), and [VectorStore](/docs/concepts#vectorstores) or [Retriever](/docs/concepts#retrievers). 

We'll use the following packages:


```python
%pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-chroma bs4
```

We need to set environment variable `OPENAI_API_KEY`, which can be done directly or loaded from a `.env` file like so:


```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# import dotenv

# dotenv.load_dotenv()
```

### LangSmith

Many of the applications you build with LangChain will contain multiple steps with multiple invocations of LLM calls. As these applications get more and more complex, it becomes crucial to be able to inspect what exactly is going on inside your chain or agent. The best way to do this is with [LangSmith](https://smith.langchain.com).

Note that LangSmith is not needed, but it is helpful. If you do want to use LangSmith, after you sign up at the link above, make sure to set your environment variables to start logging traces:


```python
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

## Chains {#chains}

In a conversational RAG application, queries issued to the retriever should be informed by the context of the conversation. LangChain provides a [create_history_aware_retriever](https://api.python.langchain.com/en/latest/chains/langchain.chains.history_aware_retriever.create_history_aware_retriever.html) constructor to simplify this. It constructs a chain that accepts keys `input` and `chat_history` as input, and has the same output schema as a retriever. `create_history_aware_retriever` requires as inputs:  

1. LLM;
2. Retriever;
3. Prompt.

First we obtain these objects:

### LLM

We can use any supported chat model:

```{=mdx}
import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs customVarName="llm" />
```


```python
# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
```

### Retriever

For the retriever, we will use [WebBaseLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.web_base.WebBaseLoader.html) to load the content of a web page. Here we instantiate a `Chroma` vectorstore and then use its [.as_retriever](https://api.python.langchain.com/en/latest/vectorstores/langchain_core.vectorstores.VectorStore.html#langchain_core.vectorstores.VectorStore.as_retriever) method to build a retriever that can be incorporated into [LCEL](/docs/concepts/#langchain-expression-language) chains.


```python
import bs4
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()
```

### Prompt

We'll use a prompt that includes a `MessagesPlaceholder` variable under the name "chat_history". This allows us to pass in a list of Messages to the prompt using the "chat_history" input key, and these messages will be inserted after the system message and before the human message containing the latest question.


```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
```

### Assembling the chain

We can then instantiate the history-aware retriever:


```python
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)
```

This chain prepends a rephrasing of the input query to our retriever, so that the retrieval incorporates the context of the conversation.

Now we can build our full QA chain.

As in the [RAG tutorial](/docs/tutorials/rag), we will use [create_stuff_documents_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.combine_documents.stuff.create_stuff_documents_chain.html) to generate a `question_answer_chain`, with input keys `context`, `chat_history`, and `input`-- it accepts the retrieved context alongside the conversation history and query to generate an answer.

We build our final `rag_chain` with [create_retrieval_chain](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval.create_retrieval_chain.html). This chain applies the `history_aware_retriever` and `question_answer_chain` in sequence, retaining intermediate outputs such as the retrieved context for convenience. It has input keys `input` and `chat_history`, and includes `input`, `chat_history`, `context`, and `answer` in its output.


```python
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
```

### Adding chat history

To manage the chat history, we will need:

1. An object for storing the chat history;
2. An object that wraps our chain and manages updates to the chat history.

For these we will use [BaseChatMessageHistory](https://api.python.langchain.com/en/latest/chat_history/langchain_core.chat_history.BaseChatMessageHistory.html) and [RunnableWithMessageHistory](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.history.RunnableWithMessageHistory.html). The latter is a wrapper for an LCEL chain and a `BaseChatMessageHistory` that handles injecting chat history into inputs and updating it after each invocation.

For a detailed walkthrough of how to use these classes together to create a stateful conversational chain, head to the [How to add message history (memory)](/docs/how_to/message_history/) LCEL how-to guide.

Below, we implement a simple example of the second option, in which chat histories are stored in a simple dict. LangChain manages memory integrations with [Redis](/docs/integrations/memory/redis_chat_message_history/) and other technologies to provide for more robust persistence.

Instances of `RunnableWithMessageHistory` manage the chat history for you. They accept a config with a key (`"session_id"` by default) that specifies what conversation history to fetch and prepend to the input, and append the output to the same conversation history. Below is an example:


```python
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```


```python
conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]
```




    'Task decomposition involves breaking down a complex task into smaller and simpler steps to make it more manageable and easier to accomplish. This process can be done using techniques like Chain of Thought (CoT) or Tree of Thoughts to guide the model in thinking step by step or exploring multiple reasoning possibilities at each step. Task decomposition can be facilitated by providing simple prompts to a language model, task-specific instructions, or human inputs.'




```python
conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]
```




    'Task decomposition can be achieved through various methods, including using techniques like Chain of Thought (CoT) or Tree of Thoughts to guide the model in breaking down complex tasks into smaller steps. Common ways of task decomposition include providing simple prompts to a language model, task-specific instructions tailored to the specific task at hand, or incorporating human inputs to guide the decomposition process effectively.'



The conversation history can be inspected in the `store` dict:


```python
from langchain_core.messages import AIMessage

for message in store["abc123"].messages:
    if isinstance(message, AIMessage):
        prefix = "AI"
    else:
        prefix = "User"

    print(f"{prefix}: {message.content}\n")
```

    User: What is Task Decomposition?
    
    AI: Task decomposition involves breaking down a complex task into smaller and simpler steps to make it more manageable and easier to accomplish. This process can be done using techniques like Chain of Thought (CoT) or Tree of Thoughts to guide the model in thinking step by step or exploring multiple reasoning possibilities at each step. Task decomposition can be facilitated by providing simple prompts to a language model, task-specific instructions, or human inputs.
    
    User: What are common ways of doing it?
    
    AI: Task decomposition can be achieved through various methods, including using techniques like Chain of Thought (CoT) or Tree of Thoughts to guide the model in breaking down complex tasks into smaller steps. Common ways of task decomposition include providing simple prompts to a language model, task-specific instructions tailored to the specific task at hand, or incorporating human inputs to guide the decomposition process effectively.
    


### Tying it together

![](../../static/img/conversational_retrieval_chain.png)

For convenience, we tie together all of the necessary steps in a single code cell:


```python
import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


### Contextualize question ###
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)


### Answer question ###
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)
```


```python
conversational_rag_chain.invoke(
    {"input": "What is Task Decomposition?"},
    config={
        "configurable": {"session_id": "abc123"}
    },  # constructs a key "abc123" in `store`.
)["answer"]
```




    'Task decomposition involves breaking down a complex task into smaller and simpler steps to make it more manageable. This process helps agents or models tackle difficult tasks by dividing them into more easily achievable subgoals. Task decomposition can be done through techniques like Chain of Thought or Tree of Thoughts, which guide the model in thinking step by step or exploring multiple reasoning possibilities at each step.'




```python
conversational_rag_chain.invoke(
    {"input": "What are common ways of doing it?"},
    config={"configurable": {"session_id": "abc123"}},
)["answer"]
```




    "Common ways of task decomposition include using techniques like Chain of Thought (CoT) or Tree of Thoughts to guide models in breaking down complex tasks into smaller steps. This can be achieved through simple prompting with LLMs, task-specific instructions, or human inputs to help the model understand and navigate the task effectively. Task decomposition aims to enhance model performance on complex tasks by utilizing more test-time computation and shedding light on the model's thinking process."



## Agents {#agents}

Agents leverage the reasoning capabilities of LLMs to make decisions during execution. Using agents allow you to offload some discretion over the retrieval process. Although their behavior is less predictable than chains, they offer some advantages in this context:
- Agents generate the input to the retriever directly, without necessarily needing us to explicitly build in contextualization, as we did above;
- Agents can execute multiple retrieval steps in service of a query, or refrain from executing a retrieval step altogether (e.g., in response to a generic greeting from a user).

### Retrieval tool

Agents can access "tools" and manage their execution. In this case, we will convert our retriever into a LangChain tool to be wielded by the agent:


```python
from langchain.tools.retriever import create_retriever_tool

tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]
```

### Agent constructor

Now that we have defined the tools and the LLM, we can create the agent. We will be using [LangGraph](/docs/concepts/#langgraph) to construct the agent. 
Currently we are using a high level interface to construct the agent, but the nice thing about LangGraph is that this high-level interface is backed by a low-level, highly controllable API in case you want to modify the agent logic.


```python
from langgraph.prebuilt import chat_agent_executor

agent_executor = chat_agent_executor.create_tool_calling_executor(llm, tools)
```

We can now try it out. Note that so far it is not stateful (we still need to add in memory)


```python
from langchain_core.messages import HumanMessage

query = "What is Task Decomposition?"

for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]},
):
    print(s)
    print("----")
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_wxRrUmNbaNny8wh9JIb5uCRB', 'function': {'arguments': '{"query":"Task Decomposition"}', 'name': 'blog_post_retriever'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 68, 'total_tokens': 87}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3b956da36b', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-57ee0d12-6142-4957-a002-cce0093efe07-0', tool_calls=[{'name': 'blog_post_retriever', 'args': {'query': 'Task Decomposition'}, 'id': 'call_wxRrUmNbaNny8wh9JIb5uCRB'}])]}}
    ----
    {'action': {'messages': [ToolMessage(content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.\n\nTree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.\n\n(3) Task execution: Expert models execute on the specific tasks and log results.\nInstruction:\n\nWith the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user\'s request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.\n\nFig. 11. Illustration of how HuggingGPT works. (Image source: Shen et al. 2023)\nThe system comprises of 4 stages:\n(1) Task planning: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, dependencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.\nInstruction:', name='blog_post_retriever', id='9c3a17f7-653c-47fa-b4e4-fa3d8d24c85d', tool_call_id='call_wxRrUmNbaNny8wh9JIb5uCRB')]}}
    ----
    {'agent': {'messages': [AIMessage(content='Task decomposition is a technique used to break down complex tasks into smaller and simpler steps. This approach helps agents in planning and executing tasks more effectively. One common method for task decomposition is the Chain of Thought (CoT) technique, where models are instructed to think step by step to decompose hard tasks into manageable steps. Another extension of CoT is the Tree of Thoughts, which explores multiple reasoning possibilities at each step by creating a tree structure of thought steps.\n\nTask decomposition can be achieved through various methods, such as using language models with simple prompting, task-specific instructions, or human inputs. By breaking down tasks into smaller components, agents can better plan and execute tasks efficiently.\n\nIf you would like more detailed information or examples on task decomposition, feel free to ask!', response_metadata={'token_usage': {'completion_tokens': 154, 'prompt_tokens': 588, 'total_tokens': 742}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3b956da36b', 'finish_reason': 'stop', 'logprobs': None}, id='run-8991fa20-c527-4f9e-a058-fc6264fe6259-0')]}}
    ----


LangGraph comes with built in persistence, so we don't need to use ChatMessageHistory! Rather, we can pass in a checkpointer to our LangGraph agent directly.

Distinct conversations are managed by specifying a key for a conversation thread in the config dict, as shown below.


```python
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

agent_executor = chat_agent_executor.create_tool_calling_executor(
    llm, tools, checkpointer=memory
)
```

This is all we need to construct a conversational RAG agent.

Let's observe its behavior. Note that if we input a query that does not require a retrieval step, the agent does not execute one:


```python
config = {"configurable": {"thread_id": "abc123"}}

for s in agent_executor.stream(
    {"messages": [HumanMessage(content="Hi! I'm bob")]}, config=config
):
    print(s)
    print("----")
```

    {'agent': {'messages': [AIMessage(content='Hello Bob! How can I assist you today?', response_metadata={'token_usage': {'completion_tokens': 11, 'prompt_tokens': 67, 'total_tokens': 78}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3b956da36b', 'finish_reason': 'stop', 'logprobs': None}, id='run-1451e59b-b135-4776-985d-4759338ffee5-0')]}}
    ----


Further, if we input a query that does require a retrieval step, the agent generates the input to the tool:


```python
query = "What is Task Decomposition?"

for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]}, config=config
):
    print(s)
    print("----")
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_ab2x4iUPSWDAHS5txL7PspSK', 'function': {'arguments': '{"query":"Task Decomposition"}', 'name': 'blog_post_retriever'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 19, 'prompt_tokens': 91, 'total_tokens': 110}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3b956da36b', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-f76b5813-b41c-4d0d-9ed2-667b988d885e-0', tool_calls=[{'name': 'blog_post_retriever', 'args': {'query': 'Task Decomposition'}, 'id': 'call_ab2x4iUPSWDAHS5txL7PspSK'}])]}}
    ----
    {'action': {'messages': [ToolMessage(content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.\n\nTree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.\n\n(3) Task execution: Expert models execute on the specific tasks and log results.\nInstruction:\n\nWith the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user\'s request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.\n\nFig. 11. Illustration of how HuggingGPT works. (Image source: Shen et al. 2023)\nThe system comprises of 4 stages:\n(1) Task planning: LLM works as the brain and parses the user requests into multiple tasks. There are four attributes associated with each task: task type, ID, dependencies, and arguments. They use few-shot examples to guide LLM to do task parsing and planning.\nInstruction:', name='blog_post_retriever', id='e0895fa5-5d41-4be0-98db-10a83d42fc2f', tool_call_id='call_ab2x4iUPSWDAHS5txL7PspSK')]}}
    ----
    {'agent': {'messages': [AIMessage(content='Task decomposition is a technique used in complex tasks where the task is broken down into smaller and simpler steps. This approach helps in managing and solving difficult tasks by dividing them into more manageable components. One common method for task decomposition is the Chain of Thought (CoT) technique, which prompts the model to think step by step and decompose hard tasks into smaller steps. Another extension of CoT is the Tree of Thoughts, which explores multiple reasoning possibilities at each step by creating a tree structure of thought steps.\n\nTask decomposition can be achieved through various methods, such as using language models with simple prompting, task-specific instructions, or human inputs. By breaking down tasks into smaller components, agents can better plan and execute complex tasks effectively.\n\nIf you would like more detailed information or examples related to task decomposition, feel free to ask!', response_metadata={'token_usage': {'completion_tokens': 165, 'prompt_tokens': 611, 'total_tokens': 776}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3b956da36b', 'finish_reason': 'stop', 'logprobs': None}, id='run-13296566-8577-4d65-982b-a39718988ca3-0')]}}
    ----


Above, instead of inserting our query verbatim into the tool, the agent stripped unnecessary words like "what" and "is".

This same principle allows the agent to use the context of the conversation when necessary:


```python
query = "What according to the blog post are common ways of doing it? redo the search"

for s in agent_executor.stream(
    {"messages": [HumanMessage(content=query)]}, config=config
):
    print(s)
    print("----")
```

    {'agent': {'messages': [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_KvoiamnLfGEzMeEMlV3u0TJ7', 'function': {'arguments': '{"query":"common ways of task decomposition"}', 'name': 'blog_post_retriever'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 21, 'prompt_tokens': 930, 'total_tokens': 951}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3b956da36b', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-dd842071-6dbd-4b68-8657-892eaca58638-0', tool_calls=[{'name': 'blog_post_retriever', 'args': {'query': 'common ways of task decomposition'}, 'id': 'call_KvoiamnLfGEzMeEMlV3u0TJ7'}])]}}
    ----
    {'action': {'messages': [ToolMessage(content='Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.\n\nFig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.\n\nResources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. Reflect on past decisions and strategies to refine your approach.\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.\n\n(3) Task execution: Expert models execute on the specific tasks and log results.\nInstruction:\n\nWith the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user\'s request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.', name='blog_post_retriever', id='c749bb8e-c8e0-4fa3-bc11-3e2e0651880b', tool_call_id='call_KvoiamnLfGEzMeEMlV3u0TJ7')]}}
    ----
    {'agent': {'messages': [AIMessage(content='According to the blog post, common ways of task decomposition include:\n\n1. Using language models with simple prompting like "Steps for XYZ" or "What are the subgoals for achieving XYZ?"\n2. Utilizing task-specific instructions, for example, using "Write a story outline" for writing a novel.\n3. Involving human inputs in the task decomposition process.\n\nThese methods help in breaking down complex tasks into smaller and more manageable steps, facilitating better planning and execution of the overall task.', response_metadata={'token_usage': {'completion_tokens': 100, 'prompt_tokens': 1475, 'total_tokens': 1575}, 'model_name': 'gpt-3.5-turbo', 'system_fingerprint': 'fp_3b956da36b', 'finish_reason': 'stop', 'logprobs': None}, id='run-98b765b3-f1a6-4c9a-ad0f-2db7950b900f-0')]}}
    ----


Note that the agent was able to infer that "it" in our query refers to "task decomposition", and generated a reasonable search query as a result-- in this case, "common ways of task decomposition".

### Tying it together

For convenience, we tie together all of the necessary steps in a single code cell:


```python
import bs4
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


### Construct retriever ###
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
retriever = vectorstore.as_retriever()


### Build retriever tool ###
tool = create_retriever_tool(
    retriever,
    "blog_post_retriever",
    "Searches and returns excerpts from the Autonomous Agents blog post.",
)
tools = [tool]


agent_executor = chat_agent_executor.create_tool_calling_executor(
    llm, tools, checkpointer=memory
)
```

## Next steps

We've covered the steps to build a basic conversational Q&A application:

- We used chains to build a predictable application that generates search queries for each user input;
- We used agents to build an application that "decides" when and how to generate search queries.

To explore different types of retrievers and retrieval strategies, visit the [retrievers](/docs/how_to#retrievers) section of the how-to guides.

For a detailed walkthrough of LangChain's conversation memory abstractions, visit the [How to add message history (memory)](/docs/how_to/message_history) LCEL page.

To learn more about agents, head to the [Agents Modules](/docs/tutorials/agents).
