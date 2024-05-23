# How to stream results from your RAG application

This guide explains how to stream results from a RAG application. It covers streaming tokens from the final output as well as intermediate steps of a chain (e.g., from query re-writing).

We'll work off of the Q&A app with sources we built over the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng in the [RAG tutorial](/docs/tutorials/rag).

## Setup

### Dependencies

We'll use OpenAI embeddings and a Chroma vector store in this walkthrough, but everything shown here works with any [Embeddings](/docs/concepts#embedding-models), [VectorStore](/docs/concepts#vectorstores) or [Retriever](/docs/concepts#retrievers). 

We'll use the following packages:


```python
%pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-openai langchain-chroma bs4
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

## RAG chain

Let's first select a LLM:

```{=mdx}
import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs customVarName="llm" />
```


```python
# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
```

Here is Q&A app with sources we built over the [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/) blog post by Lilian Weng in the [RAG tutorial](/docs/tutorials/rag):


```python
import bs4
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 1. Load, chunk and index the contents of the blog to create a retriever.
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


# 2. Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
```

## Streaming final outputs

The chain constructed by `create_retrieval_chain` returns a dict with keys `"input"`, `"context"`, and `"answer"`. The `.stream` method will by default stream each key in a sequence.

Note that here only the `"answer"` key is streamed token-by-token, as the other components-- such as retrieval-- do not support token-level streaming.


```python
for chunk in rag_chain.stream({"input": "What is Task Decomposition?"}):
    print(chunk)
```

    {'input': 'What is Task Decomposition?'}
    {'context': [Document(page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}), Document(page_content='Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}), Document(page_content='Resources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. Reflect on past decisions and strategies to refine your approach.\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}), Document(page_content="(3) Task execution: Expert models execute on the specific tasks and log results.\nInstruction:\n\nWith the input and the inference results, the AI assistant needs to describe the process and results. The previous stages can be formed as - User Input: {{ User Input }}, Task Planning: {{ Tasks }}, Model Selection: {{ Model Assignment }}, Task Execution: {{ Predictions }}. You must first answer the user's request in a straightforward manner. Then describe the task process and show your analysis and model inference results to the user in the first person. If inference results contain a file path, must tell the user the complete file path.", metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'})]}
    {'answer': ''}
    {'answer': 'Task'}
    {'answer': ' decomposition'}
    {'answer': ' involves'}
    {'answer': ' breaking'}
    {'answer': ' down'}
    {'answer': ' complex'}
    {'answer': ' tasks'}
    {'answer': ' into'}
    {'answer': ' smaller'}
    {'answer': ' and'}
    {'answer': ' simpler'}
    {'answer': ' steps'}
    {'answer': ' to'}
    {'answer': ' make'}
    {'answer': ' them'}
    {'answer': ' more'}
    {'answer': ' manageable'}
    {'answer': '.'}
    {'answer': ' This'}
    {'answer': ' process'}
    {'answer': ' can'}
    {'answer': ' be'}
    {'answer': ' facilitated'}
    {'answer': ' by'}
    {'answer': ' techniques'}
    {'answer': ' like'}
    {'answer': ' Chain'}
    {'answer': ' of'}
    {'answer': ' Thought'}
    {'answer': ' ('}
    {'answer': 'Co'}
    {'answer': 'T'}
    {'answer': ')'}
    {'answer': ' and'}
    {'answer': ' Tree'}
    {'answer': ' of'}
    {'answer': ' Thoughts'}
    {'answer': ','}
    {'answer': ' which'}
    {'answer': ' help'}
    {'answer': ' agents'}
    {'answer': ' plan'}
    {'answer': ' and'}
    {'answer': ' execute'}
    {'answer': ' tasks'}
    {'answer': ' effectively'}
    {'answer': ' by'}
    {'answer': ' dividing'}
    {'answer': ' them'}
    {'answer': ' into'}
    {'answer': ' sub'}
    {'answer': 'goals'}
    {'answer': ' or'}
    {'answer': ' multiple'}
    {'answer': ' reasoning'}
    {'answer': ' possibilities'}
    {'answer': '.'}
    {'answer': ' Task'}
    {'answer': ' decomposition'}
    {'answer': ' can'}
    {'answer': ' be'}
    {'answer': ' initiated'}
    {'answer': ' through'}
    {'answer': ' simple'}
    {'answer': ' prompts'}
    {'answer': ','}
    {'answer': ' task'}
    {'answer': '-specific'}
    {'answer': ' instructions'}
    {'answer': ','}
    {'answer': ' or'}
    {'answer': ' human'}
    {'answer': ' inputs'}
    {'answer': ' to'}
    {'answer': ' guide'}
    {'answer': ' the'}
    {'answer': ' agent'}
    {'answer': ' in'}
    {'answer': ' achieving'}
    {'answer': ' its'}
    {'answer': ' goals'}
    {'answer': ' efficiently'}
    {'answer': '.'}
    {'answer': ''}


We are free to process chunks as they are streamed out. If we just want to stream the answer tokens, for example, we can select chunks with the corresponding key:


```python
for chunk in rag_chain.stream({"input": "What is Task Decomposition?"}):
    if answer_chunk := chunk.get("answer"):
        print(f"{answer_chunk}|", end="")
```

    Task| decomposition| is| a| technique| used| to| break| down| complex| tasks| into| smaller| and| more| manageable| steps|.| This| process| helps| agents| or| models| handle| intricate| tasks| by| dividing| them| into| simpler| sub|tasks|.| By| decom|posing| tasks|,| the| model| can| effectively| plan| and| execute| each| step| towards| achieving| the| overall| goal|.|

More simply, we can use the [.pick](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.pick) method to select only the desired key:


```python
chain = rag_chain.pick("answer")

for chunk in chain.stream({"input": "What is Task Decomposition?"}):
    print(f"{chunk}|", end="")
```

    |Task| decomposition| involves| breaking| down| complex| tasks| into| smaller| and| simpler| steps| to| make| them| more| manageable| for| an| agent| or| model| to| handle|.| This| process| helps| in| planning| and| executing| tasks| efficiently| by| dividing| them| into| a| series| of| sub|goals| or| actions|.| Task| decomposition| can| be| achieved| through| techniques| like| Chain| of| Thought| (|Co|T|)| or| Tree| of| Thoughts|,| which| enhance| model| performance| on| intricate| tasks| by| guiding| them| through| step|-by|-step| thinking| processes|.||

## Streaming intermediate steps

Suppose we want to stream not only the final outputs of the chain, but also some intermediate steps. As an example let's take our [Conversational RAG](/docs/tutorials/qa_chat_history) chain. Here we reformulate the user question before passing it to the retriever. This reformulated question is not returned as part of the final output. We could modify our chain to return the new question, but for demonstration purposes we'll leave it as is.


```python
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder

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
contextualize_q_llm = llm.with_config(tags=["contextualize_q_llm"])
history_aware_retriever = create_history_aware_retriever(
    contextualize_q_llm, retriever, contextualize_q_prompt
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
```

Note that above we use `.with_config` to assign a tag to the LLM that is used for the question re-phrasing step. This is not necessary but will make it more convenient to stream output from that specific step.

To demonstrate, we will pass in an artificial message history:
```
Human: What is task decomposition?

AI: Task decomposition involves breaking up a complex task into smaller and simpler steps.
```
We then ask a follow up question: "What are some common ways of doing it?" Leading into the retrieval step, our `history_aware_retriever` will rephrase this question using the conversation's context to ensure that the retrieval is meaningful.

To stream intermediate output, we recommend use of the async `.astream_events` method. This method will stream output from all "events" in the chain, and can be quite verbose. We can filter using tags, event types, and other criteria, as we do here.

Below we show a typical `.astream_events` loop, where we pass in the chain input and emit desired results. See the [API reference](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html#langchain_core.runnables.base.Runnable.astream_events) and [streaming guide](/docs/how_to/streaming) for more detail.


```python
first_question = "What is task decomposition?"
first_answer = (
    "Task decomposition involves breaking up "
    "a complex task into smaller and simpler "
    "steps."
)
follow_up_question = "What are some common ways of doing it?"

chat_history = [
    ("human", first_question),
    ("ai", first_answer),
]


async for event in rag_chain.astream_events(
    {
        "input": follow_up_question,
        "chat_history": chat_history,
    },
    version="v1",
):
    if (
        event["event"] == "on_chat_model_stream"
        and "contextualize_q_llm" in event["tags"]
    ):
        ai_message_chunk = event["data"]["chunk"]
        print(f"{ai_message_chunk.content}|", end="")
```

    |What| are| some| typical| methods| used| for| task| decomposition|?||

Here we recover, token-by-token, the query that is passed into the retriever given our question "What are some common ways of doing it?"

If we wanted to get our retrieved docs, we could filter on name "Retriever":


```python
async for event in rag_chain.astream_events(
    {
        "input": follow_up_question,
        "chat_history": chat_history,
    },
    version="v1",
):
    if event["name"] == "Retriever":
        print(event)
        print()
```

    {'event': 'on_retriever_start', 'name': 'Retriever', 'run_id': '6834097c-07fe-42f5-a566-a4780af4d1d0', 'tags': ['seq:step:4', 'Chroma', 'OpenAIEmbeddings'], 'metadata': {}, 'data': {'input': {'query': 'What are some typical methods used for task decomposition?'}}}
    
    {'event': 'on_retriever_end', 'name': 'Retriever', 'run_id': '6834097c-07fe-42f5-a566-a4780af4d1d0', 'tags': ['seq:step:4', 'Chroma', 'OpenAIEmbeddings'], 'metadata': {}, 'data': {'input': {'query': 'What are some typical methods used for task decomposition?'}, 'output': {'documents': [Document(page_content='Tree of Thoughts (Yao et al. 2023) extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.\nTask decomposition can be done (1) by LLM with simple prompting like "Steps for XYZ.\\n1.", "What are the subgoals for achieving XYZ?", (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or (3) with human inputs.', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}), Document(page_content='Fig. 1. Overview of a LLM-powered autonomous agent system.\nComponent One: Planning#\nA complicated task usually involves many steps. An agent needs to know what they are and plan ahead.\nTask Decomposition#\nChain of thought (CoT; Wei et al. 2022) has become a standard prompting technique for enhancing model performance on complex tasks. The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}), Document(page_content='Resources:\n1. Internet access for searches and information gathering.\n2. Long Term memory management.\n3. GPT-3.5 powered Agents for delegation of simple tasks.\n4. File output.\n\nPerformance Evaluation:\n1. Continuously review and analyze your actions to ensure you are performing to the best of your abilities.\n2. Constructively self-criticize your big-picture behavior constantly.\n3. Reflect on past decisions and strategies to refine your approach.\n4. Every command has a cost, so be smart and efficient. Aim to complete tasks in the least number of steps.', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'}), Document(page_content='Fig. 9. Comparison of MIPS algorithms, measured in recall@10. (Image source: Google Blog, 2020)\nCheck more MIPS algorithms and performance comparison in ann-benchmarks.com.\nComponent Three: Tool Use#\nTool use is a remarkable and distinguishing characteristic of human beings. We create, modify and utilize external objects to do things that go beyond our physical and cognitive limits. Equipping LLMs with external tools can significantly extend the model capabilities.', metadata={'source': 'https://lilianweng.github.io/posts/2023-06-23-agent/'})]}}}
    


For more on how to stream intermediate steps check out the [streaming guide](/docs/how_to/streaming).
