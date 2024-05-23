---
sidebar_position: 7
---
# How deal with high cardinality categoricals when doing query analysis

You may want to do query analysis to create a filter on a categorical column. One of the difficulties here is that you usually need to specify the EXACT categorical value. The issue is you need to make sure the LLM generates that categorical value exactly. This can be done relatively easy with prompting when there are only a few values that are valid. When there are a high number of valid values then it becomes more difficult, as those values may not fit in the LLM context, or (if they do) there may be too many for the LLM to properly attend to.

In this notebook we take a look at how to approach this.

## Setup
#### Install dependencies


```python
# %pip install -qU langchain langchain-community langchain-openai faker langchain-chroma
```

#### Set environment variables

We'll use OpenAI in this example:


```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()

# Optional, uncomment to trace runs with LangSmith. Sign up here: https://smith.langchain.com.
# os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

#### Set up data

We will generate a bunch of fake names


```python
from faker import Faker

fake = Faker()

names = [fake.name() for _ in range(10000)]
```

Let's look at some of the names


```python
names[0]
```




    'Hayley Gonzalez'




```python
names[567]
```




    'Jesse Knight'



## Query Analysis

We can now set up a baseline query analysis


```python
from langchain_core.pydantic_v1 import BaseModel, Field
```


```python
class Search(BaseModel):
    query: str
    author: str
```


```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

system = """Generate a relevant search query for a library system"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
structured_llm = llm.with_structured_output(Search)
query_analyzer = {"question": RunnablePassthrough()} | prompt | structured_llm
```

    /Users/harrisonchase/workplace/langchain/libs/core/langchain_core/_api/beta_decorator.py:86: LangChainBetaWarning: The function `with_structured_output` is in beta. It is actively being worked on, so the API may change.
      warn_beta(


We can see that if we spell the name exactly correctly, it knows how to handle it


```python
query_analyzer.invoke("what are books about aliens by Jesse Knight")
```




    Search(query='books about aliens', author='Jesse Knight')



The issue is that the values you want to filter on may NOT be spelled exactly correctly


```python
query_analyzer.invoke("what are books about aliens by jess knight")
```




    Search(query='books about aliens', author='Jess Knight')



### Add in all values

One way around this is to add ALL possible values to the prompt. That will generally guide the query in the right direction


```python
system = """Generate a relevant search query for a library system.

`author` attribute MUST be one of:

{authors}

Do NOT hallucinate author name!"""
base_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
prompt = base_prompt.partial(authors=", ".join(names))
```


```python
query_analyzer_all = {"question": RunnablePassthrough()} | prompt | structured_llm
```

However... if the list of categoricals is long enough, it may error!


```python
try:
    res = query_analyzer_all.invoke("what are books about aliens by jess knight")
except Exception as e:
    print(e)
```

    Error code: 400 - {'error': {'message': "This model's maximum context length is 16385 tokens. However, your messages resulted in 33885 tokens (33855 in the messages, 30 in the functions). Please reduce the length of the messages or functions.", 'type': 'invalid_request_error', 'param': 'messages', 'code': 'context_length_exceeded'}}


We can try to use a longer context window... but with so much information in there, it is not garunteed to pick it up reliably


```python
llm_long = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0)
structured_llm_long = llm_long.with_structured_output(Search)
query_analyzer_all = {"question": RunnablePassthrough()} | prompt | structured_llm_long
```


```python
query_analyzer_all.invoke("what are books about aliens by jess knight")
```




    Search(query='aliens', author='Kevin Knight')



### Find and all relevant values

Instead, what we can do is create an index over the relevant values and then query that for the N most relevant values,


```python
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_texts(names, embeddings, collection_name="author_names")
```


```python
def select_names(question):
    _docs = vectorstore.similarity_search(question, k=10)
    _names = [d.page_content for d in _docs]
    return ", ".join(_names)
```


```python
create_prompt = {
    "question": RunnablePassthrough(),
    "authors": select_names,
} | base_prompt
```


```python
query_analyzer_select = create_prompt | structured_llm
```


```python
create_prompt.invoke("what are books by jess knight")
```




    ChatPromptValue(messages=[SystemMessage(content='Generate a relevant search query for a library system.\n\n`author` attribute MUST be one of:\n\nJesse Knight, Kelly Knight, Scott Knight, Richard Knight, Andrew Knight, Katherine Knight, Erica Knight, Ashley Knight, Becky Knight, Kevin Knight\n\nDo NOT hallucinate author name!'), HumanMessage(content='what are books by jess knight')])




```python
query_analyzer_select.invoke("what are books about aliens by jess knight")
```




    Search(query='books about aliens', author='Jesse Knight')



### Replace after selection

Another method is to let the LLM fill in whatever value, but then convert that value to a valid value.
This can actually be done with the Pydantic class itself!


```python
from langchain_core.pydantic_v1 import validator


class Search(BaseModel):
    query: str
    author: str

    @validator("author")
    def double(cls, v: str) -> str:
        return vectorstore.similarity_search(v, k=1)[0].page_content
```


```python
system = """Generate a relevant search query for a library system"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{question}"),
    ]
)
corrective_structure_llm = llm.with_structured_output(Search)
corrective_query_analyzer = (
    {"question": RunnablePassthrough()} | prompt | corrective_structure_llm
)
```


```python
corrective_query_analyzer.invoke("what are books about aliens by jes knight")
```




    Search(query='books about aliens', author='Jesse Knight')




```python
# TODO: show trigram similarity
```
