# How to handle long text when doing extraction

When working with files, like PDFs, you're likely to encounter text that exceeds your language model's context window. To process this text, consider these strategies:

1. **Change LLM** Choose a different LLM that supports a larger context window.
2. **Brute Force** Chunk the document, and extract content from each chunk.
3. **RAG** Chunk the document, index the chunks, and only extract content from a subset of chunks that look "relevant".

Keep in mind that these strategies have different trade off and the best strategy likely depends on the application that you're designing!

This guide demonstrates how to implement strategies 2 and 3.

## Set up

We need some example data! Let's download an article about [cars from wikipedia](https://en.wikipedia.org/wiki/Car) and load it as a LangChain [Document](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html).


```python
import re

import requests
from langchain_community.document_loaders import BSHTMLLoader

# Download the content
response = requests.get("https://en.wikipedia.org/wiki/Car")
# Write it to a file
with open("car.html", "w", encoding="utf-8") as f:
    f.write(response.text)
# Load it with an HTML parser
loader = BSHTMLLoader("car.html")
document = loader.load()[0]
# Clean up code
# Replace consecutive new lines with a single new line
document.page_content = re.sub("\n\n+", "\n", document.page_content)
```


```python
print(len(document.page_content))
```

    79174


## Define the schema

Following the [extraction tutorial](/docs/tutorials/extraction), we will use Pydantic to define the schema of information we wish to extract. In this case, we will extract a list of "key developments" (e.g., important historical events) that include a year and description.

Note that we also include an `evidence` key and instruct the model to provide in verbatim the relevant sentences of text from the article. This allows us to compare the extraction results to (the model's reconstruction of) text from the original document.


```python
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field


class KeyDevelopment(BaseModel):
    """Information about a development in the history of cars."""

    year: int = Field(
        ..., description="The year when there was an important historic development."
    )
    description: str = Field(
        ..., description="What happened in this year? What was the development?"
    )
    evidence: str = Field(
        ...,
        description="Repeat in verbatim the sentence(s) from which the year and description information were extracted",
    )


class ExtractionData(BaseModel):
    """Extracted information about key developments in the history of cars."""

    key_developments: List[KeyDevelopment]


# Define a custom prompt to provide instructions and any additional context.
# 1) You can add examples into the prompt template to improve extraction quality
# 2) Introduce additional parameters to take context into account (e.g., include metadata
#    about the document from which the text was extracted.)
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert at identifying key historic development in text. "
            "Only extract important historic developments. Extract nothing if no important information can be found in the text.",
        ),
        ("human", "{text}"),
    ]
)
```

## Create an extractor

Let's select an LLM. Because we are using tool-calling, we will need a model that supports a tool-calling feature. See [this table](/docs/integrations/chat) for available LLMs.

```{=mdx}
import ChatModelTabs from "@theme/ChatModelTabs";

<ChatModelTabs
  customVarName="llm"
  openaiParams={`model="gpt-4-0125-preview", temperature=0`}
/>
```


```python
# | output: false
# | echo: false

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4-0125-preview", temperature=0)
```


```python
extractor = prompt | llm.with_structured_output(
    schema=ExtractionData,
    include_raw=False,
)
```

## Brute force approach

Split the documents into chunks such that each chunk fits into the context window of the LLMs.


```python
from langchain_text_splitters import TokenTextSplitter

text_splitter = TokenTextSplitter(
    # Controls the size of each chunk
    chunk_size=2000,
    # Controls overlap between chunks
    chunk_overlap=20,
)

texts = text_splitter.split_text(document.page_content)
```

Use [batch](https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.base.Runnable.html) functionality to run the extraction in **parallel** across each chunk! 

:::{.callout-tip}
You can often use .batch() to parallelize the extractions! `.batch` uses a threadpool under the hood to help you parallelize workloads.

If your model is exposed via an API, this will likely speed up your extraction flow!
:::


```python
# Limit just to the first 3 chunks
# so the code can be re-run quickly
first_few = texts[:3]

extractions = extractor.batch(
    [{"text": text} for text in first_few],
    {"max_concurrency": 5},  # limit the concurrency by passing max concurrency!
)
```

### Merge results

After extracting data from across the chunks, we'll want to merge the extractions together.


```python
key_developments = []

for extraction in extractions:
    key_developments.extend(extraction.key_developments)

key_developments[:10]
```




    [KeyDevelopment(year=1966, description='The Toyota Corolla began production, becoming the best-selling series of automobile in history.', evidence='The Toyota Corolla, which has been in production since 1966, is the best-selling series of automobile in history.'),
     KeyDevelopment(year=1769, description='Nicolas-Joseph Cugnot built the first steam-powered road vehicle.', evidence='The French inventor Nicolas-Joseph Cugnot built the first steam-powered road vehicle in 1769.'),
     KeyDevelopment(year=1808, description='François Isaac de Rivaz designed and constructed the first internal combustion-powered automobile.', evidence='the Swiss inventor François Isaac de Rivaz designed and constructed the first internal combustion-powered automobile in 1808.'),
     KeyDevelopment(year=1886, description='Carl Benz patented his Benz Patent-Motorwagen, inventing the modern car.', evidence='The modern car—a practical, marketable automobile for everyday use—was invented in 1886, when the German inventor Carl Benz patented his Benz Patent-Motorwagen.'),
     KeyDevelopment(year=1908, description='Ford Model T, one of the first cars affordable by the masses, began production.', evidence='One of the first cars affordable by the masses was the Ford Model T, begun in 1908, an American car manufactured by the Ford Motor Company.'),
     KeyDevelopment(year=1888, description="Bertha Benz undertook the first road trip by car to prove the road-worthiness of her husband's invention.", evidence="In August 1888, Bertha Benz, the wife of Carl Benz, undertook the first road trip by car, to prove the road-worthiness of her husband's invention."),
     KeyDevelopment(year=1896, description='Benz designed and patented the first internal-combustion flat engine, called boxermotor.', evidence='In 1896, Benz designed and patented the first internal-combustion flat engine, called boxermotor.'),
     KeyDevelopment(year=1897, description='Nesselsdorfer Wagenbau produced the Präsident automobil, one of the first factory-made cars in the world.', evidence='The first motor car in central Europe and one of the first factory-made cars in the world, was produced by Czech company Nesselsdorfer Wagenbau (later renamed to Tatra) in 1897, the Präsident automobil.'),
     KeyDevelopment(year=1890, description='Daimler Motoren Gesellschaft (DMG) was founded by Daimler and Maybach in Cannstatt.', evidence='Daimler and Maybach founded Daimler Motoren Gesellschaft (DMG) in Cannstatt in 1890.'),
     KeyDevelopment(year=1891, description='Auguste Doriot and Louis Rigoulot completed the longest trip by a petrol-driven vehicle with a Daimler powered Peugeot Type 3.', evidence='In 1891, Auguste Doriot and his Peugeot colleague Louis Rigoulot completed the longest trip by a petrol-driven vehicle when their self-designed and built Daimler powered Peugeot Type 3 completed 2,100 kilometres (1,300 mi) from Valentigney to Paris and Brest and back again.')]



## RAG based approach

Another simple idea is to chunk up the text, but instead of extracting information from every chunk, just focus on the the most relevant chunks.

:::{.callout-caution}
It can be difficult to identify which chunks are relevant.

For example, in the `car` article we're using here, most of the article contains key development information. So by using
**RAG**, we'll likely be throwing out a lot of relevant information.

We suggest experimenting with your use case and determining whether this approach works or not.
:::

To implement the RAG based approach: 

1. Chunk up your document(s) and index them (e.g., in a vectorstore);
2. Prepend the `extractor` chain with a retrieval step using the vectorstore.

Here's a simple example that relies on the `FAISS` vectorstore.


```python
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

texts = text_splitter.split_text(document.page_content)
vectorstore = FAISS.from_texts(texts, embedding=OpenAIEmbeddings())

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 1}
)  # Only extract from first document
```

In this case the RAG extractor is only looking at the top document.


```python
rag_extractor = {
    "text": retriever | (lambda docs: docs[0].page_content)  # fetch content of top doc
} | extractor
```


```python
results = rag_extractor.invoke("Key developments associated with cars")
```


```python
for key_development in results.key_developments:
    print(key_development)
```

    year=1869 description='Mary Ward became one of the first documented car fatalities in Parsonstown, Ireland.' evidence='Mary Ward became one of the first documented car fatalities in 1869 in Parsonstown, Ireland,'
    year=1899 description="Henry Bliss one of the US's first pedestrian car casualties in New York City." evidence="Henry Bliss one of the US's first pedestrian car casualties in 1899 in New York City."
    year=2030 description='All fossil fuel vehicles will be banned in Amsterdam.' evidence='all fossil fuel vehicles will be banned in Amsterdam from 2030.'


## Common issues

Different methods have their own pros and cons related to cost, speed, and accuracy.

Watch out for these issues:

* Chunking content means that the LLM can fail to extract information if the information is spread across multiple chunks.
* Large chunk overlap may cause the same information to be extracted twice, so be prepared to de-duplicate!
* LLMs can make up data. If looking for a single fact across a large text and using a brute force approach, you may end up getting more made up data.
