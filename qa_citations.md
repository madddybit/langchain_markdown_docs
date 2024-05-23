# How to get a RAG application to add citations

This guide reviews methods to get a model to cite which parts of the source documents it referenced in generating its response.

We will cover five methods:

1. Using tool-calling to cite document IDs;
2. Using tool-calling to cite documents IDs and provide text snippets;
3. Direct prompting;
4. Retrieval post-processing (i.e., compressing the retrieved context to make it more relevant);
5. Generation post-processing (i.e., issuing a second LLM call to annotate a generated answer with citations).

We generally suggest using the first item of the list that works for your use-case. That is, if your model supports tool-calling, try methods 1 or 2; otherwise, or if those fail, advance down the list.

Let's first create a simple RAG chain. To start we'll just retrieve from Wikipedia using the [WikipediaRetriever](https://api.python.langchain.com/en/latest/retrievers/langchain_community.retrievers.wikipedia.WikipediaRetriever.html).

## Setup

First we'll need to install some dependencies and set environment vars for the models we'll be using.


```python
%pip install -qU langchain langchain-openai langchain-anthropic langchain-community wikipedia
```


```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass()
os.environ["ANTHROPIC_API_KEY"] = getpass.getpass()

# Uncomment if you want to log to LangSmith
# os.environ["LANGCHAIN_TRACING_V2"] = "true
# os.environ["LANGCHAIN_API_KEY"] = getpass.getpass()
```

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


```python
from langchain_community.retrievers import WikipediaRetriever
from langchain_core.prompts import ChatPromptTemplate

system_prompt = (
    "You're a helpful AI assistant. Given a user question "
    "and some Wikipedia article snippets, answer the user "
    "question. If none of the articles answer the question, "
    "just say you don't know."
    "\n\nHere are the Wikipedia articles: "
    "{context}"
)

retriever = WikipediaRetriever(top_k_results=6, doc_content_chars_max=2000)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
prompt.pretty_print()
```

    ================================[1m System Message [0m================================
    
    You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, answer the user question. If none of the articles answer the question, just say you don't know.
    
    Here are the Wikipedia articles: [33;1m[1;3m{context}[0m
    
    ================================[1m Human Message [0m=================================
    
    [33;1m[1;3m{input}[0m


Now that we've got a model, retriver and prompt, let's chain them all together. We'll need to add some logic for formatting our retrieved Documents to a string that can be passed to our prompt. Following the how-to guide on [adding citations](/docs/how_to/qa_citations) to a RAG application, we'll make it so our chain returns both the answer and the retrieved Documents.


```python
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


def format_docs(docs: List[Document]):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

retrieve_docs = (lambda x: x["input"]) | retriever

chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
    answer=rag_chain_from_docs
)
```


```python
result = chain.invoke({"input": "How fast are cheetahs?"})
```


```python
print(result.keys())
```

    dict_keys(['input', 'context', 'answer'])



```python
print(result["context"][0])
```

    page_content='The cheetah (Acinonyx jubatus) is a large cat and the fastest land animal. It has a tawny to creamy white or pale buff fur that is marked with evenly spaced, solid black spots. The head is small and rounded, with a short snout and black tear-like facial streaks. It reaches 67–94 cm (26–37 in) at the shoulder, and the head-and-body length is between 1.1 and 1.5 m (3 ft 7 in and 4 ft 11 in). Adults weigh between 21 and 72 kg (46 and 159 lb). The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail.\nThe cheetah was first described in the late 18th century. Four subspecies are recognised today that are native to Africa and central Iran. An African subspecies was introduced to India in 2022. It is now distributed mainly in small, fragmented populations in northwestern, eastern and southern Africa and central Iran. It lives in a variety of habitats such as savannahs in the Serengeti, arid mountain ranges in the Sahara, and hilly desert terrain.\nThe cheetah lives in three main social groups: females and their cubs, male "coalitions", and solitary males. While females lead a nomadic life searching for prey in large home ranges, males are more sedentary and instead establish much smaller territories in areas with plentiful prey and access to females. The cheetah is active during the day, with peaks during dawn and dusk. It feeds on small- to medium-sized prey, mostly weighing under 40 kg (88 lb), and prefers medium-sized ungulates such as impala, springbok and Thomson\'s gazelles. The cheetah typically stalks its prey within 60–100 m (200–330 ft) before charging towards it, trips it during the chase and bites its throat to suffocate it to death. It breeds throughout the year. After a gestation of nearly three months, females give birth to a litter of three or four cubs. Cheetah cubs are highly vulnerable to predation by other large carnivores. They are weaned a' metadata={'title': 'Cheetah', 'summary': 'The cheetah (Acinonyx jubatus) is a large cat and the fastest land animal. It has a tawny to creamy white or pale buff fur that is marked with evenly spaced, solid black spots. The head is small and rounded, with a short snout and black tear-like facial streaks. It reaches 67–94 cm (26–37 in) at the shoulder, and the head-and-body length is between 1.1 and 1.5 m (3 ft 7 in and 4 ft 11 in). Adults weigh between 21 and 72 kg (46 and 159 lb). The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail.\nThe cheetah was first described in the late 18th century. Four subspecies are recognised today that are native to Africa and central Iran. An African subspecies was introduced to India in 2022. It is now distributed mainly in small, fragmented populations in northwestern, eastern and southern Africa and central Iran. It lives in a variety of habitats such as savannahs in the Serengeti, arid mountain ranges in the Sahara, and hilly desert terrain.\nThe cheetah lives in three main social groups: females and their cubs, male "coalitions", and solitary males. While females lead a nomadic life searching for prey in large home ranges, males are more sedentary and instead establish much smaller territories in areas with plentiful prey and access to females. The cheetah is active during the day, with peaks during dawn and dusk. It feeds on small- to medium-sized prey, mostly weighing under 40 kg (88 lb), and prefers medium-sized ungulates such as impala, springbok and Thomson\'s gazelles. The cheetah typically stalks its prey within 60–100 m (200–330 ft) before charging towards it, trips it during the chase and bites its throat to suffocate it to death. It breeds throughout the year. After a gestation of nearly three months, females give birth to a litter of three or four cubs. Cheetah cubs are highly vulnerable to predation by other large carnivores. They are weaned at around four months and are independent by around 20 months of age.\nThe cheetah is threatened by habitat loss, conflict with humans, poaching and high susceptibility to diseases. In 2016, the global cheetah population was estimated at 7,100 individuals in the wild; it is listed as Vulnerable on the IUCN Red List. It has been widely depicted in art, literature, advertising, and animation. It was tamed in ancient Egypt and trained for hunting ungulates in the Arabian Peninsula and India. It has been kept in zoos since the early 19th century.', 'source': 'https://en.wikipedia.org/wiki/Cheetah'}



```python
print(result["answer"])
```

    Cheetahs are capable of running at speeds of 93 to 104 km/h (58 to 65 mph). They have evolved specialized adaptations for speed, including a light build, long thin legs, and a long tail.


LangSmith trace: https://smith.langchain.com/public/0472c5d1-49dc-4c1c-8100-61910067d7ed/r

## Function-calling

If your LLM of choice implements a [tool-calling](/docs/concepts#functiontool-calling) feature, you can use it to make the model specify which of the provided documents it's referencing when generating its answer. LangChain tool-calling models implement a `.with_structured_output` method which will force generation adhering to a desired schema (see for example [here](https://api.python.langchain.com/en/latest/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html#langchain_openai.chat_models.base.ChatOpenAI.with_structured_output)).

### Cite documents

To cite documents using an identifier, we format the identifiers into the prompt, then use `.with_structured_output` to coerce the LLM to reference these identifiers in its output.

First we define a schema for the output. The `.with_structured_output` supports multiple formats, including JSON schema and Pydantic. Here we will use Pydantic:


```python
from langchain_core.pydantic_v1 import BaseModel, Field


class CitedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[int] = Field(
        ...,
        description="The integer IDs of the SPECIFIC sources which justify the answer.",
    )
```

Let's see what the model output is like when we pass in our functions and a user input:


```python
structured_llm = llm.with_structured_output(CitedAnswer)

example_q = """What Brian's height?

Source: 1
Information: Suzy is 6'2"

Source: 2
Information: Jeremiah is blonde

Source: 3
Information: Brian is 3 inches shorter than Suzy"""
result = structured_llm.invoke(example_q)

result
```




    CitedAnswer(answer='Brian\'s height is 5\'11".', citations=[1, 3])



Or as a dict:


```python
result.dict()
```




    {'answer': 'Brian\'s height is 5\'11".', 'citations': [1, 3]}



Now we structure the source identifiers into the prompt to replicate with our chain. We will make three changes:

1. Update the prompt to include source identifiers;
2. Use the `structured_llm` (i.e., `llm.with_structured_output(CitedAnswer));
3. Remove the `StrOutputParser`, to retain the Pydantic object in the output.


```python
def format_docs_with_id(docs: List[Document]) -> str:
    formatted = [
        f"Source ID: {i}\nArticle Title: {doc.metadata['title']}\nArticle Snippet: {doc.page_content}"
        for i, doc in enumerate(docs)
    ]
    return "\n\n" + "\n\n".join(formatted)


rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
    | prompt
    | structured_llm
)

retrieve_docs = (lambda x: x["input"]) | retriever

chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
    answer=rag_chain_from_docs
)
```


```python
result = chain.invoke({"input": "How fast are cheetahs?"})
```


```python
print(result["answer"])
```

    answer='Cheetahs can run at speeds of 93 to 104 km/h (58 to 65 mph). They are known as the fastest land animals.' citations=[0]


We can inspect the document at index 0, which the model cited:


```python
print(result["context"][0])
```

    page_content='The cheetah (Acinonyx jubatus) is a large cat and the fastest land animal. It has a tawny to creamy white or pale buff fur that is marked with evenly spaced, solid black spots. The head is small and rounded, with a short snout and black tear-like facial streaks. It reaches 67–94 cm (26–37 in) at the shoulder, and the head-and-body length is between 1.1 and 1.5 m (3 ft 7 in and 4 ft 11 in). Adults weigh between 21 and 72 kg (46 and 159 lb). The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail.\nThe cheetah was first described in the late 18th century. Four subspecies are recognised today that are native to Africa and central Iran. An African subspecies was introduced to India in 2022. It is now distributed mainly in small, fragmented populations in northwestern, eastern and southern Africa and central Iran. It lives in a variety of habitats such as savannahs in the Serengeti, arid mountain ranges in the Sahara, and hilly desert terrain.\nThe cheetah lives in three main social groups: females and their cubs, male "coalitions", and solitary males. While females lead a nomadic life searching for prey in large home ranges, males are more sedentary and instead establish much smaller territories in areas with plentiful prey and access to females. The cheetah is active during the day, with peaks during dawn and dusk. It feeds on small- to medium-sized prey, mostly weighing under 40 kg (88 lb), and prefers medium-sized ungulates such as impala, springbok and Thomson\'s gazelles. The cheetah typically stalks its prey within 60–100 m (200–330 ft) before charging towards it, trips it during the chase and bites its throat to suffocate it to death. It breeds throughout the year. After a gestation of nearly three months, females give birth to a litter of three or four cubs. Cheetah cubs are highly vulnerable to predation by other large carnivores. They are weaned a' metadata={'title': 'Cheetah', 'summary': 'The cheetah (Acinonyx jubatus) is a large cat and the fastest land animal. It has a tawny to creamy white or pale buff fur that is marked with evenly spaced, solid black spots. The head is small and rounded, with a short snout and black tear-like facial streaks. It reaches 67–94 cm (26–37 in) at the shoulder, and the head-and-body length is between 1.1 and 1.5 m (3 ft 7 in and 4 ft 11 in). Adults weigh between 21 and 72 kg (46 and 159 lb). The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail.\nThe cheetah was first described in the late 18th century. Four subspecies are recognised today that are native to Africa and central Iran. An African subspecies was introduced to India in 2022. It is now distributed mainly in small, fragmented populations in northwestern, eastern and southern Africa and central Iran. It lives in a variety of habitats such as savannahs in the Serengeti, arid mountain ranges in the Sahara, and hilly desert terrain.\nThe cheetah lives in three main social groups: females and their cubs, male "coalitions", and solitary males. While females lead a nomadic life searching for prey in large home ranges, males are more sedentary and instead establish much smaller territories in areas with plentiful prey and access to females. The cheetah is active during the day, with peaks during dawn and dusk. It feeds on small- to medium-sized prey, mostly weighing under 40 kg (88 lb), and prefers medium-sized ungulates such as impala, springbok and Thomson\'s gazelles. The cheetah typically stalks its prey within 60–100 m (200–330 ft) before charging towards it, trips it during the chase and bites its throat to suffocate it to death. It breeds throughout the year. After a gestation of nearly three months, females give birth to a litter of three or four cubs. Cheetah cubs are highly vulnerable to predation by other large carnivores. They are weaned at around four months and are independent by around 20 months of age.\nThe cheetah is threatened by habitat loss, conflict with humans, poaching and high susceptibility to diseases. In 2016, the global cheetah population was estimated at 7,100 individuals in the wild; it is listed as Vulnerable on the IUCN Red List. It has been widely depicted in art, literature, advertising, and animation. It was tamed in ancient Egypt and trained for hunting ungulates in the Arabian Peninsula and India. It has been kept in zoos since the early 19th century.', 'source': 'https://en.wikipedia.org/wiki/Cheetah'}


LangSmith trace: https://smith.langchain.com/public/aff39dc7-3e09-4d64-8083-87026d975534/r

### Cite snippets

To return text spans (perhaps in addition to source identifiers), we can use the same approach. The only change will be to build a more complex output schema, here using Pydantic, that includes a "quote" alongside a source identifier.

*Aside: Note that if we break up our documents so that we have many documents with only a sentence or two instead of a few long documents, citing documents becomes roughly equivalent to citing snippets, and may be easier for the model because the model just needs to return an identifier for each snippet instead of the actual text. Probably worth trying both approaches and evaluating.*


```python
class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class QuotedAnswer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )
```


```python
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs_with_id(x["context"])))
    | prompt
    | llm.with_structured_output(QuotedAnswer)
)

retrieve_docs = (lambda x: x["input"]) | retriever

chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
    answer=rag_chain_from_docs
)
```


```python
result = chain.invoke({"input": "How fast are cheetahs?"})
```

Here we see that the model has extracted a relevant snippet of text from source 0:


```python
result["answer"]
```




    QuotedAnswer(answer='Cheetahs can run at speeds of 93 to 104 km/h (58 to 65 mph).', citations=[Citation(source_id=0, quote='The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail.')])



LangSmith trace: https://smith.langchain.com/public/0f638cc9-8409-4a53-9010-86ac28144129/r

## Direct prompting

Many models don't support function-calling. We can achieve similar results with direct prompting. Let's try instructing a model to generate structured XML for its output:


```python
xml_system = """You're a helpful AI assistant. Given a user question and some Wikipedia article snippets, \
answer the user question and provide citations. If none of the articles answer the question, just say you don't know.

Remember, you must return both an answer and citations. A citation consists of a VERBATIM quote that \
justifies the answer and the ID of the quote article. Return a citation for every quote across all articles \
that justify the answer. Use the following format for your final output:

<cited_answer>
    <answer></answer>
    <citations>
        <citation><source_id></source_id><quote></quote></citation>
        <citation><source_id></source_id><quote></quote></citation>
        ...
    </citations>
</cited_answer>

Here are the Wikipedia articles:{context}"""
xml_prompt = ChatPromptTemplate.from_messages(
    [("system", xml_system), ("human", "{input}")]
)
```

We now make similar small updates to our chain:

1. We update the formatting function to wrap the retrieved context in XML tags;
2. We do not use `.with_structured_output` (e.g., because it does not exist for a model);
3. We use [XMLOutputParser](https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.xml.XMLOutputParser.html) in place of `StrOutputParser` to parse the answer into a dict.


```python
from langchain_core.output_parsers import XMLOutputParser


def format_docs_xml(docs: List[Document]) -> str:
    formatted = []
    for i, doc in enumerate(docs):
        doc_str = f"""\
    <source id=\"{i}\">
        <title>{doc.metadata['title']}</title>
        <article_snippet>{doc.page_content}</article_snippet>
    </source>"""
        formatted.append(doc_str)
    return "\n\n<sources>" + "\n".join(formatted) + "</sources>"


rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs_xml(x["context"])))
    | xml_prompt
    | llm
    | XMLOutputParser()
)

retrieve_docs = (lambda x: x["input"]) | retriever

chain = RunnablePassthrough.assign(context=retrieve_docs).assign(
    answer=rag_chain_from_docs
)
```


```python
result = chain.invoke({"input": "How fast are cheetahs?"})
```

Note that citations are again structured into the answer:


```python
result["answer"]
```




    {'cited_answer': [{'answer': 'Cheetahs are capable of running at 93 to 104 km/h (58 to 65 mph).'},
      {'citations': [{'citation': [{'source_id': '0'},
          {'quote': 'The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail.'}]}]}]}



LangSmith trace: https://smith.langchain.com/public/a3636c70-39c6-4c8f-bc83-1c7a174c237e/r

## Retrieval post-processing

Another approach is to post-process our retrieved documents to compress the content, so that the source content is already minimal enough that we don't need the model to cite specific sources or spans. For example, we could break up each document into a sentence or two, embed those and keep only the most relevant ones. LangChain has some built-in components for this. Here we'll use a [RecursiveCharacterTextSplitter](https://api.python.langchain.com/en/latest/text_splitter/langchain_text_splitters.RecursiveCharacterTextSplitter.html#langchain_text_splitters.RecursiveCharacterTextSplitter), which creates chunks of a sepacified size by splitting on separator substrings, and an [EmbeddingsFilter](https://api.python.langchain.com/en/latest/retrievers/langchain.retrievers.document_compressors.embeddings_filter.EmbeddingsFilter.html#langchain.retrievers.document_compressors.embeddings_filter.EmbeddingsFilter), which keeps only the texts with the most relevant embeddings.

This approach effectively swaps our original retriever with an updated one that compresses the documents. To start, we build the retriever:


```python
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain_core.runnables import RunnableParallel
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=0,
    separators=["\n\n", "\n", ".", " "],
    keep_separator=False,
)
compressor = EmbeddingsFilter(embeddings=OpenAIEmbeddings(), k=10)


def split_and_filter(input) -> List[Document]:
    docs = input["docs"]
    question = input["question"]
    split_docs = splitter.split_documents(docs)
    stateful_docs = compressor.compress_documents(split_docs, question)
    return [stateful_doc for stateful_doc in stateful_docs]


new_retriever = (
    RunnableParallel(question=RunnablePassthrough(), docs=retriever) | split_and_filter
)
docs = new_retriever.invoke("How fast are cheetahs?")
for doc in docs:
    print(doc.page_content)
    print("\n\n")
```

    Adults weigh between 21 and 72 kg (46 and 159 lb). The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail
    
    
    
    The cheetah (Acinonyx jubatus) is a large cat and the fastest land animal. It has a tawny to creamy white or pale buff fur that is marked with evenly spaced, solid black spots. The head is small and rounded, with a short snout and black tear-like facial streaks. It reaches 67–94 cm (26–37 in) at the shoulder, and the head-and-body length is between 1.1 and 1.5 m (3 ft 7 in and 4 ft 11 in)
    
    
    
    2 mph), or 171 body lengths per second. The cheetah, the fastest land mammal, scores at only 16 body lengths per second, while Anna's hummingbird has the highest known length-specific velocity attained by any vertebrate
    
    
    
    It feeds on small- to medium-sized prey, mostly weighing under 40 kg (88 lb), and prefers medium-sized ungulates such as impala, springbok and Thomson's gazelles. The cheetah typically stalks its prey within 60–100 m (200–330 ft) before charging towards it, trips it during the chase and bites its throat to suffocate it to death. It breeds throughout the year
    
    
    
    The cheetah was first described in the late 18th century. Four subspecies are recognised today that are native to Africa and central Iran. An African subspecies was introduced to India in 2022. It is now distributed mainly in small, fragmented populations in northwestern, eastern and southern Africa and central Iran
    
    
    
    The cheetah lives in three main social groups: females and their cubs, male "coalitions", and solitary males. While females lead a nomadic life searching for prey in large home ranges, males are more sedentary and instead establish much smaller territories in areas with plentiful prey and access to females. The cheetah is active during the day, with peaks during dawn and dusk
    
    
    
    The Southeast African cheetah (Acinonyx jubatus jubatus) is the nominate cheetah subspecies native to East and Southern Africa. The Southern African cheetah lives mainly in the lowland areas and deserts of the Kalahari, the savannahs of Okavango Delta, and the grasslands of the Transvaal region in South Africa. In Namibia, cheetahs are mostly found in farmlands
    
    
    
    Subpopulations have been called "South African cheetah" and "Namibian cheetah."
    
    
    
    In India, four cheetahs of the subspecies are living in Kuno National Park in Madhya Pradesh after having been introduced there
    
    
    
    Acinonyx jubatus velox proposed in 1913 by Edmund Heller on basis of a cheetah that was shot by Kermit Roosevelt in June 1909 in the Kenyan highlands.
    Acinonyx rex proposed in 1927 by Reginald Innes Pocock on basis of a specimen from the Umvukwe Range in Rhodesia.
    
    
    


Next, we assemble it into our chain as before:


```python
rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

chain = RunnablePassthrough.assign(
    context=(lambda x: x["input"]) | new_retriever
).assign(answer=rag_chain_from_docs)
```


```python
result = chain.invoke({"input": "How fast are cheetahs?"})

print(result["answer"])
```

    Cheetahs are capable of running at speeds between 93 to 104 km/h (58 to 65 mph), making them the fastest land animals.


Note that the document content is now compressed, although the document objects retain the original content in a "summary" key in their metadata. These summaries are not passed to the model; only the condensed content is.


```python
result["context"][0].page_content  # passed to model
```




    'Adults weigh between 21 and 72 kg (46 and 159 lb). The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail'




```python
result["context"][0].metadata["summary"]  # original document
```




    'The cheetah (Acinonyx jubatus) is a large cat and the fastest land animal. It has a tawny to creamy white or pale buff fur that is marked with evenly spaced, solid black spots. The head is small and rounded, with a short snout and black tear-like facial streaks. It reaches 67–94 cm (26–37 in) at the shoulder, and the head-and-body length is between 1.1 and 1.5 m (3 ft 7 in and 4 ft 11 in). Adults weigh between 21 and 72 kg (46 and 159 lb). The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail.\nThe cheetah was first described in the late 18th century. Four subspecies are recognised today that are native to Africa and central Iran. An African subspecies was introduced to India in 2022. It is now distributed mainly in small, fragmented populations in northwestern, eastern and southern Africa and central Iran. It lives in a variety of habitats such as savannahs in the Serengeti, arid mountain ranges in the Sahara, and hilly desert terrain.\nThe cheetah lives in three main social groups: females and their cubs, male "coalitions", and solitary males. While females lead a nomadic life searching for prey in large home ranges, males are more sedentary and instead establish much smaller territories in areas with plentiful prey and access to females. The cheetah is active during the day, with peaks during dawn and dusk. It feeds on small- to medium-sized prey, mostly weighing under 40 kg (88 lb), and prefers medium-sized ungulates such as impala, springbok and Thomson\'s gazelles. The cheetah typically stalks its prey within 60–100 m (200–330 ft) before charging towards it, trips it during the chase and bites its throat to suffocate it to death. It breeds throughout the year. After a gestation of nearly three months, females give birth to a litter of three or four cubs. Cheetah cubs are highly vulnerable to predation by other large carnivores. They are weaned at around four months and are independent by around 20 months of age.\nThe cheetah is threatened by habitat loss, conflict with humans, poaching and high susceptibility to diseases. In 2016, the global cheetah population was estimated at 7,100 individuals in the wild; it is listed as Vulnerable on the IUCN Red List. It has been widely depicted in art, literature, advertising, and animation. It was tamed in ancient Egypt and trained for hunting ungulates in the Arabian Peninsula and India. It has been kept in zoos since the early 19th century.'



LangSmith trace: https://smith.langchain.com/public/a61304fa-e5a5-4c64-a268-b0aef1130d53/r

## Generation post-processing

Another approach is to post-process our model generation. In this example we'll first generate just an answer, and then we'll ask the model to annotate it's own answer with citations. The downside of this approach is of course that it is slower and more expensive, because two model calls need to be made.

Let's apply this to our initial chain.


```python
class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class AnnotatedAnswer(BaseModel):
    """Annotate the answer to the user question with quote citations that justify the answer."""

    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


structured_llm = llm.with_structured_output(AnnotatedAnswer)
```


```python
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}"),
        MessagesPlaceholder("chat_history", optional=True),
    ]
)
answer = prompt | llm
annotation_chain = prompt | structured_llm

chain = (
    RunnableParallel(
        question=RunnablePassthrough(), docs=(lambda x: x["input"]) | retriever
    )
    .assign(context=format)
    .assign(ai_message=answer)
    .assign(
        chat_history=(lambda x: [x["ai_message"]]),
        answer=(lambda x: x["ai_message"].content),
    )
    .assign(annotations=annotation_chain)
    .pick(["answer", "docs", "annotations"])
)
```


```python
result = chain.invoke({"input": "How fast are cheetahs?"})
```


```python
print(result["answer"])
```

    Cheetahs are capable of running at speeds between 93 to 104 km/h (58 to 65 mph). Their specialized adaptations for speed, such as a light build, long thin legs, and a long tail, allow them to be the fastest land animals.



```python
result["annotations"]
```




    AnnotatedAnswer(citations=[Citation(source_id=0, quote='The cheetah is capable of running at 93 to 104 km/h (58 to 65 mph); it has evolved specialized adaptations for speed, including a light build, long thin legs and a long tail.')])



LangSmith trace: https://smith.langchain.com/public/bf5e8856-193b-4ff2-af8d-c0f4fbd1d9cb/r
