# How to load PDFs

[Portable Document Format (PDF)](https://en.wikipedia.org/wiki/PDF), standardized as ISO 32000, is a file format developed by Adobe in 1992 to present documents, including text formatting and images, in a manner independent of application software, hardware, and operating systems.

This guide covers how to load `PDF` documents into the LangChain [Document](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html#langchain_core.documents.base.Document) format that we use downstream.

LangChain integrates with a host of PDF parsers. Some are simple and relatively low-level; others will support OCR and image-processing, or perform advanced document layout analysis. The right choice will depend on your application. Below we enumerate the possibilities.

## Using PyPDF

Here we load a PDF using `pypdf` into array of documents, where each document contains the page content and metadata with `page` number.


```python
%pip install pypdf
```


```python
from langchain_community.document_loaders import PyPDFLoader

file_path = (
    "../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf"
)
loader = PyPDFLoader(file_path)
pages = loader.load_and_split()

pages[0]
```




    Document(page_content='LayoutParser : A Uniﬁed Toolkit for Deep\nLearning Based Document Image Analysis\nZejiang Shen1( \x00), Ruochen Zhang2, Melissa Dell3, Benjamin Charles Germain\nLee4, Jacob Carlson3, and Weining Li5\n1Allen Institute for AI\nshannons@allenai.org\n2Brown University\nruochen zhang@brown.edu\n3Harvard University\n{melissadell,jacob carlson }@fas.harvard.edu\n4University of Washington\nbcgl@cs.washington.edu\n5University of Waterloo\nw422li@uwaterloo.ca\nAbstract. Recent advances in document image analysis (DIA) have been\nprimarily driven by the application of neural networks. Ideally, research\noutcomes could be easily deployed in production and extended for further\ninvestigation. However, various factors like loosely organized codebases\nand sophisticated model conﬁgurations complicate the easy reuse of im-\nportant innovations by a wide audience. Though there have been on-going\neﬀorts to improve reusability and simplify deep learning (DL) model\ndevelopment in disciplines like natural language processing and computer\nvision, none of them are optimized for challenges in the domain of DIA.\nThis represents a major gap in the existing toolkit, as DIA is central to\nacademic research across a wide range of disciplines in the social sciences\nand humanities. This paper introduces LayoutParser , an open-source\nlibrary for streamlining the usage of DL in DIA research and applica-\ntions. The core LayoutParser library comes with a set of simple and\nintuitive interfaces for applying and customizing DL models for layout de-\ntection, character recognition, and many other document processing tasks.\nTo promote extensibility, LayoutParser also incorporates a community\nplatform for sharing both pre-trained models and full document digiti-\nzation pipelines. We demonstrate that LayoutParser is helpful for both\nlightweight and large-scale digitization pipelines in real-word use cases.\nThe library is publicly available at https://layout-parser.github.io .\nKeywords: Document Image Analysis ·Deep Learning ·Layout Analysis\n·Character Recognition ·Open Source library ·Toolkit.\n1 Introduction\nDeep Learning(DL)-based approaches are the state-of-the-art for a wide range of\ndocument image analysis (DIA) tasks including document image classiﬁcation [ 11,arXiv:2103.15348v2  [cs.CV]  21 Jun 2021', metadata={'source': '../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf', 'page': 0})



An advantage of this approach is that documents can be retrieved with page numbers.

### Vector search over PDFs

Once we have loaded PDFs into LangChain `Document` objects, we can index them (e.g., a RAG application) in the usual way:


```python
import getpass
import os

os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
```


```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("What is LayoutParser?", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])
```

    13: 14 Z. Shen et al.
    6 Conclusion
    LayoutParser provides a comprehensive toolkit for deep learning-based document
    image analysis. The oﬀ-the-shelf library is easy to install, and can be used to
    build ﬂexible and accurate pipelines for processing documents with complicated
    structures. It also supports hi
    0: LayoutParser : A Uniﬁed Toolkit for Deep
    Learning Based Document Image Analysis
    Zejiang Shen1(  ), Ruochen Zhang2, Melissa Dell3, Benjamin Charles Germain
    Lee4, Jacob Carlson3, and Weining Li5
    1Allen Institute for AI
    shannons@allenai.org
    2Brown University
    ruochen zhang@brown.edu
    3Harvard University
    


### Extract text from images

Some PDFs contain images of text-- e.g., within scanned documents, or figures. Using the `rapidocr-onnxruntime` package we can extract images as text as well:


```python
%pip install rapidocr-onnxruntime
```


```python
loader = PyPDFLoader("https://arxiv.org/pdf/2103.15348.pdf", extract_images=True)
pages = loader.load()
pages[4].page_content
```




    'LayoutParser : A Uniﬁed Toolkit for DL-Based DIA 5\nTable 1: Current layout detection models in the LayoutParser model zoo\nDataset Base Model1Large Model Notes\nPubLayNet [38] F / M M Layouts of modern scientiﬁc documents\nPRImA [3] M - Layouts of scanned modern magazines and scientiﬁc reports\nNewspaper [17] F - Layouts of scanned US newspapers from the 20th century\nTableBank [18] F F Table region on modern scientiﬁc and business document\nHJDataset [31] F / M - Layouts of history Japanese documents\n1For each dataset, we train several models of diﬀerent sizes for diﬀerent needs (the trade-oﬀ between accuracy\nvs. computational cost). For “base model” and “large model”, we refer to using the ResNet 50 or ResNet 101\nbackbones [ 13], respectively. One can train models of diﬀerent architectures, like Faster R-CNN [ 28] (F) and Mask\nR-CNN [ 12] (M). For example, an F in the Large Model column indicates it has a Faster R-CNN model trained\nusing the ResNet 101 backbone. The platform is maintained and a number of additions will be made to the model\nzoo in coming months.\nlayout data structures , which are optimized for eﬃciency and versatility. 3) When\nnecessary, users can employ existing or customized OCR models via the uniﬁed\nAPI provided in the OCR module . 4)LayoutParser comes with a set of utility\nfunctions for the visualization and storage of the layout data. 5) LayoutParser\nis also highly customizable, via its integration with functions for layout data\nannotation and model training . We now provide detailed descriptions for each\ncomponent.\n3.1 Layout Detection Models\nInLayoutParser , a layout model takes a document image as an input and\ngenerates a list of rectangular boxes for the target content regions. Diﬀerent\nfrom traditional methods, it relies on deep convolutional neural networks rather\nthan manually curated rules to identify content regions. It is formulated as an\nobject detection problem and state-of-the-art models like Faster R-CNN [ 28] and\nMask R-CNN [ 12] are used. This yields prediction results of high accuracy and\nmakes it possible to build a concise, generalized interface for layout detection.\nLayoutParser , built upon Detectron2 [ 35], provides a minimal API that can\nperform layout detection with only four lines of code in Python:\n1import layoutparser as lp\n2image = cv2. imread (" image_file ") # load images\n3model = lp. Detectron2LayoutModel (\n4 "lp :// PubLayNet / faster_rcnn_R_50_FPN_3x / config ")\n5layout = model . detect ( image )\nLayoutParser provides a wealth of pre-trained model weights using various\ndatasets covering diﬀerent languages, time periods, and document types. Due to\ndomain shift [ 7], the prediction performance can notably drop when models are ap-\nplied to target samples that are signiﬁcantly diﬀerent from the training dataset. As\ndocument structures and layouts vary greatly in diﬀerent domains, it is important\nto select models trained on a dataset similar to the test samples. A semantic syntax\nis used for initializing the model weights in LayoutParser , using both the dataset\nname and model name lp://<dataset-name>/<model-architecture-name> .'



## Using PyMuPDF

This is the fastest of the PDF parsing options, and contains detailed metadata about the PDF and its pages, as well as returns one document per page.


```python
from langchain_community.document_loaders import PyMuPDFLoader

loader = PyMuPDFLoader("example_data/layout-parser-paper.pdf")
data = loader.load()
data[0]
```

Additionally, you can pass along any of the options from the [PyMuPDF documentation](https://pymupdf.readthedocs.io/en/latest/app1.html#plain-text/) as keyword arguments in the `load` call, and it will be pass along to the `get_text()` call.

## Using MathPix

Inspired by Daniel Gross's [https://gist.github.com/danielgross/3ab4104e14faccc12b49200843adab21](https://gist.github.com/danielgross/3ab4104e14faccc12b49200843adab21)


```python
from langchain_community.document_loaders import MathpixPDFLoader

file_path = (
    "../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf"
)
loader = MathpixPDFLoader(file_path)
data = loader.load()
```

## Using Unstructured

[Unstructured](https://unstructured-io.github.io/unstructured/) supports a common interface for working with unstructured or semi-structured file formats, such as Markdown or PDF. LangChain's [UnstructuredPDFLoader](https://api.python.langchain.com/en/latest/document_loaders/langchain_community.document_loaders.pdf.UnstructuredPDFLoader.html) integrates with Unstructured to parse PDF documents into LangChain [Document](https://api.python.langchain.com/en/latest/documents/langchain_core.documents.base.Document.html) objects.


```python
from langchain_community.document_loaders import UnstructuredPDFLoader

file_path = (
    "../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf"
)
loader = UnstructuredPDFLoader(file_path)
data = loader.load()
```

### Retain Elements

Under the hood, Unstructured creates different "elements" for different chunks of text. By default we combine those together, but you can easily keep that separation by specifying `mode="elements"`.


```python
file_path = (
    "../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf"
)
loader = UnstructuredPDFLoader(file_path, mode="elements")

data = loader.load()
data[0]
```




    Document(page_content='1 2 0 2', metadata={'source': '../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf', 'coordinates': {'points': ((16.34, 213.36), (16.34, 253.36), (36.34, 253.36), (36.34, 213.36)), 'system': 'PixelSpace', 'layout_width': 612, 'layout_height': 792}, 'file_directory': '../../../docs/integrations/document_loaders/example_data', 'filename': 'layout-parser-paper.pdf', 'languages': ['eng'], 'last_modified': '2024-03-18T13:22:22', 'page_number': 1, 'filetype': 'application/pdf', 'category': 'UncategorizedText'})



See the full set of element types for this particular document:


```python
set(doc.metadata["category"] for doc in data)
```




    {'ListItem', 'NarrativeText', 'Title', 'UncategorizedText'}



### Fetching remote PDFs using Unstructured

This covers how to load online PDFs into a document format that we can use downstream. This can be used for various online PDF sites such as https://open.umn.edu/opentextbooks/textbooks/ and https://arxiv.org/archive/

Note: all other PDF loaders can also be used to fetch remote PDFs, but `OnlinePDFLoader` is a legacy function, and works specifically with `UnstructuredPDFLoader`.


```python
from langchain_community.document_loaders import OnlinePDFLoader

loader = OnlinePDFLoader("https://arxiv.org/pdf/2302.03803.pdf")
data = loader.load()
```

## Using PyPDFium2


```python
from langchain_community.document_loaders import PyPDFium2Loader

file_path = (
    "../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf"
)
loader = PyPDFium2Loader(file_path)
data = loader.load()
```

## Using PDFMiner


```python
from langchain_community.document_loaders import PDFMinerLoader

file_path = (
    "../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf"
)
loader = PDFMinerLoader(file_path)
data = loader.load()
```

### Using PDFMiner to generate HTML text

This can be helpful for chunking texts semantically into sections as the output html content can be parsed via `BeautifulSoup` to get more structured and rich information about font size, page numbers, PDF headers/footers, etc.


```python
from langchain_community.document_loaders import PDFMinerPDFasHTMLLoader

file_path = (
    "../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf"
)
loader = PDFMinerPDFasHTMLLoader(file_path)
data = loader.load()[0]
```


```python
from bs4 import BeautifulSoup

soup = BeautifulSoup(data.page_content, "html.parser")
content = soup.find_all("div")
```


```python
import re

cur_fs = None
cur_text = ""
snippets = []  # first collect all snippets that have the same font size
for c in content:
    sp = c.find("span")
    if not sp:
        continue
    st = sp.get("style")
    if not st:
        continue
    fs = re.findall("font-size:(\d+)px", st)
    if not fs:
        continue
    fs = int(fs[0])
    if not cur_fs:
        cur_fs = fs
    if fs == cur_fs:
        cur_text += c.text
    else:
        snippets.append((cur_text, cur_fs))
        cur_fs = fs
        cur_text = c.text
snippets.append((cur_text, cur_fs))
# Note: The above logic is very straightforward. One can also add more strategies such as removing duplicate snippets (as
# headers/footers in a PDF appear on multiple pages so if we find duplicates it's safe to assume that it is redundant info)
```


```python
from langchain_core.documents import Document

cur_idx = -1
semantic_snippets = []
# Assumption: headings have higher font size than their respective content
for s in snippets:
    # if current snippet's font size > previous section's heading => it is a new heading
    if (
        not semantic_snippets
        or s[1] > semantic_snippets[cur_idx].metadata["heading_font"]
    ):
        metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
        metadata.update(data.metadata)
        semantic_snippets.append(Document(page_content="", metadata=metadata))
        cur_idx += 1
        continue

    # if current snippet's font size <= previous section's content => content belongs to the same section (one can also create
    # a tree like structure for sub sections if needed but that may require some more thinking and may be data specific)
    if (
        not semantic_snippets[cur_idx].metadata["content_font"]
        or s[1] <= semantic_snippets[cur_idx].metadata["content_font"]
    ):
        semantic_snippets[cur_idx].page_content += s[0]
        semantic_snippets[cur_idx].metadata["content_font"] = max(
            s[1], semantic_snippets[cur_idx].metadata["content_font"]
        )
        continue

    # if current snippet's font size > previous section's content but less than previous section's heading than also make a new
    # section (e.g. title of a PDF will have the highest font size but we don't want it to subsume all sections)
    metadata = {"heading": s[0], "content_font": 0, "heading_font": s[1]}
    metadata.update(data.metadata)
    semantic_snippets.append(Document(page_content="", metadata=metadata))
    cur_idx += 1
```


```python
semantic_snippets[4]
```




    Document(page_content='Recently, various DL models and datasets have been developed for layout analysis\ntasks. The dhSegment [22] utilizes fully convolutional networks [20] for segmen-\ntation tasks on historical documents. Object detection-based methods like Faster\nR-CNN [28] and Mask R-CNN [12] are used for identifying document elements [38]\nand detecting tables [30, 26]. Most recently, Graph Neural Networks [29] have also\nbeen used in table detection [27]. However, these models are usually implemented\nindividually and there is no uniﬁed framework to load and use such models.\nThere has been a surge of interest in creating open-source tools for document\nimage processing: a search of document image analysis in Github leads to 5M\nrelevant code pieces 6; yet most of them rely on traditional rule-based methods\nor provide limited functionalities. The closest prior research to our work is the\nOCR-D project7, which also tries to build a complete toolkit for DIA. However,\nsimilar to the platform developed by Neudecker et al. [21], it is designed for\nanalyzing historical documents, and provides no supports for recent DL models.\nThe DocumentLayoutAnalysis project8 focuses on processing born-digital PDF\ndocuments via analyzing the stored PDF data. Repositories like DeepLayout9\nand Detectron2-PubLayNet10 are individual deep learning models trained on\nlayout analysis datasets without support for the full DIA pipeline. The Document\nAnalysis and Exploitation (DAE) platform [15] and the DeepDIVA project [2]\naim to improve the reproducibility of DIA methods (or DL models), yet they\nare not actively maintained. OCR engines like Tesseract [14], easyOCR11 and\npaddleOCR12 usually do not come with comprehensive functionalities for other\nDIA tasks like layout analysis.\nRecent years have also seen numerous eﬀorts to create libraries for promoting\nreproducibility and reusability in the ﬁeld of DL. Libraries like Dectectron2 [35],\n6 The number shown is obtained by specifying the search type as ‘code’.\n7 https://ocr-d.de/en/about\n8 https://github.com/BobLd/DocumentLayoutAnalysis\n9 https://github.com/leonlulu/DeepLayout\n10 https://github.com/hpanwar08/detectron2\n11 https://github.com/JaidedAI/EasyOCR\n12 https://github.com/PaddlePaddle/PaddleOCR\n4\nZ. Shen et al.\nFig. 1: The overall architecture of LayoutParser. For an input document image,\nthe core LayoutParser library provides a set of oﬀ-the-shelf tools for layout\ndetection, OCR, visualization, and storage, backed by a carefully designed layout\ndata structure. LayoutParser also supports high level customization via eﬃcient\nlayout annotation and model training functions. These improve model accuracy\non the target samples. The community platform enables the easy sharing of DIA\nmodels and whole digitization pipelines to promote reusability and reproducibility.\nA collection of detailed documentation, tutorials and exemplar projects make\nLayoutParser easy to learn and use.\nAllenNLP [8] and transformers [34] have provided the community with complete\nDL-based support for developing and deploying models for general computer\nvision and natural language processing problems. LayoutParser, on the other\nhand, specializes speciﬁcally in DIA tasks. LayoutParser is also equipped with a\ncommunity platform inspired by established model hubs such as Torch Hub [23]\nand TensorFlow Hub [1]. It enables the sharing of pretrained models as well as\nfull document processing pipelines that are unique to DIA tasks.\nThere have been a variety of document data collections to facilitate the\ndevelopment of DL models. Some examples include PRImA [3](magazine layouts),\nPubLayNet [38](academic paper layouts), Table Bank [18](tables in academic\npapers), Newspaper Navigator Dataset [16, 17](newspaper ﬁgure layouts) and\nHJDataset [31](historical Japanese document layouts). A spectrum of models\ntrained on these datasets are currently available in the LayoutParser model zoo\nto support diﬀerent use cases.\n', metadata={'heading': '2 Related Work\n', 'content_font': 9, 'heading_font': 11, 'source': '../../../docs/integrations/document_loaders/example_data/layout-parser-paper.pdf'})



## PyPDF Directory

Load PDFs from directory


```python
from langchain_community.document_loaders import PyPDFDirectoryLoader
```


```python
directory_path = "../../../docs/integrations/document_loaders/example_data/"
loader = PyPDFDirectoryLoader("example_data/")


docs = loader.load()
```

## Using PDFPlumber

Like PyMuPDF, the output Documents contain detailed metadata about the PDF and its pages, and returns one document per page.


```python
from langchain_community.document_loaders import PDFPlumberLoader

data = loader.load()
data[0]
```

## Using AmazonTextractPDFParser

The AmazonTextractPDFLoader calls the [Amazon Textract Service](https://aws.amazon.com/textract/) to convert PDFs into a Document structure. The loader does pure OCR at the moment, with more features like layout support planned, depending on demand.  Single and multi-page documents are supported with up to 3000 pages and 512 MB of size.

For the call to be successful an AWS account is required, similar to the [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) requirements.

Besides the AWS configuration, it is very similar to the other PDF loaders, while also supporting JPEG, PNG and TIFF and non-native PDF formats.


```python
from langchain_community.document_loaders import AmazonTextractPDFLoader

loader = AmazonTextractPDFLoader("example_data/alejandro_rosalez_sample-small.jpeg")
documents = loader.load()
```

## Using AzureAIDocumentIntelligenceLoader

[Azure AI Document Intelligence](https://aka.ms/doc-intelligence) (formerly known as `Azure Form Recognizer`) is machine-learning 
based service that extracts texts (including handwriting), tables, document structures (e.g., titles, section headings, etc.) and key-value-pairs from
digital or scanned PDFs, images, Office and HTML files. Document Intelligence supports `PDF`, `JPEG/JPG`, `PNG`, `BMP`, `TIFF`, `HEIF`, `DOCX`, `XLSX`, `PPTX` and `HTML`.

This [current implementation](https://aka.ms/di-langchain) of a loader using `Document Intelligence` can incorporate content page-wise and turn it into LangChain documents. The default output format is markdown, which can be easily chained with `MarkdownHeaderTextSplitter` for semantic document chunking. You can also use `mode="single"` or `mode="page"` to return pure texts in a single page or document split by page.

### Prerequisite

An Azure AI Document Intelligence resource in one of the 3 preview regions: **East US**, **West US2**, **West Europe** - follow [this document](https://learn.microsoft.com/azure/ai-services/document-intelligence/create-document-intelligence-resource?view=doc-intel-4.0.0) to create one if you don't have. You will be passing `<endpoint>` and `<key>` as parameters to the loader.


```python
%pip install --upgrade --quiet  langchain langchain-community azure-ai-documentintelligence
```


```python
from langchain_community.document_loaders import AzureAIDocumentIntelligenceLoader

file_path = "<filepath>"
endpoint = "<endpoint>"
key = "<key>"
loader = AzureAIDocumentIntelligenceLoader(
    api_endpoint=endpoint, api_key=key, file_path=file_path, api_model="prebuilt-layout"
)

documents = loader.load()
```
