#!/usr/bin/env python
# coding: utf-8

# # Your First RAG Application
# 
# In this notebook, we'll walk you through each of the components that are involved in a simple RAG application.
# 
# We won't be leveraging any fancy tools, just the OpenAI Python SDK, Numpy, and some classic Python.
# 
# > NOTE: This was done with Python 3.11.4.
# 
# > NOTE: There might be [compatibility issues](https://github.com/wandb/wandb/issues/7683) if you're on NVIDIA driver >552.44 As an interim solution - you can rollback your drivers to the 552.44.

# ## Table of Contents:
# 
# - Task 1: Imports and Utilities
# - Task 2: Documents
# - Task 3: Embeddings and Vectors
# - Task 4: Prompts
# - Task 5: Retrieval Augmented Generation
#   - 🚧 Activity #1: Augment RAG

# Let's look at a rather complicated looking visual representation of a basic RAG application.
# 
# <img src="https://i.imgur.com/vD8b016.png" />

# ## Task 1: Imports and Utility
# 
# We're just doing some imports and enabling `async` to work within the Jupyter environment here, nothing too crazy!

# In[1]:


from aimakerspace.text_utils import TextFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
import asyncio


# In[2]:


import nest_asyncio
nest_asyncio.apply()


# ## Task 2: Documents
# 
# We'll be concerning ourselves with this part of the flow in the following section:
# 
# <img src="https://i.imgur.com/jTm9gjk.png" />

# ### Loading Source Documents
# 
# So, first things first, we need some documents to work with.
# 
# While we could work directly with the `.txt` files (or whatever file-types you wanted to extend this to) we can instead do some batch processing of those documents at the beginning in order to store them in a more machine compatible format.
# 
# In this case, we're going to parse our text file into a single document in memory.
# 
# Let's look at the relevant bits of the `TextFileLoader` class:
# 
# ```python
# def load_file(self):
#         with open(self.path, "r", encoding=self.encoding) as f:
#             self.documents.append(f.read())
# ```
# 
# We're simply loading the document using the built in `open` method, and storing that output in our `self.documents` list.
# 
# > NOTE: We're using blogs from PMarca (Marc Andreessen) as our sample data. This data is largely irrelevant as we want to focus on the mechanisms of RAG, which includes out data's shape and quality - but not specifically what the contents of the data are. 
# 

# In[3]:


text_loader = TextFileLoader("data/PMarcaBlogs.txt")
documents = text_loader.load_documents()
len(documents)


# In[4]:


print(documents[0][:100])


# ### Splitting Text Into Chunks
# 
# As we can see, there is one massive document.
# 
# We'll want to chunk the document into smaller parts so it's easier to pass the most relevant snippets to the LLM.
# 
# There is no fixed way to split/chunk documents - and you'll need to rely on some intuition as well as knowing your data *very* well in order to build the most robust system.
# 
# For this toy example, we'll just split blindly on length.
# 
# >There's an opportunity to clear up some terminology here, for this course we will be stick to the following:
# >
# >- "source documents" : The `.txt`, `.pdf`, `.html`, ..., files that make up the files and information we start with in its raw format
# >- "document(s)" : single (or more) text object(s)
# >- "corpus" : the combination of all of our documents

# As you can imagine (though it's not specifically true in this toy example) the idea of splitting documents is to break them into managable sized chunks that retain the most relevant local context.

# In[5]:


text_splitter = CharacterTextSplitter()
split_documents = text_splitter.split_texts(documents)
len(split_documents)


# Let's take a look at some of the documents we've managed to split.

# In[6]:


split_documents[0:1]


# ## Task 3: Embeddings and Vectors
# 
# Next, we have to convert our corpus into a "machine readable" format as we explored in the Embedding Primer notebook.
# 
# Today, we're going to talk about the actual process of creating, and then storing, these embeddings, and how we can leverage that to intelligently add context to our queries.

# ### OpenAI API Key
# 
# In order to access OpenAI's APIs, we'll need to provide our OpenAI API Key!
# 
# You can work through the folder "OpenAI API Key Setup" for more information on this process if you don't already have an API Key!

# In[7]:


import os
import openai
# from getpass import getpass
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = getpass("OpenAI API Key: ")
# os.environ["OPENAI_API_KEY"] = openai.api_key


# ### Vector Database
# 
# Let's set up our vector database to hold all our documents and their embeddings!

# While this is all baked into 1 call - we can look at some of the code that powers this process to get a better understanding:
# 
# Let's look at our `VectorDatabase().__init__()`:
# 
# ```python
# def __init__(self, embedding_model: EmbeddingModel = None):
#         self.vectors = defaultdict(np.array)
#         self.embedding_model = embedding_model or EmbeddingModel()
# ```
# 
# As you can see - our vectors are merely stored as a dictionary of `np.array` objects.
# 
# Secondly, our `VectorDatabase()` has a default `EmbeddingModel()` which is a wrapper for OpenAI's `text-embedding-3-small` model.
# 
# > **Quick Info About `text-embedding-3-small`**:
# > - It has a context window of **8191** tokens
# > - It returns vectors with dimension **1536**

# #### ❓Question #1:
# 
# The default embedding dimension of `text-embedding-3-small` is 1536, as noted above. 
# 
# 1. Is there any way to modify this dimension?
# 2. What technique does OpenAI use to achieve this?
# 
# > NOTE: Check out this [API documentation](https://platform.openai.com/docs/api-reference/embeddings/create) for the answer to question #1, and [this documentation](https://platform.openai.com/docs/guides/embeddings/use-cases) for an answer to question #2!

# ##### ❗Answer #1❗
# The text-embedding-3-small one can modify the dimensions with the `dimensions` parameter but it only permits one smaller choise, 512.  
# 
# Parameter `dimensions` could be added in the following method of the EmbeddingModel():
# async def process_batch(batch):
#             embedding_response = await self.async_client.embeddings.create(
#                 input=batch, model=self.embeddings_model_name, dimensions = 512)
#                 aget_embeddings(list_of_text=list_of_text, engine=self.embeddings_model_name, dimensions=512)

# We can call the `async_get_embeddings` method of our `EmbeddingModel()` on a list of `str` and receive a list of `float` back!
# 
# ```python
# async def async_get_embeddings(self, list_of_text: List[str]) -> List[List[float]]:
#         return await aget_embeddings(
#             list_of_text=list_of_text, engine=self.embeddings_model_name
#         )
# ```

# We cast those to `np.array` when we build our `VectorDatabase()`:
# 
# ```python
# async def abuild_from_list(self, list_of_text: List[str]) -> "VectorDatabase":
#         embeddings = await self.embedding_model.async_get_embeddings(list_of_text)
#         for text, embedding in zip(list_of_text, embeddings):
#             self.insert(text, np.array(embedding))
#         return self
# ```
# 
# And that's all we need to do!

# In[20]:


vector_db = VectorDatabase()
vector_db = asyncio.run(vector_db.abuild_from_list(split_documents))


# 

# #### ❓Question #2:
# 
# What are the benefits of using an `async` approach to collecting our embeddings?
# 
# > NOTE: Determining the core difference between `async` and `sync` will be useful! If you get stuck - ask ChatGPT!
# #### ❗ANSWER #2❗
# `sync`: synchronous: tasks executed one at a time, in order. Next task is blocked until previous completes.  
# 
# `async`: asynchronous: tasks can run in background. Program does other tasks while waiting. Requires event loop (asyncio in Python). 
# Best for I/O-bound.   
# 
# **Benefits:**  
#  1. Faster 
#  2. Efficient: Uses CPU,GPU, and network more efficiently while waiting 
# 3. Scalable: Handles large text lists 
# 4. Keeps UI or other logic responsive  

# So, to review what we've done so far in natural language:
# 
# 1. We load source documents
# 2. We split those source documents into smaller chunks (documents)
# 3. We send each of those documents to the `text-embedding-3-small` OpenAI API endpoint
# 4. We store each of the text representations with the vector representations as keys/values in a dictionary

# ### Semantic Similarity
# 
# The next step is to be able to query our `VectorDatabase()` with a `str` and have it return to us vectors and text that is most relevant from our corpus.
# 
# We're going to use the following process to achieve this in our toy example:
# 
# 1. We need to embed our query with the same `EmbeddingModel()` as we used to construct our `VectorDatabase()`
# 2. We loop through every vector in our `VectorDatabase()` and use a distance measure to compare how related they are
# 3. We return a list of the top `k` closest vectors, with their text representations
# 
# There's some very heavy optimization that can be done at each of these steps - but let's just focus on the basic pattern in this notebook.
# 
# > We are using [cosine similarity](https://www.engati.com/glossary/cosine-similarity) as a distance metric in this example - but there are many many distance metrics you could use - like [these](https://flavien-vidal.medium.com/similarity-distances-for-natural-language-processing-16f63cd5ba55)
# 
# > We are using a rather inefficient way of calculating relative distance between the query vector and all other vectors - there are more advanced approaches that are much more efficient, like [ANN](https://towardsdatascience.com/comprehensive-guide-to-approximate-nearest-neighbors-algorithms-8b94f057d6b6)

# In[21]:


vector_db.search_by_text("What is the Michael Eisner Memorial Weak Executive Problem?", k=3)


# ## Task 4: Prompts
# 
# In the following section, we'll be looking at the role of prompts - and how they help us to guide our application in the right direction.
# 
# In this notebook, we're going to rely on the idea of "zero-shot in-context learning".
# 
# This is a lot of words to say: "We will ask it to perform our desired task in the prompt, and provide no examples."

# ### XYZRolePrompt
# 
# Before we do that, let's stop and think a bit about how OpenAI's chat models work.
# 
# We know they have roles - as is indicated in the following API [documentation](https://platform.openai.com/docs/api-reference/chat/create#chat/create-messages)
# 
# There are three roles, and they function as follows (taken directly from [OpenAI](https://platform.openai.com/docs/guides/gpt/chat-completions-api)):
# 
# - `{"role" : "system"}` : The system message helps set the behavior of the assistant. For example, you can modify the personality of the assistant or provide specific instructions about how it should behave throughout the conversation. However note that the system message is optional and the model’s behavior without a system message is likely to be similar to using a generic message such as "You are a helpful assistant."
# - `{"role" : "user"}` : The user messages provide requests or comments for the assistant to respond to.
# - `{"role" : "assistant"}` : Assistant messages store previous assistant responses, but can also be written by you to give examples of desired behavior.
# 
# The main idea is this:
# 
# 1. You start with a system message that outlines how the LLM should respond, what kind of behaviours you can expect from it, and more
# 2. Then, you can provide a few examples in the form of "assistant"/"user" pairs
# 3. Then, you prompt the model with the true "user" message.
# 
# In this example, we'll be forgoing the 2nd step for simplicities sake.

# #### Utility Functions
# 
# You'll notice that we're using some utility functions from the `aimakerspace` module - let's take a peek at these and see what they're doing!

# ##### XYZRolePrompt

# Here we have our `system`, `user`, and `assistant` role prompts.
# 
# Let's take a peek at what they look like:
# 
# ```python
# class BasePrompt:
#     def __init__(self, prompt):
#         """
#         Initializes the BasePrompt object with a prompt template.
# 
#         :param prompt: A string that can contain placeholders within curly braces
#         """
#         self.prompt = prompt
#         self._pattern = re.compile(r"\{([^}]+)\}")
# 
#     def format_prompt(self, **kwargs):
#         """
#         Formats the prompt string using the keyword arguments provided.
# 
#         :param kwargs: The values to substitute into the prompt string
#         :return: The formatted prompt string
#         """
#         matches = self._pattern.findall(self.prompt)
#         return self.prompt.format(**{match: kwargs.get(match, "") for match in matches})
# 
#     def get_input_variables(self):
#         """
#         Gets the list of input variable names from the prompt string.
# 
#         :return: List of input variable names
#         """
#         return self._pattern.findall(self.prompt)
# ```
# 
# Then we have our `RolePrompt` which laser focuses us on the role pattern found in most API endpoints for LLMs.
# 
# ```python
# class RolePrompt(BasePrompt):
#     def __init__(self, prompt, role: str):
#         """
#         Initializes the RolePrompt object with a prompt template and a role.
# 
#         :param prompt: A string that can contain placeholders within curly braces
#         :param role: The role for the message ('system', 'user', or 'assistant')
#         """
#         super().__init__(prompt)
#         self.role = role
# 
#     def create_message(self, **kwargs):
#         """
#         Creates a message dictionary with a role and a formatted message.
# 
#         :param kwargs: The values to substitute into the prompt string
#         :return: Dictionary containing the role and the formatted message
#         """
#         return {"role": self.role, "content": self.format_prompt(**kwargs)}
# ```
# 
# We'll look at how the `SystemRolePrompt` is constructed to get a better idea of how that extension works:
# 
# ```python
# class SystemRolePrompt(RolePrompt):
#     def __init__(self, prompt: str):
#         super().__init__(prompt, "system")
# ```
# 
# That pattern is repeated for our `UserRolePrompt` and our `AssistantRolePrompt` as well.

# ##### ChatOpenAI

# Next we have our model, which is converted to a format analagous to libraries like LangChain and LlamaIndex.
# 
# Let's take a peek at how that is constructed:
# 
# ```python
# class ChatOpenAI:
#     def __init__(self, model_name: str = "gpt-4o-mini"):
#         self.model_name = model_name
#         self.openai_api_key = os.getenv("OPENAI_API_KEY")
#         if self.openai_api_key is None:
#             raise ValueError("OPENAI_API_KEY is not set")
# 
#     def run(self, messages, text_only: bool = True):
#         if not isinstance(messages, list):
#             raise ValueError("messages must be a list")
# 
#         openai.api_key = self.openai_api_key
#         response = openai.ChatCompletion.create(
#             model=self.model_name, messages=messages
#         )
# 
#         if text_only:
#             return response.choices[0].message.content
# 
#         return response
# ```

# #### ❓ Question #3:
# 
# When calling the OpenAI API - are there any ways we can achieve more reproducible outputs?
# 
# > NOTE: Check out [this section](https://platform.openai.com/docs/guides/text-generation/) of the OpenAI documentation for the answer!

# ##### ! ANSWER #3 !
# 1. Specify the exact model, and use same unchanged model every time. i.e. a "static model snapshot"
# 2. Set the temperature to 0
# 3. Use seed parameter when supported   
# 4. Iterate on the prompt   
# Example:
# * Pin production applications to specific model snapshots (like gpt-4.1-2025-04-14 for example) to ensure consistent behavior.
# * Build evals that will measure the behavior of the prompts, so that the performance of the prompts can be monitored while iterating on them

# ### Creating and Prompting OpenAI's `gpt-4o-mini`!
# 
# Let's tie all these together and use it to prompt `gpt-4o-mini`!

# In[28]:


from aimakerspace.openai_utils.prompts import (
    UserRolePrompt,
    SystemRolePrompt,
    #AssistantRolePrompt, # not used
)

from aimakerspace.openai_utils.chatmodel import ChatOpenAI

chat_openai = ChatOpenAI()
user_prompt_template = "{content}"
user_role_prompt = UserRolePrompt(user_prompt_template)
system_prompt_template = (
    "You are an expert in {expertise}, you always answer in a kind way."
)
system_role_prompt = SystemRolePrompt(system_prompt_template)

messages = [
    system_role_prompt.create_message(expertise="Python"),
    user_role_prompt.create_message(
        content="What is the best way to write a loop?"
    ),
]

response = chat_openai.run(messages)


# Chris's output:   
# 
# The best way to write a loop in Python depends on the specific task you're trying to accomplish and the type of data you're working with. Here are some common patterns for writing loops, along with best practices:
# 
# ### 1. Using a `for` loop
# 
# The `for` loop is often preferred for iterating over sequences like lists, tuples, strings, and dictionaries. Here's a simple example:
# 
# ```python
# fruits = ["apple", "banana", "cherry"]
# 
# for fruit in fruits:
#     print(fruit)
# ```
# 
# **Best Practices:**
# - Use `for` loops when you know the number of iterations or when iterating over a collection.
# - Keep the loop's body simple for readability.
# 
# ### 2. Using a `while` loop
# 
# A `while` loop is useful when the number of iterations isn't predetermined. Here's an example:
# 
# ```python
# count = 0
# 

# In[23]:


print(response)


# ## Task 5: Retrieval Augmented Generation
# 
# Now we can create a RAG prompt - which will help our system behave in a way that makes sense!
# 
# There is much you could do here, many tweaks and improvements to be made!

# In[29]:


RAG_SYSTEM_TEMPLATE = """You are a knowledgeable assistant that answers questions based strictly on provided context.

Instructions:
- Only answer questions using information from the provided context
- If the context doesn't contain relevant information, respond with "I don't know"
- Be accurate and cite specific parts of the context when possible
- Keep responses {response_style} and {response_length}
- Only use the provided context. Do not use external knowledge.
- Only provide answers when you are confident the context supports your response."""

RAG_USER_TEMPLATE = """Context Information:
{context}

Number of relevant sources found: {context_count}
{similarity_scores}

Question: {user_query}

Please provide your answer based solely on the context above."""

rag_system_prompt = SystemRolePrompt(
    RAG_SYSTEM_TEMPLATE,
    strict=True,
    defaults={
        "response_style": "concise",
        "response_length": "brief"
    }
)

rag_user_prompt = UserRolePrompt(
    RAG_USER_TEMPLATE,
    strict=True,
    defaults={
        "context_count": "",
        "similarity_scores": ""
    }
)


# Now we can create our pipeline!

# In[31]:


class RetrievalAugmentedQAPipeline:
    def __init__(self, llm: ChatOpenAI, vector_db_retriever: VectorDatabase, 
                 response_style: str = "detailed", include_scores: bool = False) -> None:
        self.llm = llm
        self.vector_db_retriever = vector_db_retriever
        self.response_style = response_style
        self.include_scores = include_scores

    def run_pipeline(self, user_query: str, k: int = 4, **system_kwargs) -> dict:
        # Retrieve relevant contexts
        context_list = self.vector_db_retriever.search_by_text(user_query, k=k)
        
        context_prompt = ""
        similarity_scores = []
        
        for i, (context, score) in enumerate(context_list, 1):
            context_prompt += f"[Source {i}]: {context}\n\n"
            similarity_scores.append(f"Source {i}: {score:.3f}")
        
        # Create system message with parameters
        system_params = {
            "response_style": self.response_style,
            "response_length": system_kwargs.get("response_length", "detailed")
        }
        
        formatted_system_prompt = rag_system_prompt.create_message(**system_params)
        
        user_params = {
            "user_query": user_query,
            "context": context_prompt.strip(),
            "context_count": len(context_list),
            "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
        }
        
        formatted_user_prompt = rag_user_prompt.create_message(**user_params)

        return {
            "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]), 
            "context": context_list,
            "context_count": len(context_list),
            "similarity_scores": similarity_scores if self.include_scores else None,
            "prompts_used": {
                "system": formatted_system_prompt,
                "user": formatted_user_prompt
            }
        }


# In[32]:


rag_pipeline = RetrievalAugmentedQAPipeline(
    vector_db_retriever=vector_db,
    llm=chat_openai,
    response_style="detailed",
    include_scores=True
)

result = rag_pipeline.run_pipeline(
    "What is the 'Michael Eisner Memorial Weak Executive Problem'?",
    k=3,
    response_length="comprehensive", 
    include_warnings=True,
    confidence_required=True
)

print(f"Response: {result['response']}")
print(f"\nContext Count: {result['context_count']}")
print(f"Similarity Scores: {result['similarity_scores']}")


# #### ❓ Question #4:
# 
# What prompting strategies could you use to make the LLM have a more thoughtful, detailed response?
# 
# What is that strategy called?
# 
# > NOTE: You can look through ["Accessing GPT-3.5-turbo Like a Developer"](https://colab.research.google.com/drive/1mOzbgf4a2SP5qQj33ZxTz2a01-5eXqk2?usp=sharing) for an answer to this question if you get stuck!

# #### ⚡ ANSWER #4: 
# Depending on what our goals are, we can use several prompting strategies:
# 1. Chain of Thought (CoT) reasoning: Ask the model to "think step by step".
# 2. Self-ask: Tell the model to ask itself questions needed to solve the problem, answer each of them and finally give the answer.
# 3. Explain like I'm 5, or Analyze: Instruct the model to explain at certain level of detail, or simplify the answer.
# 4. Specify format: e.g. a detailed step-by-step breakdown, or bullet points, or sections, etc.
# 5. Ask for comarison analysis (e.g. pros/cons)
# 6. Tell the model to "act as"... e.g.: an expert in something, and explain its answer 
# 7. A combination of few-shot examples which *contain* a step-by-step reasoning

# ### 🏗️ Activity #1:
# 
# Enhance your RAG application in some way! 
# 
# Suggestions are: 
# 
# - Allow it to work with PDF files
# - Implement a new distance metric
# - Add metadata support to the vector database
# 
# While these are suggestions, you should feel free to make whatever augmentations you desire! 
# 
# > NOTE: These additions might require you to work within the `aimakerspace` library - that's expected!
# 
# > NOTE: If you're not sure where to start - ask Cursor (CMD/CTRL+L) to guide you through the changes!

# In[23]:


get_ipython().system(' pip install pdfplumber PyPDF2')


# In[28]:


import os
import openai
# from getpass import getpass
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# In[29]:


from aimakerspace.text_utils import PDFFileLoader, CharacterTextSplitter
from aimakerspace.vectordatabase import VectorDatabase
from aimakerspace.openai_utils.prompts import UserRolePrompt, SystemRolePrompt
from aimakerspace.openai_utils.chatmodel import ChatOpenAI


# In[43]:


import pdfplumber
import PyPDF2
import asyncio


# In[31]:


def load_pdf_documents(pdf_file: str, use_pdfplumber: bool = True):
    """
    Load documents from a PDF file using PDFFileLoader.

    Args:
        pdf_path: Path to the PDF file
        use_pdfplumber: Whether to use pdfplumber (better) or PyPDF2 (faster)

    Returns:
        List of document texts
    """
    print(f"Loading PDF from: {pdf_file}")

    try:
        pdf_loader = PDFFileLoader(pdf_file, use_pdfplumber=use_pdfplumber)
        documents = pdf_loader.load_documents()
        print(f"Successfully loaded {len(documents)} document(s) from PDF")
        return documents
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages: pip install pdfplumber PyPDF2")
        return []
    except FileNotFoundError:
        print(f"PDF file not found: {pdf_file}")
        return []
    except Exception as e:
        print(f"Error loading PDF: {e}")
        return []


# In[40]:


def documents_splitter(documents, chunk_size: int = 1000, chunk_overlap: int = 200): 
    """
    Split documents into smaller chunks for processing.

    Args:
        documents: List of document texts
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks

    Returns:
        List of text chunks
    """
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_documents = text_splitter.split_texts(documents)
    print(f"Split documents into {len(split_documents)} chunks")
    return split_documents


# In[33]:


async def build_vector_database(split_documents):
    """
    Build vector database from document chunks.

    Args:
        split_documents: List of text chunks

    Returns:
        VectorDatabase instance
    """
    print("Building vector database...")
    vector_db = VectorDatabase()
    vector_db = await vector_db.abuild_from_list(split_documents)
    print("Vector database built successfully")
    return vector_db


# In[34]:


def create_rag_pipeline(vector_db, llm):
    """
    Create a RAG pipeline for question answering.

    Args:
        vector_db: VectorDatabase instance
        llm: ChatOpenAI instance

    Returns:
        RetrievalAugmentedQAPipeline instance
    """
    # Define RAG prompts
    RAG_SYSTEM_TEMPLATE = """You are a knowledgeable assistant that answers questions based strictly on provided context.

    Instructions:
    - Only answer questions using information from the provided context
    - If the context doesn't contain relevant information, respond with "I don't know"
    - Be accurate and cite specific parts of the context when possible
    - Keep responses {response_style} and {response_length}
    - Only use the provided context. Do not use external knowledge.
    - Only provide answers when you are confident the context supports your response.
    """

    RAG_USER_TEMPLATE = """Context Information:
    {context}

    Number of relevant sources found: {context_count}
    {similarity_scores}

    Question: {user_query}

    Please provide your answer based solely on the context above."""

    rag_system_prompt = SystemRolePrompt(RAG_SYSTEM_TEMPLATE)
    rag_user_prompt = UserRolePrompt(RAG_USER_TEMPLATE)

    class RetrievalAugmentedQAPipeline:
        def __init__(self, llm, vector_db_retriever,
                     response_style: str = "detailed", include_scores: bool = False):
            self.llm = llm
            self.vector_db_retriever = vector_db_retriever
            self.response_style = response_style
            self.include_scores = include_scores

        def run_pipeline(self, user_query: str, k: int = 4, **system_kwargs):
            # Retrieve relevant contexts
            context_list = self.vector_db_retriever.search_by_text(user_query, k=k)
                                                                                         
            context_prompt = ""
            similarity_scores = []

            for i, (context, score) in enumerate(context_list, 1):
                context_prompt += f"[Source {i}]: {context}\n\n"
                similarity_scores.append(f"Source {i}: {score:.3f}")

            # Create system message with parameters
            system_params = {
                "response_style": self.response_style,
                "response_length": system_kwargs.get("response_length", "detailed")
            }

            formatted_system_prompt = rag_system_prompt.create_message(**system_params)

            user_params = {
                "user_query": user_query,
                "context": context_prompt.strip(),
                "context_count": len(context_list),
                "similarity_scores": f"Relevance scores: {', '.join(similarity_scores)}" if self.include_scores else ""
            }

            formatted_user_prompt = rag_user_prompt.create_message(**user_params)

            return {
                "response": self.llm.run([formatted_system_prompt, formatted_user_prompt]),
                "context": context_list,
                "context_count": len(context_list),
                "similarity_scores": similarity_scores if self.include_scores else None,
                "prompts_used": {
                    "system": formatted_system_prompt,
                    "user": formatted_user_prompt
                }
            }

    return RetrievalAugmentedQAPipeline(llm, vector_db)


# ##### RAG Pipeline for RAG on a PDF file    

# In[35]:


# 1. LOAD PDF DOCUMENT    
pdf_path = "/home/olb/AIE7-BC/AIM_AIE7/02_Embeddings_and_RAG/data/"
pdf_filename = "Apple_Illusion_of_thinking_2025.pdf"
documents = load_pdf_documents(pdf_path+pdf_filename)
if not documents:
    print("No documents loaded. Please check the PDF path and try again.")
else:
# Display sample of loaded content
    print(f"\nSample content (first 200 characters):")
    print(documents[0][:200] + ".../n")


# In[16]:


get_ipython().system('pwd')


# In[109]:


# 2. SPLIT DOCUMENTS INTO CHUNKS
split_documents = documents_splitter(documents)


# In[110]:


type(split_documents)


# In[111]:


import tracemalloc
tracemalloc.start()


# In[112]:


async def main():
    vector_db = await build_vector_database(split_documents)
    return vector_db

# Run the async function
vector_db = await main()


# In[117]:


get_ipython().system('pwd')


# In the following:      
# * Use `search_with_source_filter()` when:
# * You have multiple PDF files
# * You want to search only in one specific file
# * You know the exact file name
# * Use `search_by_tags()` when:
# * You want to search by topic/keyword
# * You have documents with different themes
# * You want to filter by content type

# In[119]:


# SIMPLE METADATA SUPPORT - GENERIC VERSION
import datetime
import os

class SimpleMetadataDB:
    def __init__(self, vector_db):
        self.vector_db = vector_db
        self.metadata = {}
    
    def add_basic_metadata(self, text, source_file=None, tags=None, chunk_index=None, file_dir="data"):
        """Add basic metadata to a document.
           text is each document chunk in the split_documents
           source_file is the name of the source file
           tags is a list of keywords to filter in the document
           chunk_index is the index of this chunk in the document
        """
        if text in self.vector_db.vectors:
            # Use defaults if not provided
            if source_file is None:
                source_file = "unknown_file"
            
            if tags is None:
                tags = ["general", "document", "text"]
            
            if chunk_index is None:
                chunk_index = 0
            
            # Get file info if possible
            try:
                # Use current working directory + data subfolder (generic)
                current_dir = os.getcwd()
                file_path = os.path.join(current_dir, file_dir, source_file)
                file_size = os.path.getsize(file_path) if os.path.exists(file_path) else 0
            except:
                file_path = os.path.join("data", source_file)  # fallback
                file_size = 0
            # Determine file type from extension
            file_type = "pdf"  # default
            if source_file:
                if source_file.lower().endswith('.pdf'):
                    file_type = "pdf"
                elif source_file.lower().endswith('.txt'):
                    file_type = "txt"
                elif source_file.lower().endswith('.docx'):
                    file_type = "docx"
                elif source_file.lower().endswith('.html'):
                    file_type = "html"
            
            self.metadata[text] = {
                "chunk_index": chunk_index,
                "source_file": source_file,
                "file_path": file_path,
                "file_size": file_size,
                "file_type": file_type,
                "file_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "file_keywords": tags,
                "file_language": "English",
                "file_encoding": "UTF-8",
                "file_creation_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "file_modification_date": datetime.datetime.now().strftime("%Y-%m-%d"),
                "file_last_modified": datetime.datetime.now().strftime("%Y-%m-%d")
            }     
            print(f"Added metadata for chunk {chunk_index} from: {source_file}")
        else:
            print(f"Document not found: {text[:50]}...")
    
    def add_metadata_to_all_documents(self, source_file, tags=None):
        """Add metadata to all documents in the vector database."""
        print(f"Adding metadata to all documents from: {source_file}")
        
        for i, text in enumerate(self.vector_db.vectors.keys()):
            self.add_basic_metadata(
                text=text,
                source_file=source_file,
                tags=tags,
                chunk_index=i
            )
        
        print(f"Added metadata to {len(self.vector_db.vectors)} documents")
    
    def search_with_source_filter(self, query, source_file=None):
        """Search and optionally filter by source file."""
        results = self.vector_db.search_by_text(query, k=10)
        
        filtered_results = []
        for text, score in results:
            metadata = self.metadata.get(text, {})
            
            # If source filter is specified, only include matching documents
            if source_file is None or metadata.get('source_file') == source_file:
                filtered_results.append((text, score, metadata))
        
        return filtered_results
    
    def search_by_tags(self, query, tags=None):
        """Search in documents tagged with tags and filter by tags."""
        results = self.vector_db.search_by_text(query, k=20)  # Get more results for filtering
        
        filtered_results = []
        for text, score in results:
            metadata = self.metadata.get(text, {})
            document_tags = metadata.get('file_keywords', [])
            
            # If tags filter is specified, check if any tag matches
            if tags is None or any(tag in document_tags for tag in tags):
                filtered_results.append((text, score, metadata))
        
        return filtered_results[:10]  # Return top 10
    
    def get_metadata_summary(self):
        """Get a summary of all metadata."""
        if not self.metadata:
            return {"total_documents": 0}
        
        summary = {
            "total_documents": len(self.metadata),
            "source_files": set(),
            "file_types": set(),
            "all_tags": set()
        }
        
        for metadata in self.metadata.values():
            if 'source_file' in metadata:
                summary["source_files"].add(metadata['source_file'])
            if 'file_type' in metadata:
                summary["file_types"].add(metadata['file_type'])
            if 'file_keywords' in metadata:
                summary["all_tags"].update(metadata['file_keywords'])
        
        # Convert sets to lists for easier handling
        summary["source_files"] = list(summary["source_files"])
        summary["file_types"] = list(summary["file_types"])
        summary["all_tags"] = list(summary["all_tags"])
        
        return summary


# In[121]:


# 1. Create the enhanced database
enhanced_db = SimpleMetadataDB(vector_db)

# 2. Add metadata to your documents
enhanced_db.add_metadata_to_all_documents(
    source_file=pdf_filename,
    tags=["Illusion of Thinking", "Large Reasoning Models", "LLMs", "RLMs", "Problem Complexity", "Reasoning Models", "AI", "Research"]
)


# In[122]:


# Test with more specific queries
test_queries = [
    "Large Reasoning Models",  # More specific
    "LLMs reasoning capabilities",  # More specific
    "thinking models",  # Different phrasing
    "reasoning limitations",  # Different aspect
    "problem complexity"  # Another topic from your tags
]

for query in test_queries:
    print(f"\n🔍 Query: '{query}'")
    results = enhanced_db.search_with_source_filter(
        query=query,
        source_file="Apple_Illusion_of_thinking_2025.pdf"
    )
    
    if results:
        best_score = results[0][1]  # Get the highest score
        print(f"   Best score: {best_score:.4f}")
        print(f"   Best match: {results[0][0][:80]}...")
    else:
        print("   No results found")


# In[123]:


# 4. Initialize LLM
chat_openai = ChatOpenAI()

# 5. CREATE RAG PIPELINE
rag_pipeline = create_rag_pipeline(vector_db, chat_openai)

print("Vector database built successfully!")
print("RAG pipeline created successfully!")


# In[ ]:


# 6. TEST RAG PIPELINE 
test_queries = [
    "What is the main topic of this document?",
    "What are the key points discussed?",
    "Can you summarize the content?"
]
for query in test_queries:
    print(f"\nQuestion: {query}")
    try:
        result = rag_pipeline.run_pipeline(
            query,
            k=3,
            include_scores=True,
            response_length="comprehensive"
        )    
    
        print(f"Response: {result['response']}")
        print(f"Context Count: {result['context_count']}")
        if result['similarity_scores']:
            print(f"Similarity Scores: {result['similarity_scores']}")
    except Exception as e:
        print(f"Error processing query: {e}")


# In[125]:


def cos_similarity(vector_db, test_query: str = "What are the limitations of reasoning models?"):
    """
    Quick test of cos similarity distance.
    """
    print(f"Quick cos similarity Distance Test")
    print(f"Query: '{test_query}'")
    print("-" * 50)
    
    try:
        # Use the existing search method which already handles embeddings internally
        # This avoids the async issues entirely
        search_results = vector_db.search_by_text(test_query, k=10)
        
        if not search_results:
            print("No results found. Please check your vector database.")
            return []
        
        print("Top 3 results from existing search method:")
        for i, (text, score) in enumerate(search_results[:3], 1):
            print(f"{i}. Similarity Score: {score:.4f}")
            print(f"   Text: {text[:100]}...")
            print()
            
        return search_results[:3]
        
    except Exception as e:
        print(f"Error: {e}")
        return [] 


# In[126]:


test_query = "What are the limitations of reasoning models?"
cos_similarity(vector_db, test_query = test_query)
search_results = vector_db.search_by_text(test_query, k=10)


# In[124]:


# 6. TEST RAG PIPELINE 
test_queries = [
    "What is the main topic of this document?",
    "What are the key points discussed?",
    "Can you summarize the content?"
]
for query in test_queries:
    print(f"\nQuestion: {query}")
    try:
        result = rag_pipeline.run_pipeline(
            query,
            k=3,
            include_scores=True,
            response_length="comprehensive"
        )    
    
        print(f"Response: {result['response']}")
        print(f"Context Count: {result['context_count']}")
        if result['similarity_scores']:
            print(f"Similarity Scores: {result['similarity_scores']}")
    except Exception as e:
        print(f"Error processing query: {e}")


# In[127]:


# CALCULATE METRICS BETWEEN EXISTING VECTORS
import numpy as np

def euclidean_distance(vec1, vec2):
    return np.sqrt(np.sum((vec1 - vec2) ** 2))

def manhattan_distance(vec1, vec2):
    return np.sum(np.abs(vec1 - vec2))

def dot_product_similarity(vec1, vec2):
    return np.dot(vec1, vec2)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2) if norm1 * norm2 != 0 else 0

def compare_metrics_between_vectors(vector_db, k=5):
    """Compare different metrics between existing vectors in the database."""
    
    print("�� COMPARING DISTANCE METRICS BETWEEN EXISTING VECTORS")
    print("=" * 60)
    
    # Get all vectors from the database
    all_texts = list(vector_db.vectors.keys())
    print(f"Found {len(all_texts)} vectors in database")
    
    if len(all_texts) < 2:
        print("Need at least 2 vectors to compare")
        return
    
    # Pick the first vector as reference
    reference_text = all_texts[0]
    reference_vector = np.array(vector_db.vectors[reference_text])
    
    print(f"Using as reference: {reference_text[:80]}...")
    print(f"Reference vector dimension: {len(reference_vector)}")
    print()
    
    # Calculate all metrics for all other vectors
    results = []
    
    for text in all_texts[1:]:  # Skip the reference vector
        vector = np.array(vector_db.vectors[text])
        
        # Calculate all metrics
        euclidean_dist = euclidean_distance(reference_vector, vector)
        manhattan_dist = manhattan_distance(reference_vector, vector)
        dot_prod = dot_product_similarity(reference_vector, vector)
        cosine_sim = cosine_similarity(reference_vector, vector)
        
        results.append({
            'text': text,
            'euclidean': euclidean_dist,
            'manhattan': manhattan_dist,
            'dot_product': dot_prod,
            'cosine': cosine_sim
        })
    
    # Sort by each metric
    print("📊 RANKING BY EACH METRIC:")
    print()
    
    # Euclidean (lower is better)
    euclidean_sorted = sorted(results, key=lambda x: x['euclidean'])
    print("�� EUCLIDEAN DISTANCE (lower = better):")
    for i, result in enumerate(euclidean_sorted[:k], 1):
        print(f"{i}. Distance: {result['euclidean']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
    print()
    
    # Manhattan (lower is better)
    manhattan_sorted = sorted(results, key=lambda x: x['manhattan'])
    print("🗺️ MANHATTAN DISTANCE (lower = better):")
    for i, result in enumerate(manhattan_sorted[:k], 1):
        print(f"{i}. Distance: {result['manhattan']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
    print()
    
    # Dot Product (higher is better)
    dot_sorted = sorted(results, key=lambda x: x['dot_product'], reverse=True)
    print("🔗 DOT PRODUCT SIMILARITY (higher = better):")
    for i, result in enumerate(dot_sorted[:k], 1):
        print(f"{i}. Score: {result['dot_product']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
    print()
    
    # Cosine (higher is better)
    cosine_sorted = sorted(results, key=lambda x: x['cosine'], reverse=True)
    print("📐 COSINE SIMILARITY (higher = better):")
    for i, result in enumerate(cosine_sorted[:k], 1):
        print(f"{i}. Score: {result['cosine']:.4f}")
        print(f"   Text: {result['text'][:80]}...")
    print()
    
    return {
        'euclidean': euclidean_sorted[:k],
        'manhattan': manhattan_sorted[:k],
        'dot_product': dot_sorted[:k],
        'cosine': cosine_sorted[:k]
    }

# Run the comparison
results = compare_metrics_between_vectors(vector_db, k=5)


# In[129]:


# COMPARE MULTIPLE VECTORS
def compare_multiple_vectors(vector_db, num_vectors=3):
    """Compare multiple vectors against each other."""
    
    all_texts = list(vector_db.vectors.keys())
    if len(all_texts) < num_vectors:
        num_vectors = len(all_texts)
    
    print(f"🔍 COMPARING {num_vectors} VECTORS")
    print("=" * 50)
    
    # Pick first few vectors
    selected_texts = all_texts[:num_vectors]
    
    for i, text1 in enumerate(selected_texts):
        vector1 = np.array(vector_db.vectors[text1])
        print(f"\n�� Reference {i+1}: {text1[:60]}...")
        
        for j, text2 in enumerate(selected_texts):
            if i != j:  # Don't compare with itself
                vector2 = np.array(vector_db.vectors[text2])
                
                euclidean = euclidean_distance(vector1, vector2)
                manhattan = manhattan_distance(vector1, vector2)
                dot_prod = dot_product_similarity(vector1, vector2)
                cosine = cosine_similarity(vector1, vector2)
                
                print(f"  vs {j+1}: Euclidean={euclidean:.4f}, Manhattan={manhattan:.4f}, Dot={dot_prod:.4f}, Cosine={cosine:.4f}")

# Run the comparison
compare_multiple_vectors(vector_db, num_vectors=3)


# ### ADD METADATA TO SPLITTED CHUNKS BEFORE CREATING `vector_db`

# In[ ]:


# import os
# import openai
# # from getpass import getpass
# from dotenv import load_dotenv

# load_dotenv()
# openai.api_key = os.getenv("OPENAI_API_KEY")
# # openai.api_key = getpass("OpenAI API Key: ")
# # os.environ["OPENAI_API_KEY"] = openai.api_key


# In[134]:


def add_metadata_to_existing_chunks(split_documents, source_file, tags=None, file_dir="data"):
    """
    Add metadata directly to each text chunk.
    """
    enhanced_chunks = []
    
    for i, chunk in enumerate(split_documents):
        metadata_info = f"""
        CHUNK {i+1}
        Date: {datetime.datetime.now().strftime("%Y-%m-%d")}
        Source: {source_file}
        Path: {os.path.join(file_dir, source_file)}
        Type: {"pdf" if source_file.lower().endswith('.pdf') else "txt"}
        Keywords: {', '.join(tags or ["general", "document"])}
        Date: {datetime.datetime.now().strftime("%Y-%m-%d")}
        Total_chunks: {len(split_documents)}
        Language: English
        Encoding: UTF-8
        """
            
        enhanced_text = metadata_info + chunk
        enhanced_chunks.append(enhanced_text)
    
    print(f"Added metadata to {len(enhanced_chunks)} existing chunks")
    return enhanced_chunks


# In[135]:


documents_with_metadata = add_metadata_to_existing_chunks(
    split_documents,
    source_file=pdf_filename,  # Just the filename
    tags=["Illusion of Thinking", "Large Reasoning Models", "LLMs", "RLMs", "Problem Complexity", "Reasoning Models", "AI", "Research"],
    file_dir="data"  # Specify directory here
)


# In[139]:


def fix_reversed_text(text):
    """
    Fix common text reversal issues.
    """
    # Fix common reversed patterns
    corrections = {
        "5202": "2025",
        "viXra": "arXiv",
        "nuJ": "Jun",  # or "Jul" depending on context
        "]IA.sc[": "cs.AI",  # Common arXiv category
        "1v14960.6052": "2025.06052v1"  # arXiv ID format
    }
    
    for wrong, correct in corrections.items():
        text = text.replace(wrong, correct)
    
    return text


# In[144]:


if documents_with_metadata:
    documents_with_metadata[0] = fix_reversed_text(documents_with_metadata[0])
    print("Fixed text sample:")
    print(documents_with_metadata[0][400:500])


# In[145]:


vector_db = await build_vector_database(documents_with_metadata)


# #### Repeat the RAG pipeline from above, and add the distance metrics to see if there is any difference in the responses and the distance metrics.

# In[146]:


# 4. Initialize LLM
chat_openai = ChatOpenAI()

# 5. CREATE RAG PIPELINE
rag_pipeline = create_rag_pipeline(vector_db, chat_openai)

print("Vector database built successfully!")
print("RAG pipeline created successfully!")

