import os
import openai
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain import PromptTemplate
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import PromptLayerChatOpenAI
from langchain.chat_models import ChatOpenAI

from langchain.document_loaders import UnstructuredFileLoader
from unstructured.cleaners.core import clean_extra_whitespace

from langchain.document_loaders import PyMuPDFLoader

#from pdfminer.high_level import extract_text


from langchain.retrievers.self_query.base import SelfQueryRetriever

from langchain.chains.query_constructor.base import AttributeInfo


from langchain.chains.question_answering import load_qa_chain


from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env



openai.api_key = os.getenv("OPENAI_API_KEY")

## I. CVs Ingestion

## 1) CVs loading


docs=[]
for file in os.listdir("data"):
    if file.endswith('.pdf'):
        file_path='./data/'+ file
        loader=PyMuPDFLoader(file_path)
        #loader=UnstructuredFileLoader(file_path, post_processors=[clean_extra_whitespace])
        docs.extend(loader.load())

#print(len(docs))



# 2) Texts extraction and embeddings
          
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 600,   #1000
    chunk_overlap = 10,
    length_function=len
)

splits = text_splitter.split_documents(docs)


embedding = OpenAIEmbeddings()


persist_directory = 'docs/chroma/'

# 3) Storing of the embeddings

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)


EV = vectordb.get(include=["embeddings", "documents", "metadatas"])


## II. Query pipeline

# 1) Query embedding

print("\n")

query=input("Enter your query:\n\n")


embedded_query=embedding.embed_query(query)


# 2) Retrieval, Once the data is in the database, you still need to retrieve it

import numpy as np
from math import sqrt
import heapq

def squared_sum(X):
 
  return round(sqrt(sum([a*a for a in X])),4)


def cos_similarity(X,Y):
  """ return cosine similarity between two lists """
 
  numerator = sum(a*b for a,b in zip(X,Y))
  denominator = squared_sum(X)*squared_sum(Y)
  return round((numerator/float(denominator)),4)



L=[]
for j in range(len(splits)):
    L.append(cos_similarity(EV["embeddings"][j], embedded_query))
#print("\n")
#print("The relevant documents score with cosine similarity:\n")
#print(L)
A=heapq.nlargest(1,L)
#print(A)
#print("\n")
M=[L.index(i) for i in A]
#print("\n")
Text=[]
Meta=[]

for m in M:
    Text.append(EV["documents"][m])
    Meta.append(EV["metadatas"][m])



context=Text


## 3) Prompt engineering


template = """ You are an AI assistant who loves to help people!

Use the following pieces of context to answer the question at the end.

Take your time to read carrefuly the pieces in the context to answer the question.

Do provide answer out of the context pieces.

If you don't know the answer, just say that you don't know, don't try to make up an answer.

Keep the answer as concise as possible. Provide reasoning.

Always say "thanks for asking!" at the end of the answer.

{context}

Question: {query}

Answer the question in the language of the question

Helpful Answer:"""


# Request LLMs

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo", 
    temperature=0, 
    max_tokens=100)


prompt = PromptTemplate(template=template, input_variables=["context", "query"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

print("\n")
print("The answer using my own retriever to retrieve relevant chains and passing them to the template context:\n")
print(llm_chain.run({"context": context, "query": query}))
#print("\n")
#print(Meta)
print("\n")

## With load_qa_chain
print("The answer when using load_qa_chain:\n")
Docs=vectordb.similarity_search(query, k=3)
chain = load_qa_chain(llm, chain_type="stuff")
response=chain.run(input_documents=Docs, question=query)

print(response)


print("\n")
#print(template.format(context="context", query="query"))


