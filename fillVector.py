import os
import pinecone

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore


def fill():
    load_dotenv()

    loader = Docx2txtLoader('spring-framework123.docx')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    documents = loader.load_and_split(text_splitter=text_splitter)

    tagged_documents = []
    for doc in documents:
        doc.metadata["tag"] = "SPRING"
        tagged_documents.append(doc)

    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    return PineconeVectorStore.from_documents(tagged_documents, embedding, index_name='vector-store')
