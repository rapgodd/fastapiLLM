import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore

from pinecone import Pinecone


def fill():
    load_dotenv()

    loader = Docx2txtLoader('spring-framework123.docx')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
    )
    document = loader.load_and_split(text_splitter=text_splitter)
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')

    return PineconeVectorStore.from_documents(document, embedding, index_name='spring-framework')
