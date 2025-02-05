from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import os
from langchain_openai import ChatOpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
loader = Docx2txtLoader('./spring-framework123.docx')
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=200,
)
document = loader.load_and_split(text_splitter=text_splitter)
embedding = OpenAIEmbeddings(model='text-embedding-3-large')
Chroma.from_documents(documents=document, embedding=embedding)

app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/llm")
def get_llm_response(prompt: str = "안녕하세용!"):


    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)



    response = llm.invoke(prompt)
    return {"prompt": prompt, "response": response}