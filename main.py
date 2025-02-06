import os

from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from fillVector import fill

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=pinecone_api_key)
DATABASE = None
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
app = FastAPI()
prompt = hub.pull("rlm/rag-prompt")
requiredOrNot = os.environ.get("IF_NEED_TO_BE_FILLED")

if "need_to_be_filled" == requiredOrNot:
    DATABASE = fill()
else:
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    DATABASE = PineconeVectorStore(embedding=embedding, index_name='spring-framework')


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/llm")
def get_llm_response(query: str = "nothing"):
    llm = ChatOpenAI(model='gpt-4o-mini')
    retriever = DATABASE.as_retriever(search_kwargs={'k': 2})

    print("\n retrieve start \n")
    print(retriever.invoke(query))
    print("\n end \n")

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt}
    )
    ai_message = qa_chain.invoke({"query": query})

    return ai_message
