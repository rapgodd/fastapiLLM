import os

from fastapi import FastAPI
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate


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
system_text = os.environ.get("SYSTEM_MESSAGE")
if "need_to_be_filled" == requiredOrNot:
    DATABASE = fill()
else:
    embedding = OpenAIEmbeddings(model='text-embedding-3-large')
    DATABASE = PineconeVectorStore(embedding=embedding, index_name='spring-framework')

dic = os.environ.get("DICTIONARY")


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/llm")
def get_llm_response(query: str = "nothing"):
    llm = ChatOpenAI(model='gpt-4o-mini')
    retriever = DATABASE.as_retriever(search_kwargs={'k': 3})

    print("\n retrieve start \n")
    print(retriever.invoke(query))
    print("\n end \n")
    prompt_obj = ChatPromptTemplate.from_template(system_text)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt_obj}
    )
    # ai_message = qa_chain.invoke({"query": query})

    word_return_prompt = ChatPromptTemplate.from_template(f"""
        Review the user’s question and modify it based on our dictionary.
        Your modification will be used for embedding similarity cosine search with official backend Spring framework document.
        If you determine that no modification is needed, return the original question as is.
        
        Dictionary : {dic}

        question : {{question}}
    """)
    dictionary_chain = word_return_prompt | llm | StrOutputParser()
    final_chain = {"query": dictionary_chain} | qa_chain
    llm_response = final_chain.invoke({"question": query})

    return llm_response
