from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio

load_dotenv()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

llm = ChatOpenAI(  
    model= "gpt-4o-mini",
    temperature= 0
)
 
qdrant_client = QdrantClient(
    url="https://b0568da4-f965-4c35-b0d4-272da69b3fd1.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2ODE1Mjc4fQ.PdWPFugLqE05sSTFIF7KAy6xGaynxNkWIR2f1krS9AE",
)

vector_store = Qdrant(
    client=qdrant_client,  
    collection_name="langchainjs-testing",
    embeddings=embeddings
)

def sim_search(user_input):
    results = vector_store.similarity_search(user_input, k=1)
    for res in results:
        return(f"*{res.page_content}")

print(sim_search("What role does memory play in improving the adaptability and long-term decision-making capabilities of AI agents?"))