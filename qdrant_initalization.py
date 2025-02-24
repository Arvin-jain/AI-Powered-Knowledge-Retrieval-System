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
from langchain_openai import OpenAIEmbeddings
import asyncio

load_dotenv()

llm = ChatOpenAI(  
    model= "gpt-4o-mini",
    temperature= 0
)

embeddings = OpenAIEmbeddings(
  model= "text-embedding-3-large"
)

qdrant_client = QdrantClient(
    url="https://b0568da4-f965-4c35-b0d4-272da69b3fd1.us-east-1-0.aws.cloud.qdrant.io:6333", 
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIiwiZXhwIjoxNzQ2ODE1Mjc4fQ.PdWPFugLqE05sSTFIF7KAy6xGaynxNkWIR2f1krS9AE",
)

async def webLoader():
    # Add error handling and headers to avoid blocking
    webpageLoader = WebBaseLoader(
        "https://lilianweng.github.io/posts/2023-06-23-agent/",
    )
    try:
        docs =  webpageLoader.load()
        return docs
    except Exception as e:
        print(f"Error loading webpage: {e}")
        return None


async def splitter(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # corrected from chunkSize
        chunk_overlap=200  # corrected from chunkOverlap
    )
    # Split the documents
    splits = text_splitter.split_documents(documents)
    return splits
    
async def embeddings_function(split_documents):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )
    vectors = embeddings.embed_documents(split_documents)
    return vectors


vector_store = Qdrant(
    client=qdrant_client,  # Fetch URL from environment variable
    collection_name="langchainjs-testing",
    embeddings=embeddings
)


def sim_search(user_input):
    return vector_store.similarity_search(user_input)


async def main():
    webpageLoader_test = await webLoader()
    allSplits = await splitter(webpageLoader_test)
    vector_store.add_documents(allSplits)


if __name__ == "__main__":
    asyncio.run(main())

