from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import asyncio

qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),  # Fetch URL from environment variable
    api_key=os.getenv("QDRANT_API_KEY"),  # Fetch API key from environment variable
)
# Function to Create a New Collection
def create_collection():
    qdrant_client.create_collection(
        collection_name="langchainjs-testing",
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE)  # Adjust vector size as needed
    )
    print("Collection created successfully!")


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


async def main():
    create_collection()

if __name__ == "__main__":
    asyncio.run(main())