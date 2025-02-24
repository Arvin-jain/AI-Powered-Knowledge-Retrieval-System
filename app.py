from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain.vectorstores import Qdrant
from langchain import hub
from langchain.schema.runnable import (
    RunnableBranch,
    RunnableLambda,
    RunnableMap,       ## Wrap an implicit "dictionary" runnable
    RunnablePassthrough,
)
from langchain.schema.runnable.passthrough import RunnableAssign
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
    url=os.getenv("QDRANT_URL"),  # Fetch URL from environment variable
    api_key=os.getenv("QDRANT_API_KEY"),  # Fetch API key from environment variable
)

vector_store = Qdrant(
    client=qdrant_client,  
    collection_name="langchainjs-testing",
    embeddings=embeddings
)

def sim_search(user_input):
    results = vector_store.similarity_search(user_input, k=2)
    for res in results:
        return(f"*{res.page_content}")

prompt = hub.pull("rlm/rag-prompt")

chain = RunnableAssign({'context': lambda x: sim_search(x['question'])}) | prompt | llm | StrOutputParser()

input_data = {'question': 'How do hierarchical planning and recursive task decomposition contribute to the efficiency of AI agents in complex decision-making environments?'}

input_test = 'How do hierarchical planning and recursive task decomposition contribute to the efficiency of AI agents in complex decision-making environments?'

async def main():
    print(chain.invoke(input_data))


if __name__ == "__main__":
    asyncio.run(main())

