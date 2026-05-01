from dotenv import load_dotenv
import os
from pinecone import Pinecone
from pinecone import ServerlessSpec
from src.helper import load_pdf,filter_to_minimal_docs,text_split,download_embeddings

load_dotenv()

#Getting API keys from environment
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")
GROQ_API_KEY=os.getenv("GROQ_API_KEY")

#Setting environment variables in Python
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

extracted_data=load_pdf("data/")
minimal_docs=filter_to_minimal_docs(extracted_data)
text_chunks=text_split(minimal_docs)
embedding=download_embeddings()

pinecone_api_key=PINECONE_API_KEY
pc=Pinecone(api_key=pinecone_api_key) #Initialize Pinecone client,Create a connection to Pinecone

index_name="medical-chatbot" #vector database name

#create a vector database 
if not pc.has_index(index_name):
   pc.create_index(
      name=index_name,
      dimension=384, #dimension of the embeddings
      metric="cosine", #cosine similarity
      spec=ServerlessSpec(cloud="aws",region="us-east-1")
   )

#connect to created index
index =pc.Index(index_name)

#Load Existing index 

from langchain_pinecone import PineconeVectorStore
#Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch=PineconeVectorStore.from_existing_index(
   index_name=index_name,
   embedding=embedding
)