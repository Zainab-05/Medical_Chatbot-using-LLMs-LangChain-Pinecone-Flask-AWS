from langchain_community.document_loaders import PyPDFLoader,DirectoryLoader #to load my book
from langchain.text_splitter import RecursiveCharacterTextSplitter #to split text into chunks
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

def load_pdf(data):
   loader=DirectoryLoader(
      data,
      glob="*.pdf", #to load only pdf file formats , loads all pdf files
      loader_cls=PyPDFLoader 
   )
   documents=loader.load()
   return documents

def filter_to_minimal_docs(docs: List[Document])-> List[Document]:
   """
   Given a list of document objects, return a new list of documents objects
   containing only 'source' in metadata and the original page_content
   """
   minimal_docs : List[Document]=[]
   for doc in docs:
      src=doc.metadata.get("source")
      minimal_docs.append(
         Document(
            page_content=doc.page_content,
            metadata={"source":src}
         )
      )
   return minimal_docs

#split the documents into smaller chunks
def text_split(minimal_docs):
   text_splitter=RecursiveCharacterTextSplitter(
      chunk_size=500,       # max size of each chunk
      chunk_overlap=20,    # repeated part between chunks
      length_function=len   # how size is measured
   )
   text_chunks=text_splitter.split_documents(minimal_docs)
   return text_chunks

def download_embeddings():
   """
   Download and return the HuggingFace embeddings model
   """
   model_name = "sentence-transformers/all-MiniLM-L6-v2"
   embeddings = HuggingFaceEmbeddings(
      model_name = model_name
   )
   return embeddings