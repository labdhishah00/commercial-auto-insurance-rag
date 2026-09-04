# Phase: Vector store for chunking strategy 2

from chunking_strategy2 import final_strategy2_chunks
from embeddings import embedding_model 
from langchain_community.vectorstores import FAISS

#Build vector store

vector_store_strategy2 = FAISS.from_documents(documents = final_strategy2_chunks, embedding = embedding_model)

#sanity check

print("Documents added to strategy 2 vector store:", len(final_strategy2_chunks))