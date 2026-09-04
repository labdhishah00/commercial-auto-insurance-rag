#---Phase: vector store---#
from chunking import final_chunks
from embeddings import embedding_model
from langchain_community.vectorstores import FAISS

#create vector store

vector_store = FAISS.from_documents(documents = final_chunks, embedding = embedding_model)

#verify vector store creation

print("Documents added to vector store:", len(final_chunks))