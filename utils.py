# utils
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

def load_vectorstore(vectorstore_path,index_name):
    vectorstore = FAISS.load_local(folder_path = vectorstore_path,index_name = index_name, embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") , allow_dangerous_deserialization=True)
    return vectorstore

vectorstore = load_vectorstore(vectorstore_path="shopify_langchain_testing_vectorstore",index_name="products")
