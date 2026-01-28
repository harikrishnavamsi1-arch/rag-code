import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader
)

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

load_dotenv()

# ENV
AZURE_CONN = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
CONTAINER = os.getenv("AZURE_CONTAINER_NAME")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX")

TEMP_DIR = "temp_docs"
os.makedirs(TEMP_DIR, exist_ok=True)

# CONNECT AZURE
blob_service = BlobServiceClient.from_connection_string(AZURE_CONN)
container_client = blob_service.get_container_client(CONTAINER)

documents = []

print("✅ Connected to Azure Blob Storage")

# LOAD FILES
for blob in container_client.list_blobs():

    file_name = blob.name
    file_path = os.path.join(TEMP_DIR, file_name)

    with open(file_path, "wb") as f:
        f.write(container_client.download_blob(blob.name).readall())

    print(f"⬇️ Downloaded: {file_name}")

    if file_name.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_name.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    elif file_name.lower().endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        continue

    docs = loader.load()

    for doc in docs:
        doc.metadata["source_file"] = file_name
        doc.metadata["page"] = doc.metadata.get("page", "N/A")

    documents.extend(docs)

print(f"📄 Loaded pages: {len(documents)}")

# CHUNKING
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"✂️ Chunks created: {len(chunks)}")

# EMBEDDINGS
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

# PINECONE
pc = Pinecone(api_key=PINECONE_API_KEY)

PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=INDEX_NAME
)

print("✅ Documents indexed into Pinecone")
