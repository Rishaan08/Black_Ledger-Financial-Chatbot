import os
import uuid
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from document_loader import load_documents

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST_URL")

# Load Documents 
print("Loading Documents....")
pdf_docs, text_docs = load_documents()

# Chunking
print("Starting Chunking....")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=200
)

pdf_chunks = splitter.split_documents(pdf_docs)
text_chunks = splitter.split_documents(text_docs)

print(f"PDF chunks: {len(pdf_chunks)}")
print(f"Glossary: {len(text_chunks)}")
print(f"Total Chunks: {len(pdf_chunks) + len(text_chunks)}")

# Load Embedding Model
print("\nLoading embedding model....")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device":"cpu"},
    encode_kwargs={"normalize_embeddings":True}
)

print("Embedding model loaded.")

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

print("Connected to Pinecone.")

# Embed and upsert function
def upsert_chunks(chunks, namespace):
    """Embed chunks in batches and upsert to Pinecone under given namespace."""
    
    print(f"Upserting {len(chunks)} chunks into namespace: '{namespace}'")
    
    batch = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        metadata = chunk.metadata
        metadata['text'] = text
        
        vector = embeddings.embed_query(text)
        
        record = {
            "id": str(uuid.uuid4()),
            "values": vector,
            "metadata": metadata
        }
        
        batch.append(record)
        
        if len(batch) == 100:
            index.upsert(vectors=batch, namespace=namespace)
            print(f"Upserted chunks {i - 100 + 2} to {i + 1}")
            batch = []
            
    if batch:
        index.upsert(vectors=batch, namespace=namespace)
        print(f"Upserted final {len(batch)} chunks")
    
    print(f"Done upserting namespace '{namespace}'")
    
upsert_chunks(pdf_chunks, namespace="annual_report")
upsert_chunks(text_chunks, namespace="glossary")

stats = index.describe_index_stats()
print("\nPinecone index stats:")
print(f"Total vectors: {stats['total_vector_count']}")
print(f"Namespaces: {stats['namespaces']}")

print("\nAll documents embedded and stored in Pinecone.")