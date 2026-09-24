import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST_URL")

threshold = 0.60

# Load Embeddings
embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-large-en-v1.5",
    model_kwargs = {"device":"cpu"},
    encode_kwargs = {"normalize_embeddings": True}
)

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

# Scope Check Function
def is_finance_related(question: str) -> bool:
    """
    Embeds the question and check the similarity.
    Returns True if finance related, False if not.
    """
    
    question_vector = embeddings.embed_query(question)
    
    # Query annual reports namespace
    pdf_result = index.query(
        vector = question_vector,
        top_k = 3,
        namespace = "annual_reports",
        include_metadata = True
    )
    
    # Query glossary namespace
    text_result = index.query(
        vector = question_vector,
        top_k = 3,
        namespace = "glossary",
        include_metadata = False
    )
    
    # Get highest score from either namespace
    scores = []
    
    if pdf_result["matches"]:
        scores.extend([m["score"] for m in pdf_result["matches"]])

    if text_result["matches"]:
        scores.extend([m["score"] for m in text_result["matches"]])
        
    if not scores:
        return False
    
    best_score = max(scores)
    print(f"Scope check score: {best_score: .4f} - threshold: {threshold}")
    
    return best_score >= threshold

if __name__ == "__main__":
    print("Ask the question")
    print("Type 'exit' to quit.\n")
    
    while True:
        question = input("Your question: ").strip()
        if question.lower() == 'exit':
            break
        
        if not question:
            continue
        
        result = is_finance_related(question)
        status = "Finance related — will proceed to crew" if result else "Out of scope — will decline"
        print(f"{status}\n")