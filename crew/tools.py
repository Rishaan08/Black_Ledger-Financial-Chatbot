import os
import sqlite3
from crewai.tools import tool
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_HOST = os.getenv("PINECONE_HOST_URL")
db_path = "Cleaned Financial Data/financial_data.db"
schema_path = "Cleaned Financial Data/schema_summary.txt"

# Load Embedding Model
embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# Connect to Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(host=PINECONE_HOST)

# Load Schema Summary
with open(schema_path, "r") as f:
    schema_summary = f.read()


# Tool 1: Vector Search
@tool("vector_search")
def vector_search(question: str) -> str:
    """
    Use this tool for questions about company REVENUE, PROFITS, EARNINGS,
    ANNUAL REPORTS, financial strategies, risks, and qualitative information.
    Use this for ANY question about Apple, Amazon, AMD, Intel, Meta, 
    Microsoft, NVIDIA or Tesla financials.
    Available years: 2021, 2022, 2023.
    Do NOT use for stock prices or crypto prices.
    """

    question_vector = embedding_model.embed_query(question)

    # Search both namespaces
    pdf_results = index.query(
        vector=question_vector,
        top_k=5,
        namespace="annual_report",
        include_metadata=True,
    )

    text_results = index.query(
        vector=question_vector, top_k=3, namespace="glossary", include_metadata=True
    )

    output = []

    # Process PDF results
    if pdf_results["matches"]:
        output.append("From Annual Reports")
        for match in pdf_results["matches"]:
            source = match["metadata"].get("source", "Unknown")
            page = match["metadata"].get("page", "N/A")
            text = match["metadata"].get("text", "")
            output.append(f"Source: {source} | Page: {page}")
            output.append(f"{text}\n")

    # Process Glossary results
    if text_results["matches"]:
        output.append("From Glossary")
        for match in text_results["matches"]:
            source = match["metadata"].get("source", "Unknown")
            text = match["metadata"].get("text", "")
            output.append(f"Source: {source}")
            output.append(f"{text}\n")

    if not output:
        return "No relevant information found in annual reports or glossary."

    return "\n".join(output)


# Tool 2: SQL Query
@tool("sql_query")
def sql_query(question: str) -> str:
    """
    Use this tool for questions about stock prices, cryptocurrency prices,
    trading volumes, ETF prices, mutual fund prices, and numerical market data.
    Data available: Indian NIFTY50 stocks, cryptocurrencies, ETFs, mutual funds, US stocks.
    Data cutoff: maximum 2021. Cannot answer questions about 2022, 2023, or 2024.
    Do NOT use this for company strategies, annual reports, or financial terminology.
    """

    from groq import Groq

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    sql_prompt = f"""You are an expert SQL writer. 
    
                Here is the database schema:
                {schema_summary}

                Write a single SQL query to answer this question: {question}

                Rules:
                - Return ONLY the SQL query, nothing else
                - No explanation, no markdown, no backticks
                - Use exact table and column names from the schema
                - If the question is about data after 2021, return: SELECT 'Data not available after 2021' AS               message
                - If no relevant table exists, return: SELECT 'No relevant data found' AS message
                - Keep the query simple and safe, no DELETE or UPDATE
            """

    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": sql_prompt}],
        temperature=0.6,
    )

    sql = response.choices[0].message.content.strip()
    print(f"Generated SQL: {sql}")

    # Execute SQL against SQLite
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchmany(20)
        columns = [description[0] for description in cursor.description]

        if not rows:
            return "No data found for the given query."

        result_lines = [", ".join(columns)]
        for row in rows:
            result_lines.append(", ".join(str(v) for v in row))

        return "\n".join(result_lines)

    except Exception as e:
        return f"SQL execution error: {str(e)}. Generated SQL was: {sql}"
