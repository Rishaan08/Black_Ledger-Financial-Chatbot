from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader

def load_documents(pdf_path='Datasets/Financial Reports', txt_path='Datasets/Finance Terms'):
    
    #Load PDFs
    pdf_loader = DirectoryLoader(
        path=pdf_path,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True     
   )
    
    #Load Text Files
    text_loader = DirectoryLoader(
        path=txt_path,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True
    )
    
    print("Loading PDFs...")
    pdf_docs = pdf_loader.load()
    for doc in pdf_docs:
        doc.metadata["type"] = "annual_report"
    
    print("Loading text files...")
    text_docs = text_loader.load()
    for doc in text_docs:
        doc.metadata["type"] = "glossary"
    
    print(f"Loaded {len(pdf_docs)} PDF pages and {len(text_docs)} text files")
    
    return pdf_docs, text_docs

if __name__ == "__main__":
    pdf_docs, text_docs = load_documents()
    
    # Check first PDF
    print("First PDF document")
    print(pdf_docs[0].page_content[:300])
    print(pdf_docs[0].metadata)
    
    # Check first txt
    print("\nFirst text file")
    print(text_docs[0].page_content[:300])
    print(text_docs[0].metadata)