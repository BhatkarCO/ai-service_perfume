import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader, TextLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

hf_token = os.getenv("HUGGINGFACE_API_KEY")

def init_chroma_db():
    print("Loading website.md...")
    file_path = os.path.join(os.path.dirname(__file__), "scraped_data", "website.md")
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}")
        return

    # Load markdown document
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Split markdown by headers for better semantic chunks
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(content)

    # Further split to ensure chunks aren't too large
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(md_header_splits)
    
    print(f"Split document into {len(splits)} chunks.")

    print("Initializing HuggingFace embeddings...")
    # Initialize the embedding model
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )

    print("Storing embeddings in ChromaDB...")
    # Save to local chroma_db directory or Render Disk
    persist_directory = os.environ.get("CHROMA_PATH", os.path.join(os.path.dirname(__file__), "chroma_db"))
    
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Successfully initialized Chroma DB at {persist_directory} with website.md data.")

if __name__ == "__main__":
    init_chroma_db()
