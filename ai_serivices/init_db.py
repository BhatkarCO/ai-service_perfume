import os
import re
import json
from dotenv import load_dotenv
# pyrefly: ignore [missing-import]
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
import certifi

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

def format_product_document(prod):
    # Extract fields
    prod_id = ""
    if "_id" in prod:
        if isinstance(prod["_id"], dict) and "$oid" in prod["_id"]:
            prod_id = prod["_id"]["$oid"]
        else:
            prod_id = str(prod["_id"])
            
    name = prod.get("name", "N/A")
    slug = prod.get("slug", "")
    desc = prod.get("description", "No description available.")
    short_desc = prod.get("short_description")
    price = prod.get("price", "N/A")
    sale_price = prod.get("sale_price")
    stock = prod.get("stock_quantity", 0)
    
    category = prod.get("category")
    if not category and "category_id" in prod:
        cat_id = prod["category_id"]
        if isinstance(cat_id, dict) and "$oid" in cat_id:
            category = cat_id["$oid"]
        else:
            category = str(cat_id)
    if not category:
        category = "Perfume"
        
    gender = prod.get("gender", "Unisex")
    rating = prod.get("rating", 0)
    
    # Handle fragrance_notes
    notes_str = ""
    notes = prod.get("fragrance_notes")
    if notes:
        if isinstance(notes, dict):
            top = ", ".join(notes.get("top", [])) if notes.get("top") else "N/A"
            heart = ", ".join(notes.get("heart", [])) if notes.get("heart") else "N/A"
            base = ", ".join(notes.get("base", [])) if notes.get("base") else "N/A"
            notes_str = f"Top notes: {top}. Heart notes: {heart}. Base notes: {base}."
        elif isinstance(notes, str):
            notes_str = notes
            
    # Handle images (only keep the first image)
    images = prod.get("images", [])
    image_url = ""
    if images:
        if isinstance(images, list):
            first_img = images[0]
            if isinstance(first_img, dict):
                image_url = first_img.get("image_url", "")
            else:
                image_url = str(first_img)
        elif isinstance(images, dict):
            image_url = images.get("image_url", "")
        elif isinstance(images, str):
            image_url = images
        
    # Redirection Link format
    redirection_link = f"#/product/{slug}" if slug else ""
    
    # Build content text for vector DB
    doc_text = f"Product ID: {prod_id}\n" \
               f"Product Name: {name}\n" \
               f"Slug: {slug}\n" \
               f"Description: {desc}\n"
    if short_desc:
        doc_text += f"Short Description: {short_desc}\n"
    doc_text += f"Price: Rs. {price}\n"
    if sale_price:
        doc_text += f"Sale Price: Rs. {sale_price}\n"
    doc_text += f"Stock Quantity: {stock}\n" \
                f"Category: {category}\n" \
                f"Gender: {gender}\n" \
                f"Rating: {rating}/5\n"
    if notes_str:
        doc_text += f"Fragrance Notes: {notes_str}\n"
    if image_url:
        doc_text += f"Image URL: {image_url}\n"
    if redirection_link:
        doc_text += f"Redirection Link: {redirection_link}\n"
        
    metadata = {
        "source": "database",
        "type": "product",
        "product_id": prod_id,
        "name": name,
        "slug": slug,
        "image_url": image_url,
        "redirection_link": redirection_link
    }
    
    return Document(page_content=doc_text, metadata=metadata)

def load_products_from_mongodb():
    mongo_uri = os.getenv("MONGO_URI")
    db_name = os.getenv("MONGO_DB_NAME", "test")
    if not mongo_uri:
        print("Warning: MONGO_URI not found in env.")
        return None
        
    try:
        print(f"Connecting to MongoDB database '{db_name}'...")
        # Use certifi for secure TLS connection in Python on Windows/Mac
        client = MongoClient(mongo_uri, tlsCAFile=certifi.where(), serverSelectionTimeoutMS=5000)
        # Test connection
        client.admin.command('ping')
        db = client[db_name]
        collection = db["products"]
        products = list(collection.find({}))
        print(f"Successfully retrieved {len(products)} products from MongoDB.")
        return products
    except Exception as e:
        print(f"Could not connect to MongoDB: {e}")
        return None

def load_products_from_fallback_file():
    file_path = os.path.join(os.path.dirname(__file__), "scraped_data", "product.md")
    if not os.path.exists(file_path):
        print(f"Error: Fallback file {file_path} not found.")
        return []
        
    print(f"Reading fallback product details from {file_path}...")
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    products = []
    content_len = len(content)
    i = 0
    while i < content_len:
        if content[i] == '{':
            # Find the matching closing brace
            brace_count = 1
            j = i + 1
            while j < content_len and brace_count > 0:
                if content[j] == '{':
                    brace_count += 1
                elif content[j] == '}':
                    brace_count -= 1
                j += 1
            if brace_count == 0:
                json_str = content[i:j]
                try:
                    prod = json.loads(json_str)
                    products.append(prod)
                except Exception as e:
                    # Simple regex fallback if json validation fails
                    print(f"JSON validation failed for block, falling back to regex: {e}")
                    name_match = re.search(r'"name":\s*"([^"]+)"', json_str)
                    slug_match = re.search(r'"slug":\s*"([^"]+)"', json_str)
                    desc_match = re.search(r'"description":\s*"([^"]+)"', json_str)
                    gender_match = re.search(r'"gender":\s*"([^"]+)"', json_str)
                    oid_match = re.search(r'"\$oid":\s*"([^"]+)"', json_str)
                    
                    prod = {
                        "_id": {"$oid": oid_match.group(1)} if oid_match else None,
                        "name": name_match.group(1) if name_match else "N/A",
                        "slug": slug_match.group(1) if slug_match else "",
                        "description": desc_match.group(1) if desc_match else "",
                        "gender": gender_match.group(1) if gender_match else "Unisex"
                    }
                    products.append(prod)
                i = j - 1
        i += 1
        
    print(f"Parsed {len(products)} products from product.md fallback file.")
    return products

def init_chroma_db():
    print("Loading website.md...")
    website_path = os.path.join(os.path.dirname(__file__), "scraped_data", "website.md")
    
    if not os.path.exists(website_path):
        print(f"Error: Could not find {website_path}")
        return
        
    # 1. Load and process website.md
    with open(website_path, "r", encoding="utf-8") as f:
        content = f.read()

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    website_splits = text_splitter.split_documents(md_header_splits)
    print(f"Split website.md into {len(website_splits)} chunks.")

    # 2. Retrieve products (try MongoDB first, fallback to product.md)
    products = load_products_from_mongodb()
    if products is None:
        products = load_products_from_fallback_file()
        
    product_docs = []
    for prod in products:
        doc = format_product_document(prod)
        product_docs.append(doc)
    print(f"Formatted {len(product_docs)} product documents for vector database.")

    # 3. Combine website chunks and product documents
    all_documents = []
    all_documents.extend(website_splits)
    all_documents.extend(product_docs)
    print(f"Total documents to embed: {len(all_documents)}")

    # 4. Generate Embeddings & Save to Chroma
    print("Initializing FastEmbed embeddings...")
    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    persist_directory = os.environ.get("CHROMA_PATH", os.path.join(os.path.dirname(__file__), "chroma_db"))
    print(f"Storing embeddings in ChromaDB at {persist_directory}...")
    
    # Remove existing db to prevent duplicate records
    import shutil
    if os.path.exists(persist_directory):
        print(f"Clearing existing Chroma DB at {persist_directory}...")
        try:
            shutil.rmtree(persist_directory)
        except Exception as err:
            print(f"Could not clear directory: {err}")

    vectorstore = Chroma.from_documents(
        documents=all_documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print("Successfully initialized Chroma DB with website and product data.")

if __name__ == "__main__":
    init_chroma_db()
