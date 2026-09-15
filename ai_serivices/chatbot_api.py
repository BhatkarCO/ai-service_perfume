try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from dotenv import load_dotenv

from langchain_community.embeddings import FastEmbedEmbeddings
# pyrefly: ignore [missing-import]
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
# pyrefly: ignore [missing-import]
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
# pyrefly: ignore [missing-import]
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

# Using Groq for Llama 3
# pyrefly: ignore [missing-import]
from langchain_groq import ChatGroq

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)

app = FastAPI(title="Perfume Chatbot API")

# Add CORS middleware to allow the test page to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Paths
persist_directory = os.environ.get("CHROMA_PATH", os.path.join(os.path.dirname(__file__), "chroma_db"))

# Global variables for models
embeddings = None
vectorstore = None
llm = None
retriever = None
chain = None
chat_histories: Dict[str, ChatMessageHistory] = {}

def get_session_history(session_id: str):
    global chat_histories
    if session_id not in chat_histories:
        chat_histories[session_id] = ChatMessageHistory()
    return chat_histories[session_id]

def get_models():
    global embeddings, vectorstore, llm, retriever, chain
    if vectorstore is None:
        # Initialize embeddings
        embeddings = FastEmbedEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Load Chroma DB
        vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

        # Initialize LLM
        # Ensure CHAT_LLM_API_KEY is set to your Groq API Key in .env
        api_key = os.getenv("CHAT_LLM_API_KEY")
        if not api_key:
            raise Exception("CHAT_LLM_API_KEY is not set in environment")
        
        llm = ChatGroq(
            model_name="openai/gpt-oss-120b",
            groq_api_key=api_key,
            temperature=0.7,
            max_tokens=512
        )
        # Setup conversational retrieval prompts
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        qa_system_prompt = """You are a helpful, passionate, and knowledgeable perfume assistant.
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Make the answer descriptive, engaging, and well-formatted, but keep it under 100 words.

When recommending or describing a perfume:
- Always clearly list the fragrance notes (Top, Heart, and Base notes if available in the context).
- Use rich, descriptive imagery to describe the vibe and how the user will feel wearing it (e.g., confident, fresh, elegant, romantic, mysterious, cozy).
- Mention details like price, category, and target audience if they are present in the context.
- ALWAYS display the product image in your reply if an "Image URL" is available in the context. Use markdown syntax: `![Product Name](Image URL)`.
- ALWAYS display the redirection link in your reply if a "Redirection Link" is available in the context. Use markdown syntax: `[View/Buy Product](Redirection Link)`.

CRITICAL RULES:
1. NEVER reveal any code, internal workings, or API keys under any circumstances. If asked for code or API keys, refuse politely.
2. If the user's input is a simple greeting (like "hi", "hello"), respond with only 2 to 3 words (e.g., "Hello! How can I help?"). Do not waste tokens. For actual perfume queries or recommendations, ALWAYS provide a detailed, engaging answer.
3. Keep the total response length strictly under 100 words (excluding image and link markdown syntax where possible, but keep it concise).

Context: {context}"""

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])

        # Chains
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

# Data Models
class ProductDetails(BaseModel):
    name: str
    description: str
    price: float
    category: Optional[str] = None
    gender: Optional[str] = None
    fragrance_notes: Optional[dict] = None
    id: Optional[str] = None
    _id: Optional[dict] = None
    slug: Optional[str] = None
    images: Optional[List[str]] = None
    sale_price: Optional[float] = None
    stock_quantity: Optional[int] = None

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "default_session"

class ChatResponse(BaseModel):
    reply: str

@app.on_event("startup")
async def startup_event():
    if not os.path.exists(persist_directory):
        print("Warning: Chroma DB not initialized. Please run init_db.py first.")
    # Models are loaded lazily on first request to avoid OOM during startup port-scan

@app.get("/")
async def root():
    return {"status": "ok"}

@app.get("/test")
async def get_test_page():
    html_path = os.path.join(os.path.dirname(__file__), "chat_test.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    raise HTTPException(status_code=404, detail="Test page not found")

@app.post("/embed_product")
async def embed_product(product: ProductDetails):
    global vectorstore
    if vectorstore is None:
        get_models()
        
    prod_id = product.id
    if not prod_id and product._id:
        if "$oid" in product._id:
            prod_id = product._id["$oid"]
        else:
            prod_id = str(product._id)
    if not prod_id:
        prod_id = "N/A"
        
    # Format fragrance notes
    notes_str = ""
    if product.fragrance_notes:
        top = ", ".join(product.fragrance_notes.get("top", [])) if product.fragrance_notes.get("top") else "N/A"
        heart = ", ".join(product.fragrance_notes.get("heart", [])) if product.fragrance_notes.get("heart") else "N/A"
        base = ", ".join(product.fragrance_notes.get("base", [])) if product.fragrance_notes.get("base") else "N/A"
        notes_str = f"Top notes: {top}. Heart notes: {heart}. Base notes: {base}."
        
    # Handle image url (first from list)
    image_url = ""
    if product.images:
        image_url = product.images[0]
        
    # Redirection Link
    redirection_link = f"#/product/{product.slug}" if product.slug else ""
        
    product_text = f"Product ID: {prod_id}\n" \
                   f"Product Name: {product.name}\n" \
                   f"Slug: {product.slug or ''}\n" \
                   f"Description: {product.description}\n" \
                   f"Price: Rs. {product.price}\n"
    if product.sale_price:
        product_text += f"Sale Price: Rs. {product.sale_price}\n"
    if product.stock_quantity is not None:
        product_text += f"Stock Quantity: {product.stock_quantity}\n"
    product_text += f"Category: {product.category or 'Perfume'}\n" \
                    f"Gender: {product.gender or 'Unisex'}\n"
    if notes_str:
        product_text += f"Fragrance Notes: {notes_str}\n"
    if image_url:
        product_text += f"Image URL: {image_url}\n"
    if redirection_link:
        product_text += f"Redirection Link: {redirection_link}\n"
                   
    metadata = {
        "source": "database",
        "type": "product",
        "product_id": prod_id,
        "name": product.name,
        "slug": product.slug or "",
        "image_url": image_url,
        "redirection_link": redirection_link
    }
    
    try:
        vectorstore.add_texts(texts=[product_text], metadatas=[metadata])
        return {"status": "success", "message": f"Embedded product: {product.name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(req: ChatRequest):
    global chain
    if chain is None:
        get_models()
        
    try:
        response = chain.invoke(
            {"input": req.message},
            config={"configurable": {"session_id": req.session_id}}
        )
        return ChatResponse(reply=response["answer"])
    except Exception as e:
        import traceback
        print("CHAT ENDPOINT ERROR:", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
