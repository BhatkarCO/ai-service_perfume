# Perfume Chatbot Workflow Details

This document explains the architecture and the data flow for the Perfume Website Chatbot.

## System Architecture

The system consists of two main components:
1. **Node.js Express Backend**: Manages your products in MongoDB and handles standard API requests.
2. **Python FastAPI Service**: Manages AI tasks, specifically generating vector embeddings, storing them in Chroma DB, and querying the Google Gemini Chat LLM using LangChain.

## Workflow: Adding a New Product

1. **User Action**: An admin adds a new perfume product (via frontend or API) to the Node.js backend (`POST /products`).
2. **Database Save**: The Node.js backend successfully saves the product into the MongoDB database using Mongoose.
3. **Webhook Trigger**: Immediately after the save, the Node.js backend sends a background `POST` request (using `fetch`) to the Python AI service at `http://127.0.0.1:8000/embed_product`.
4. **Data Formatting**: The Python service receives the product data (Name, Description, Price, Notes, etc.) and formats it into a single clean text string.
5. **Vector Embedding**: The Python service uses a local HuggingFace embedding model (`sentence-transformers/all-MiniLM-L6-v2`) to convert this text string into mathematical vectors.
6. **Storage**: The vector is saved into the local Chroma DB instance. 
7. **Result**: The new product is instantly available to the Chatbot without requiring any manual re-indexing or restarting.

## Workflow: User Chatting with Bot

1. **User Query**: A customer sends a message to the Chatbot (e.g., "Do you have any vanilla perfumes under $50?").
2. **API Call**: The frontend sends this query to the Python AI service at `http://127.0.0.1:8000/chat`.
3. **Similarity Search**: LangChain converts the user's question into a vector and searches Chroma DB for the top 4 most relevant pieces of context (this could be from `website.md` or dynamically added products).
4. **LLM Generation**: LangChain packages the user's question, the retrieved context, and our strict system rules into a prompt. It sends this prompt to the Chat LLM (e.g., Google Gemini).
5. **Strict Rules Application**: 
   - If the user asks for code or API keys, the LLM politely refuses.
   - If the user just says "Hi", the LLM responds in 2 to 3 words.
6. **Response**: The LLM generates the answer, which is sent back to the customer.

## Initial Setup Workflow

- **Website Data**: The script `init_db.py` reads `scraped_data/website.md`, splits the markdown by headers, chunks the text, generates embeddings, and creates the initial `chroma_db` directory.
