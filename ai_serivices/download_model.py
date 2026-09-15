from langchain_community.embeddings import FastEmbedEmbeddings

print("Downloading and caching embedding model...")
FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
print("Model cached successfully.")