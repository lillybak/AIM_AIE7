import asyncio
from langchain.chat_models import ChatOpenAI

# 2. SPLIT DOCUMENTS INTO CHUNKS
split_documents = split_documents(documents)

# 3. BUILD VECTOR DATABASE - FIXED: Use asyncio.run() for async function
vector_db = asyncio.run(build_vector_database(split_documents))

# 4. Initialize LLM
chat_openai = ChatOpenAI()

# 5. CREATE RAG PIPELINE
rag_pipeline = create_rag_pipeline(vector_db, chat_openai)

print("Vector database built successfully!")
print("RAG pipeline created successfully!") 