from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# === 1️⃣ Load and split PDF ===
loader = PyPDFLoader("your_file.pdf")
pages = loader.load()

# Further split large pages into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)
docs = text_splitter.split_documents(pages)

# === 2️⃣ Add metadata to each chunk ===
for i, doc in enumerate(docs):
    doc.metadata = {
        "page_number": doc.metadata.get("page", i + 1),
        "source": "your_file.pdf",
        "document_type": "Research report",
        "section": "Unknown"  # You can set more specific sections using regex or other logic
    }

# === 3️⃣ Embed and create vector store ===
embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(docs, embeddings)

# === 4️⃣ Example: Save or query with metadata filter ===
# You can save to disk if needed: vectorstore.save_local("faiss_index")

# Example retriever with metadata filtering
retriever = vectorstore.as_retriever(search_kwargs={
    "filter": {"document_type": "Research report"}
})

# === 5️⃣ Example query ===
query = "Summarize the results section"
retrieved_docs = retriever.get_relevant_documents(query)

for doc in retrieved_docs:
    print(f"Page: {doc.metadata['page_number']}, Section: {doc.metadata['section']}")
    print(doc.page_content)
    print("\n---\n")

