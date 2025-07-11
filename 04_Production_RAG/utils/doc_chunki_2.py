import fitz  # PyMuPDF
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# -------------------------------
# CONFIGURATION
# -------------------------------
pdf_path = "your_file.pdf"  # ← Replace with your PDF file path
openai.api_key = "YOUR_OPENAI_API_KEY"  # ← Replace with your OpenAI key

embedding_model = "text-embedding-3-small"  # Or another embedding model

# -------------------------------
# GET EMBEDDING FUNCTION
# -------------------------------
def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model=embedding_model
    )
    return np.array(response['data'][0]['embedding'])

# -------------------------------
# EXTRACT TEXT BLOCKS FROM PDF
# -------------------------------
doc = fitz.open(pdf_path)
blocks = []

for page in doc:
    for b in page.get_text("blocks"):
        text = b[4].strip()
        if text:
            blocks.append(text)

print(f"✅ Extracted {len(blocks)} blocks from PDF.")

# -------------------------------
# COMPUTE EMBEDDINGS
# -------------------------------
embeddings = np.array([get_embedding(block) for block in blocks])

# -------------------------------
# COMPUTE DISTANCES BETWEEN BLOCKS
# -------------------------------
distances = [cosine_distances([embeddings[i]], [embeddings[i+1]])[0][0] for i in range(len(embeddings)-1)]

# -------------------------------
# IDENTIFY BREAKPOINTS
# -------------------------------
threshold = np.percentile(distances, 90)  # Can change (e.g., 85, 95)
breakpoints = [i + 1 for i, dist in enumerate(distances) if dist > threshold]

print(f"✅ Threshold (90th percentile): {threshold:.4f}")
print("✅ Breakpoints at indices:", breakpoints)

# -------------------------------
# SPLIT INTO CHUNKS
# -------------------------------
chunks = []
start = 0
for bp in breakpoints:
    chunk_text = " ".join(blocks[start:bp])
    chunks.append(chunk_text)
    start = bp
chunks.append(" ".join(blocks[start:]))

print(f"✅ Final number of semantic chunks: {len(chunks)}")

# -------------------------------
# OPTIONAL: SHOW EXAMPLES
# -------------------------------
for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
    print(f"\n--- Chunk {i + 1} ---")
    print(chunk[:500])  # Print first 500 characters

# -------------------------------
# USE: Now pass `chunks` to your RAG pipeline
# -------------------------------

