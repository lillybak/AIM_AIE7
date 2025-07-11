# Let's put together the full example code that:
# 1. Loads a PDF with PyMuPDF (fitz)
# 2. Extracts text blocks
# 3. Embeds each block using OpenAI embeddings
# 4. Computes distances to find semantic shifts
# 5. Identifies breakpoints and splits document into semantically coherent chunks

import fitz  # PyMuPDF
import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_distances

# -------------------------------
# CONFIGURATION
# -------------------------------
pdf_path = "your_file.pdf"
openai.api_key = "YOUR_OPENAI_API_KEY"  # Replace with your key

embedding_model = "text-embedding-3-small"

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

print(f"Extracted {len(blocks)} blocks from PDF.")

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
threshold = np.percentile(distances, 90)  # Can adjust percentile
breakpoints = [i + 1 for i, dist in enumerate(distances) if dist > threshold]

print("Threshold distance (90th percentile):", threshold)
print("Breakpoints at block indices:", breakpoints)

# -------------------------------
# SPLIT DOCUMENT INTO CHUNKS
# -------------------------------
chunks = []
start = 0
for bp in breakpoints:
    chunk_text = " ".join(blocks[start:bp])
    chunks.append(chunk_text)
    start = bp
# Add last chunk
chunks.append(" ".join(blocks[start:]))

print(f"Final number of semantic chunks: {len(chunks)}")

# -------------------------------
# SHOW SAMPLE CHUNKS
# -------------------------------
for i, chunk in enumerate(chunks[:3]):  # Print first 3 chunks as example
    print(f"\n--- Chunk {i+1} ---")
    print(chunk[:500])  # Print first 500 characters only

# -------------------------------
# DONE!
# -------------------------------

# Save or pass `chunks` to your vector database (e.g., for RAG indexing).
