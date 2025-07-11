import base64
from io import BytesIO
from PIL import Image
import os

# Create directory for images if needed
images_dir = "/mnt/data/rag_images"
os.makedirs(images_dir, exist_ok=True)

image_count = 0
image_files = []

# Extract images from notebook code cell outputs
for i, cell in enumerate(cells):
    if cell.cell_type == "code" and "outputs" in cell:
        for output in cell["outputs"]:
            if "data" in output and "image/png" in output["data"]:
                img_data = output["data"]["image/png"]
                img_bytes = base64.b64decode(img_data)
                img = Image.open(BytesIO(img_bytes))
                img_filename = f"{images_dir}/image_{image_count + 1}.png"
                img.save(img_filename)
                image_files.append(img_filename)
                image_count += 1

image_files:
