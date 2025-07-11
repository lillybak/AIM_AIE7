import requests

# Image URLs from notebook markdown cells
image_urls = [
    "https://i.imgur.com/vD8b016.png",
    "https://i.imgur.com/jTm9gjk.png"
]

downloaded_image_paths = []

# Download each image and save locally
for idx, url in enumerate(image_urls, start=1):
    response = requests.get(url)
    if response.status_code == 200:
        img_path = f"/mnt/data/image_{idx}.png"
        with open(img_path, "wb") as f:
            f.write(response.content)
        downloaded_image_paths.append(img_path)

downloaded_image_paths


