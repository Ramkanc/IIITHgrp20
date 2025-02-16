import torch
import clip
import pandas as pd
from PIL import Image
import os
import numpy as np

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the captions dataset
captions_file = "Flickr30k_t.csv"
captions_df = pd.read_csv(captions_file, delimiter=",", header=0)
captions_df["image"] = captions_df["image"].str.split(".").str[0]
captions_df.to_csv("Flickr30k.csv", index=False)

print("text features are being computed")

# Define the maximum context length for the CLIP model
max_context_length = 77

# Precompute and save text features
text_features_cache = []
for idx, row in captions_df.iterrows():
    caption = row['caption']
    # Truncate the caption if it exceeds the maximum context length
    truncated_caption = caption[:max_context_length]
    text = clip.tokenize([truncated_caption]).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        text_features_cache.append(text_features.cpu().numpy())

text_features_cache = np.vstack(text_features_cache)
np.save("text_features_cache_30k.npy", text_features_cache)
print("text features computed")

# Precompute and save image features
print("image features are being computed")
image_features_cache = []
image_paths = []
for idx, row in captions_df.iterrows():
    image_id = row['image']
    image_path = os.path.join(r"C:\Users\kancharr\Documents\IIIT\Flickr30k\flickr30k_images", f"{image_id}.jpg")
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features_cache.append(image_features.cpu().numpy())
        image_paths.append(image_path)
image_features_cache = np.vstack(image_features_cache)
np.save("image_features_cache_30k.npy", image_features_cache)
np.save("image_paths_30k.npy", image_paths)
print("image features computed")