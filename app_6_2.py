#import streamlit as st
import torch
import clip
from PIL import Image
import numpy as np
import pandas as pd

# Load the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# Load the captions dataset
captions_file = r"Flickr30k.csv"
captions_df = pd.read_csv(captions_file, delimiter=",", header=0)
#captions_df["image"] = captions_df["image"].str.split(".").str[0]

# Load precomputed features
text_features_cache = np.load("text_features_cache_30k.npy")
image_features_cache = np.load("image_features_cache_30k.npy")
image_paths = np.load("image_paths_30k.npy", allow_pickle=True)

def cos_30k_image_to_text(image):
    try:
        # Preprocess the uploaded image
        image = Image.open(image).convert("RGB")        
        image = preprocess(image).unsqueeze(0).to(device)

        # Encode the image
        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        # Compute similarities in batch
        image_features_np = image_features.cpu().numpy()
        similarities = np.dot(text_features_cache, image_features_np.T).squeeze()

        # Find the most similar caption
        most_similar_idx = np.argmax(similarities)
        most_similar_caption = captions_df.iloc[most_similar_idx]['caption']
        return most_similar_caption, most_similar_idx  
    except Exception as e:
        raise RuntimeError(f"Error processing uploaded image: {e}")
      


def cos_30k_text_to_image(caption):
    try:
        # Tokenize and encode the input caption
        text = clip.tokenize([caption]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Compute similarities in batch
        text_features_np = text_features.cpu().numpy()
        similarities = np.dot(image_features_cache, text_features_np.T).squeeze()

        # Find the most similar image
        most_similar_idx = np.argmax(similarities)
        most_similar_image_path = image_paths[most_similar_idx]
        return most_similar_image_path, most_similar_idx
    except Exception as e:
        raise RuntimeError(f"Error processing input caption: {e}")