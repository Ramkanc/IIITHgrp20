import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
#import shutil

from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embed_dim = 512  # Embedding dimension of CLIP
hidden_dim = 512  # Hidden dimension of LSTM
tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer
vocab_size = tokenizer.vocab_size
#print(f"length of tokenizer: {vocab_size}")

# Image Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Encoder: CLIP
class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32"):
        super(CLIPEncoder, self).__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.processor = CLIPProcessor.from_pretrained(model_name)


    def forward(self, images):
        with torch.no_grad():
            image_features = self.clip_model.get_image_features(images)

        return image_features
    


# Decoder: GPT-2
class GPT2Decoder(nn.Module):
    def __init__(self, model_name="gpt2"):
        super(GPT2Decoder, self).__init__()
        self.decoder = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Set pad token for GPT-2

    def forward(self, features, captions):
        # Prepare inputs for GPT-2 decoder
        input_ids = captions
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).type(torch.float32)    

        # Generate outputs from the decoder
        outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            
        )
        return outputs.logits
    
# Load the trained encoder and decoder models
encoder = CLIPEncoder().to(device)
encoder.load_state_dict(torch.load("encoder_model_clip.pth"))

decoder = GPT2Decoder().to(device)
decoder.load_state_dict(torch.load("decoder_model_gpt2.pth"))



# Caption Generation
def generate_transform_caption(img, encoder=encoder, decoder=decoder, max_length=20):
    img= Image.open(img)
    image_tensor = transform(img).to(device)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        image_features = encoder(image_tensor.unsqueeze(0))  # Encode the image
        caption_ids = [decoder.tokenizer.bos_token_id]  # Start with BOS token
        for _ in range(max_length):
            caption_tensor = torch.tensor([caption_ids]).to(image_features.device)  # Move to the same device
            output = decoder(image_features, caption_tensor)
            predicted_id = output.argmax(2)[:, -1].item()
            if predicted_id == decoder.tokenizer.eos_token_id:  # Stop when EOS token is predicted
                break
            caption_ids.append(predicted_id)
        decoded_caption = decoder.tokenizer.decode(caption_ids[1:], skip_special_tokens=True)  # Decode, skipping BOS
        return decoded_caption  


# img = r"C:\Users\kancharr\Documents\IIIT\Flickr8k\Images\229978782_3c690f5a0e.jpg"
# img = Image.open(img)
# img = transform(img).to(device)
# caption = generate_caption(img, encoder, decoder)

# print(caption)