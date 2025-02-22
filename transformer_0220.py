from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image
import torch

# Load the saved model and tokenizer
model_path = "vit-gpt2-flickr8k" 
model = VisionEncoderDecoderModel.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    
    model.eval()  # Set the model to evaluation mode
    generated_ids = model.generate(pixel_values, max_length=50)  
    
    generated_caption = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_caption

# Example usage:
image_path = "/content/dataset/Flicker8k_Dataset/1012212859_01547e3f17.jpg"  # Replace with your image path
caption = generate_caption(image_path)
print(caption)