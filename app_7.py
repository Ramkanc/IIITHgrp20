import torch
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from PIL import Image

from transformers import ViTImageProcessor

#processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")  
#processor.save_pretrained(r"./vit-gpt2-flickr8k")

# Load the saved model and tokenizer
model_path = "vit-gpt2-flickr8k"  # Update with your model's path
model = VisionEncoderDecoderModel.from_pretrained(model_path)
processor = ViTImageProcessor.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Image preprocessing function
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure RGB format
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    return pixel_values.to(device)

# Caption generation function
def generate_transform_caption(image_path):
    pixel_values = preprocess_image(image_path)
    
    # Generate caption
    gen_kwargs = {"max_length": 30, "num_beams": 5}  # Adjust parameters as needed
    output_ids = model.generate(pixel_values, **gen_kwargs)

    # Decode caption
    caption = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return caption.strip()

# Example usage
#image_path = r"C:\Users\Hp\Documents\IIIT\Capstone\App\flickr8k\Images\55470226_52ff517151.jpg"  # Update with your image path
#caption = generate_transform_caption(image_path)
#print("Generated Caption:", caption)