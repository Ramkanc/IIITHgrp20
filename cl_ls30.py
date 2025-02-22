
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import nltk
import torch.nn as nn
import torch.nn.functional as F

# Define device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained model
embed_dim = 512
hidden_dim = 512
tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer
vocab_size = tokenizer.vocab_size
# Define the model architecture
# Assuming the model architecture is defined as in your notebook
# (CLIPEncoder, DecoderWithAttention, ImageCaptioningModel)

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

    
# Decoder: LSTM without Attention
class Decoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)
        features = features.unsqueeze(1).repeat(1, captions.size(1), 1)  # Expand features to match caption length
        lstm_input = torch.cat((embeddings, features), dim=2)
        output, _ = self.lstm(lstm_input)
        outputs = self.fc(output)
        return outputs
    
# Combined Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = CLIPEncoder()
        self.decoder = Decoder(embed_dim, hidden_dim, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


model = ImageCaptioningModel(embed_dim, hidden_dim, vocab_size).to(device)
model.load_state_dict(torch.load('Clip_lstm_noat_8_20.pth', map_location=device))


# Image Transform
from torchvision import transforms
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 

def gen_caption_beam_search30(img, beam_size=3):
    img = Image.open(img)
    image_tensor = transform(img).to(device)
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)

        # Initialize beam search
        start_token = torch.tensor([tokenizer.bos_token_id]).to(device)
        sequences = [[start_token, 0.0]]  # (sequence, log_probability)

        for _ in range(20):  # Maximum caption length
            all_candidates = []
            for seq, score in sequences:
                inputs = seq.unsqueeze(0)
                outputs = model(image_tensor, inputs)
                probs = F.softmax(outputs[:, -1], dim=-1)  # Probabilities for next word

                # Get top k probabilities and their indices
                top_k_probs, top_k_indices = probs.topk(beam_size)

                for prob, index in zip(top_k_probs[0], top_k_indices[0]):
                    candidate = [torch.cat([seq, index.unsqueeze(0)]), score + torch.log(prob).item()]
                    all_candidates.append(candidate)

            # Order candidates by score and select top k
            ordered = sorted(all_candidates, key=lambda tup: tup[1], reverse=True)
            sequences = ordered[:beam_size]

            # Check for EOS token
            best_sequence = sequences[0][0]
            if best_sequence[-1].item() == tokenizer.eos_token_id:
                break

        # Decode the best sequence
        decoded_caption = tokenizer.decode(best_sequence, skip_special_tokens=True)
        return decoded_caption



    

# Load an image
#img= r"flickr8k\Images\124972799_de706b6d0b.jpg"

#caption  = gen_caption_beam_search8(img)
#print(caption)