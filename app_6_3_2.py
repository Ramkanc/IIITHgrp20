
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
    
# Attention Mechanism for Encoder-Decoder
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden: [batch_size, hidden_size]
        # encoder_outputs: [batch_size, seq_len, hidden_size]

        hidden = hidden.unsqueeze(1).repeat(1, encoder_outputs.size(1), 1)  # [batch_size, seq_len, hidden_size]

        combined = torch.cat((hidden, encoder_outputs), dim=2)

        energy = torch.tanh(self.attn(combined))  # [batch_size, seq_len, hidden_size]
        attention = self.v(energy).squeeze(2)  # [batch_size, seq_len]

        return F.softmax(attention, dim=1)
    
# Decoder: LSTM with Encoder-Decoder Attention
class DecoderWithAttention(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size, num_layers=1):
        super(DecoderWithAttention, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)

        hidden_states = []
        output, (hidden, cell) = self.lstm(embeddings)

        context = []
        for t in range(output.size(1)):
            attn_weights = self.attention(hidden[-1], features.unsqueeze(1))  # features expanded for batch processing
            context_vector = torch.bmm(attn_weights.unsqueeze(1), features.unsqueeze(1)).squeeze(1)
            context.append(context_vector)

        context = torch.stack(context, dim=1)  # [batch_size, seq_len, hidden_size]

        combined = torch.cat((output, context), dim=2)  # [batch_size, seq_len, hidden_size * 2]

        outputs = self.fc(combined)
        return outputs
    
# Combined Model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_dim, hidden_dim, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = CLIPEncoder()
        self.decoder = DecoderWithAttention(embed_dim, hidden_dim, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


model = ImageCaptioningModel(embed_dim, hidden_dim, vocab_size).to(device)
model.load_state_dict(torch.load('Clip_lstm_att_8_20.pth', map_location=device))


# Image Transform
from torchvision import transforms
transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]) 

    


# Caption generation
# def generate_caption(img):
#     img = Image.open(img)
#     image_tensor = transform(img).to(device)
#     model.eval()
#     with torch.no_grad():
#         image_tensor = image_tensor.unsqueeze(0).to(device)
#         caption = [1]  # Assuming <SOS> token is 1
#         for _ in range(20):  # Maximum caption length
#             caption_tensor = torch.tensor(caption).unsqueeze(0).to(device)
#             output = model(image_tensor, caption_tensor)
#             next_word = output.argmax(2)[:, -1].item()
#             if next_word == 2:  # Assuming <EOS> token is 2
#                 break
#             caption.append(next_word)
#         decoded_caption = tokenizer.decode(caption, skip_special_tokens=True)
#         return decoded_caption


def generate_caption_beam_search_30(img, beam_size=3):
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

#caption  = generate_caption_beam_search(img)
#print(caption)