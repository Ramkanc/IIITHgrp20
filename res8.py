import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import pickle
import torchvision.models as models
import spacy
import os
import pandas as pd
from collections import Counter




# Load Spacy Tokenizer
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self,freq_threshold):
        self.itos = {0:"<PAD>",1:"<SOS>",2:"<EOS>",3:"<UNK>"}
        self.stoi = {v:k for k,v in self.itos.items()}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [token.text.lower() for token in spacy_eng.tokenizer(text)]

    def build_vocab(self,sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenize(sentence):
                frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self,text):
        tokenized_text = self.tokenize(text)
        return [self.stoi[token] if token in self.stoi else self.stoi["<UNK>"] for token in tokenized_text]


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        #resnet = models.resnet50(pretrained=True)
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        for param in resnet.parameters():
            param.requires_grad_(False)

        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
  
    def forward(self, images):        
        features = self.resnet(images)  # Output shape: (batch, 2048, 1, 1)
        #print("Raw ResNet output shape:", features.shape)

        features = features.view(features.size(0), -1)  # Flatten to (batch, 2048)
        #print("Flattened ResNet output shape:", features.shape)

        features = self.embed(features)  # Linear layer to (batch, embed_size)
        #print("Final encoded feature shape:", features.shape)

        return features


class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.fcn = nn.Linear(hidden_size, vocab_size)
        self.drop = nn.Dropout(drop_prob)

    def forward(self, features, captions):
        embeds = self.embedding(captions[:, :-1])
        x = torch.cat((features.unsqueeze(1), embeds), dim=1)
        x, _ = self.lstm(x)
        x = self.fcn(x)
        return x

    def generate_caption_res8(self, inputs, hidden=None, max_len=20, vocab=None):
        """Generate captions given image features"""
        batch_size = inputs.size(0)
        captions = []

        if hidden is None:
            hidden = (
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(inputs.device),
                torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size).to(inputs.device),
            )

        inputs = inputs.unsqueeze(1)  # Add sequence length = 1
        #inputs = inputs.view(1, 1, -1)
        #print("LSTM input shape:", inputs.shape) 

        for _ in range(max_len):
            output, hidden = self.lstm(inputs, hidden)
            output = self.fcn(output.squeeze(1))
            predicted_word_idx = output.argmax(dim=1)
            captions.append(predicted_word_idx.item())

            if vocab.itos[predicted_word_idx.item()] == "<EOS>":
                break

            inputs = self.embedding(predicted_word_idx.unsqueeze(1))

        return [vocab.itos[idx] for idx in captions]


class EncoderDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, drop_prob=0.3):
        super(EncoderDecoder, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers, drop_prob)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df_cap = pd.read_csv(r"Flickr8k.csv", delimiter=",", header=0)

cap_list = df_cap['caption'].tolist()
print(f"Loaded {len(cap_list)} captions from dataset.")





# Load Vocabulary
vocab = Vocabulary(freq_threshold=5)

# Build vocabulary from captions

vocab.build_vocab(cap_list)  # Builds the vocab dynamically

print(f"Vocabulary built with {len(vocab)} unique words.")




# Model Parameters
embed_size = 512
hidden_size = 512
vocab_size = len(vocab)
num_layers = 2

# Load Model
model_path = "resnet_lstm_8.pth"
model = EncoderDecoder(embed_size, hidden_size, vocab_size, num_layers).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Image Transformations
transforms = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])




def gen_caption_res8(image_path):
    """Generate caption for an input image""" 
    
    image = Image.open(image_path).convert("RGB")
    image = transforms(image).unsqueeze(0).to(device)
    features = model.encoder(image)
    caption = model.decoder.generate_caption_res8(features, vocab=vocab)
    return " ".join(caption)


# Test with an image
#img_path = r"C:\Users\Hp\Documents\IIIT\Capstone\App\flickr8k\Images\50030244_02cd4de372.jpg"
#cap = gen_caption_res8(img_path)
#print(cap)