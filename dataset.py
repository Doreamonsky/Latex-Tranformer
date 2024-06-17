import os
import json
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class LatexImageDataset(Dataset):
    def __init__(self, json_dir, img_dir, transform=None, max_seq_length=128):
        self.json_dir = json_dir
        self.img_dir = img_dir
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.char_list = self.getCharList()
        self.data = self.load_data()

    def getCharList(self):
        return ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '+', '-', '.',
                r'\sqrt', 'sqrt{', 'sqrt}', 'sqrt[', 'sqrt]', r'\frac', 'frac{', 'frac}', '^',
                'pow{', 'pow}', '=', '<', '>', '\pm', '\leq', '\geq', '\{', '\}', '(', ')', '[', ']', '\mid', '∪',
                '\pi', 'e', '\ln', '\lg', '\log_', '\log{', '\log}',
                '\sin', '\cos', '\tan', '\cot', '\csc', '\sec',
                'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z',
                '\alpha', '\beta', '\gamma', '\theta',
                '或', ',', '\infty', '\cup', '\cap', '_', '_{', '_}', '\emptyset', '\in',
                '|', '{', '}', 'blank14', 'blank15', 'blank16', 'blank17', 'blank18', 'blank19', 'blank20'
                ]

    def encode_to_labels(self, text):
        text = str(text)
        dig_lst = []
        for index, word in enumerate(text.split(" ")):
            try:
                dig_lst.append(self.char_list.index(word))
            except:
                for char in word:
                    try:
                        dig_lst.append(self.char_list.index(char.lower()))
                    except:
                        print("Not found char: " + char + " in " + word)
        return dig_lst

    def decode_from_labels(self, labels):
        decoded_text = []
        for label in labels:
            if label != 0:  # Skip padding index
                decoded_text.append(self.char_list[label])
        return ''.join(decoded_text)

    def load_data(self):
        data = []
        for json_file in os.listdir(self.json_dir):
            if json_file.endswith('.json'):
                json_path = os.path.join(self.json_dir, json_file)
                with open(json_path, 'r') as f:
                    annotations = json.load(f)
                img_folder = os.path.join(self.img_dir, os.path.splitext(json_file)[0])
                if os.path.exists(img_folder):
                    for i, latex in enumerate(annotations['crnn_label_list'], start=1):
                        img_path = os.path.join(img_folder, f'{i}.png')
                        if os.path.exists(img_path):
                            label = self.encode_to_labels(latex)
                            label = label[:self.max_seq_length]  # truncate if necessary
                            if len(label) < self.max_seq_length:
                                label += [0] * (self.max_seq_length - len(label))  # pad with zeros
                            data.append((img_path, torch.tensor(label)))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_dataloaders(json_dir, img_dir, batch_size=64, max_seq_length=128):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    dataset = LatexImageDataset(json_dir, img_dir, transform=transform, max_seq_length=max_seq_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
