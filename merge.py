import os
import json
import h5py
from PIL import Image
import torch
import numpy as np
from torchvision import transforms

def getCharList():
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

def encode_to_labels(text, char_list):
    text = str(text)
    dig_lst = []
    for index, word in enumerate(text.split(" ")):
        try:
            dig_lst.append(char_list.index(word))
        except:
            for char in word:
                try:
                    dig_lst.append(char_list.index(char.lower()))
                except:
                    print("Not found char: " + char + " in " + word)
    return dig_lst

def merge_data(json_dir, img_dir, output_file, max_seq_length=128, img_size=(128, 128)):
    char_list = getCharList()
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
    ])
    
    images = []
    labels = []

    for json_file in os.listdir(json_dir):
        if json_file.endswith('.json'):
            json_path = os.path.join(json_dir, json_file)
            with open(json_path, 'r') as f:
                annotations = json.load(f)
            img_folder = os.path.join(img_dir, os.path.splitext(json_file)[0])
            if os.path.exists(img_folder):
                for i, latex in enumerate(annotations['crnn_label_list'], start=1):
                    img_path = os.path.join(img_folder, f'{i}.png')
                    if os.path.exists(img_path):
                        image = Image.open(img_path).convert("RGB")
                        image = transform(image)
                        label = encode_to_labels(latex, char_list)
                        label = label[:max_seq_length]  # truncate if necessary
                        if len(label) < max_seq_length:
                            label += [0] * (max_seq_length - len(label))  # pad with zeros
                        images.append(image.numpy())
                        labels.append(label)
    
    images = np.stack(images)
    labels = np.array(labels)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('images', data=images, compression='gzip')
        f.create_dataset('labels', data=labels, compression='gzip')
        # Encode char_list as UTF-8
        f.attrs['char_list'] = np.array(char_list, dtype=h5py.special_dtype(vlen=str))
    print(f"Data saved to {output_file}")

# Example usage
json_dir = 'latex_db/json'
img_dir = 'latex_db/output'
output_file = 'merged_data.h5'
merge_data(json_dir, img_dir, output_file)
