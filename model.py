import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim, dropout=0.1):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.patch_embedding.n_patches + 1, embed_dim))
        encoder_layers = nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)

    def forward(self, x):
        x = self.patch_embedding(x)
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = x.transpose(0, 1)
        x = self.transformer_encoder(x)
        x = x[1:].transpose(0, 1)  # Exclude cls_token and transpose for LSTM input
        return x

class LatexVisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim, vocab_size, max_seq_length, dropout=0.1):
        super(LatexVisionTransformer, self).__init__()
        self.encoder = VisionTransformer(img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.lstm = nn.LSTM(embed_dim, 256, bidirectional=True, num_layers=2, dropout=0.2, batch_first=True)
        self.dense = nn.Linear(512, vocab_size)  # 512 = 256 * 2 (bidirectional LSTM)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt=None):
        enc_out = self.encoder(src)
        lstm_out, _ = self.lstm(enc_out)
        output = self.dropout(lstm_out)
        output = self.dense(output)
        return output

def get_model(vocab_size, max_seq_length):
    img_size = (128, 64)  # Image size (height, width)
    patch_size = (16, 16)  # Adjust patch size to fit the image dimensions
    in_channels = 3
    embed_dim = 512
    num_layers = 6
    num_heads = 8
    mlp_dim = 2048
    dropout = 0.1

    model = LatexVisionTransformer(
        img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim, vocab_size, max_seq_length, dropout
    )
    return model
