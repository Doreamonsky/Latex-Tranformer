import torch
import torch.nn as nn
from torch.utils.data import DataLoader

class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
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
        self.pos_embedding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
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
        x = x[0]
        return x

class LatexVisionTransformer(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim, vocab_size, max_seq_length, dropout=0.1):
        super(LatexVisionTransformer, self).__init__()
        self.encoder = VisionTransformer(img_size, patch_size, in_channels, embed_dim, num_layers, num_heads, mlp_dim, dropout)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, num_heads, mlp_dim, dropout),
            num_layers
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_length, embed_dim))
        self.embed_dim = embed_dim

    def forward(self, src, tgt):
        enc_out = self.encoder(src)
        tgt_seq_len = tgt.shape[1]
        tgt_emb = self.embedding(tgt) * (self.embed_dim ** 0.5)
        tgt_emb = tgt_emb + self.pos_embedding[:, :tgt_seq_len, :]
        tgt_emb = tgt_emb.transpose(0, 1)
        enc_out = enc_out.unsqueeze(0).repeat(tgt_seq_len, 1, 1)
        output = self.decoder(tgt_emb, enc_out)
        output = self.fc_out(output)
        return output

def get_model(vocab_size, max_seq_length):
    img_size = 128
    patch_size = 16
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
