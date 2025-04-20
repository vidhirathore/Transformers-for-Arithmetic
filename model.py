# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import config

# --- Positional Encoding ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe) 

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# --- Position-wise Feed-Forward Network ---
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

# --- Encoder Layer ---
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        attn_output, _ = self.self_attn(src, src, src,
                                        attn_mask=src_mask,
                                        key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(attn_output)
        src = self.norm1(src)

        ff_output = self.feed_forward(src)
        src = src + self.dropout(ff_output)
        src = self.norm2(src)
        return src

# --- Decoder Layer ---
class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        attn_output, _ = self.self_attn(tgt, tgt, tgt,
                                        attn_mask=tgt_mask,
                                        key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm1(tgt)

        attn_output, _ = self.cross_attn(tgt, memory, memory,
                                         attn_mask=memory_mask,
                                         key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(attn_output)
        tgt = self.norm2(tgt)

        ff_output = self.feed_forward(tgt)
        tgt = tgt + self.dropout(ff_output)
        tgt = self.norm3(tgt)
        return tgt

# --- Encoder ---
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layers, n_heads, d_ff, dropout, max_len):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(config.DEVICE) 

    def forward(self, src, src_key_padding_mask=None):
        embedded = self.dropout(self.embedding(src) * self.scale) 
        pos_encoded = self.pos_encoding(embedded)

        output = pos_encoded
        for layer in self.layers:
            output = layer(output, src_key_padding_mask=src_key_padding_mask)
        return output

# --- Decoder ---
class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layers, n_heads, d_ff, dropout, max_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout)
                                     for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(config.DEVICE) 

    def forward(self, tgt, memory, tgt_mask=None, memory_key_padding_mask=None, tgt_key_padding_mask=None):
        embedded = self.dropout(self.embedding(tgt) * self.scale) 
        pos_encoded = self.pos_encoding(embedded)

        output = pos_encoded
        for layer in self.layers:
            output = layer(output, memory,
                           tgt_mask=tgt_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask)
        return output

# --- Seq2Seq Transformer Model ---
class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers,
                 d_model, n_head, input_vocab_size, output_vocab_size,
                 d_ff, dropout, max_len):
        super(Seq2SeqTransformer, self).__init__()

        self.encoder = Encoder(input_vocab_size, d_model, num_encoder_layers, n_head, d_ff, dropout, max_len)
        self.decoder = Decoder(output_vocab_size, d_model, num_decoder_layers, n_head, d_ff, dropout, max_len)
        self.fc_out = nn.Linear(d_model, output_vocab_size)

    def _create_padding_mask(self, seq, pad_idx):
        return seq == pad_idx

    def _create_look_ahead_mask(self, size, device):
        
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, src, tgt):
        """
        Args:
            src: Input sequence tensor, shape [batch_size, src_len]
            tgt: Target sequence tensor, shape [batch_size, tgt_len]
        """
        src_key_padding_mask = self._create_padding_mask(src, config.PAD_IDX) 
        tgt_len = tgt.shape[1]
        tgt_mask = self._create_look_ahead_mask(tgt_len, src.device) # [tgt_len, tgt_len]
        tgt_key_padding_mask = self._create_padding_mask(tgt, config.PAD_IDX) 
        memory_key_padding_mask = src_key_padding_mask

        memory = self.encoder(src, src_key_padding_mask=src_key_padding_mask)

        decoder_output = self.decoder(tgt, memory,
                                      tgt_mask=tgt_mask,
                                      memory_key_padding_mask=memory_key_padding_mask,
                                      tgt_key_padding_mask=tgt_key_padding_mask)

        output = self.fc_out(decoder_output)
        return output

def build_model():
    model = Seq2SeqTransformer(
        num_encoder_layers=config.N_LAYERS,
        num_decoder_layers=config.N_LAYERS,
        d_model=config.D_MODEL,
        n_head=config.N_HEADS,
        input_vocab_size=config.VOCAB_SIZE,
        output_vocab_size=config.VOCAB_SIZE,
        d_ff=config.D_FF,
        dropout=config.DROPOUT,
        max_len=config.MAX_SEQ_LEN
    )

    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model.to(config.DEVICE)

if __name__ == "__main__":
    model = build_model()
    print(f"Model initialized on {config.DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    batch_size = 4
    src_len = 10
    tgt_len = 8
    dummy_src = torch.randint(0, config.VOCAB_SIZE, (batch_size, src_len), device=config.DEVICE)
    dummy_tgt = torch.randint(0, config.VOCAB_SIZE, (batch_size, tgt_len), device=config.DEVICE)
    dummy_src[0, 5:] = config.PAD_IDX 
    dummy_tgt[0, 6:] = config.PAD_IDX

    print("Dummy Source Shape:", dummy_src.shape)
    print("Dummy Target Shape:", dummy_tgt.shape)

   
    model.train()
    output = model(dummy_src, dummy_tgt)
    print("Output Logits Shape:", output.shape) 