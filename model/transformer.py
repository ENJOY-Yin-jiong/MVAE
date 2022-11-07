import torch.nn as nn
import torch
import math

from operation import Conv1D, mask_logits, ConvTranspose1D


class TransformerPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, dim).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, dim, 2).float() *
                    -(math.log(10000.0) / dim)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class PositionalEmbedding(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""
    def __init__(self, embedding_dim, num_embeddings):
        super(PositionalEmbedding, self).__init__()
        self.position_embeddings = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, inputs):
        bsz, seq_length = inputs.shape[:2]
        position_ids = torch.arange(seq_length,
                                    dtype=torch.long,
                                    device=inputs.device)
        position_ids = position_ids.unsqueeze(0).repeat(bsz, 1)  # (N, L)
        position_embeddings = self.position_embeddings(position_ids)
        return position_embeddings


class MultiHeadAttention(nn.Module):
    def __init__(self, configs):
        super(MultiHeadAttention, self).__init__()
        dim = configs.dim
        num_heads = configs.num_heads
        drop_rate = configs.drop_rate
        assert dim % num_heads == 0, 'The channels (%d) is not a multiple of attention heads (%d)' % (
            dim, num_heads)
        self.head_size, self.num_heads, self.dim = int(
            dim / num_heads), num_heads, dim
        self.dropout = nn.Dropout(p=drop_rate)
        self.query = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        self.key = Conv1D(in_dim=dim,
                          out_dim=dim,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True)
        self.value = Conv1D(in_dim=dim,
                            out_dim=dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True)
        # self.value_visual = None
        self.layer_norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.layer_norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.out_layer1 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)
        self.output_activation = nn.GELU()
        self.out_layer2 = Conv1D(in_dim=dim,
                                 out_dim=dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=True)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_heads, self.head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, num_heads, w_seq_len, head_size)

    @staticmethod
    def combine_last_two_dim(x):
        old_shape = list(x.size())
        new_shape = old_shape[:-2] + [old_shape[-2] * old_shape[-1]]
        return x.reshape(shape=new_shape)

    def forward(self, x, mask=None):
        output = self.layer_norm1(x)  # (batch_size, seq_len, dim)
        # output = self.dropout(output)
        # multi-head attention layer
        query = self.transpose_for_scores(
            self.query(output))  # (batch_size, num_heads, seq_len, head_size)
        key = self.transpose_for_scores(self.key(output))
        value = self.transpose_for_scores(self.value(output))
        attention_scores = torch.matmul(query, key.transpose(
            -1, -2))  # (batch_size, num_heads, seq_len, seq_len)
        attention_scores = attention_scores / math.sqrt(self.head_size)
        if mask is not None:  # masking
            mask = mask.unsqueeze(1).unsqueeze(
                2)  # (batch_size, 1, 1, seq_len)
            attention_scores = mask_logits(attention_scores, mask)
        attention_probs = torch.softmax(
            attention_scores,
            dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        attention_probs = self.dropout(attention_probs)
        value = torch.matmul(
            attention_probs,
            value)  # (batch_size, num_heads, seq_len, head_size)
        value = self.combine_last_two_dim(value.permute(
            0, 2, 1, 3))  # (batch_size, seq_len, dim)
        # intermediate layer
        output = self.dropout(value)
        residual = x + output
        output = self.layer_norm2(residual)
        output = self.out_layer1(output)
        output = self.output_activation(output)
        output = self.dropout(output)
        output = self.out_layer2(output) + residual
        return output


class ConvMLPEncoder(nn.Module):
    def __init__(self, configs):
        super(ConvMLPEncoder, self).__init__()

        self.in_dim = configs.MODEL.ENCODER.in_dim
        self.out_dim = configs.MODEL.ENCODER.out_dim

        self.mlp1 = Conv1D(self.in_dim, self.out_dim)
        self.activation = nn.GELU()
        self.mlp1 = Conv1D(self.in_dim, self.out_dim)
        self.dropout = nn.Dropout()


    def forward(self, input):
        input = self.mlp1(input)
        input = self.activation(input)
        input = self.mlp2(input)
        output =  self.dropout(input)
        return output


class ConvMLPDecoder(nn.Module):
    def __init__(self, configs):
        super(ConvMLPEncoder, self).__init__()

        self.in_dim = configs.MODEL.DECODER.in_dim
        self.out_dim = configs.MODEL.DECODER.out_dim

        self.mlp1 = ConvTranspose1D(self.in_dim, self.out_dim)
        self.activation = nn.GELU()
        self.mlp1 = ConvTranspose1D(self.in_dim, self.out_dim)
        self.dropout = nn.Dropout()

    def forward(self, input):
        input = self.mlp1(input)
        input = self.activation(input)
        input = self.mlp2(input)
        output = self.dropout(input)
        return output


class MusicTransformerEncoder(nn.Module):
    def __init__(self, configs):
        super(MusicTransformerEncoder, self).__init__()

        self.n_layers = configs.MODEL.ENCODER.num_layers
        self.blocks = nn.Sequential(*[MultiHeadAttention(configs.MODEL.ENCODER)
                                      for _ in range(self.n_layers)])
        self.mlp = ConvMLPEncoder(configs)


    def forward(self, input, mask=None):

        out = input
        for block_idx in range(len(self.blocks)):
            out = self.blocks[block_idx](out, mask)

        out = self.mlp(out)

        return out


class MusicTransformerDecoder(nn.Module):
    def __init__(self, configs):
        super(MusicTransformerDecoder, self).__init__()

        self.n_layers = configs.MODEL.DECODER.num_layers
        self.blocks = nn.Sequential(*[MultiHeadAttention(configs.MODEL.DECODER)
                                      for _ in range(self.n_layers)])

        self.mlp = ConvMLPDecoder(configs)

    def forward(self, input, mask=None):

        out = self.mlp(input)
        for block_idx in range(len(self.blocks)):
            out = self.blocks[block_idx](out, mask)

        return out



