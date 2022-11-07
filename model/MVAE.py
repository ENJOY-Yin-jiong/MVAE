import torch
import math
import torch.nn as nn

from transformer import PositionalEmbedding, MusicTransformerEncoder, MusicTransformerDecoder
from music_vqvae import VectorQuantizer


class Embeddings(nn.Module):
    def __init__(self, n_token, d_model):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(n_token, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class MVAE(nn.Module):
    def __init__(self, configs, n_class):
        super(MVAE, self).__init__()

        # [128, 256, 64, 32, 512, 128, 128]
        self.emb_sizes = configs.MODEL.emb_sizes
        n_class = [56, 135, 18, 3, 87, 18, 25]
        self.n_token = n_class

        self.dropout_rate = configs.MODEL.drop_rate
        self.dim = configs.MODEL.dim

        # VQ-VAE config
        self.cb_length = configs.MODEL.VQVAE.codebook_len
        self.e_dim = configs.MODEL.VQVAE.e_dim
        self.beta = configs.MODEL.VQVAE.beta

        # Embedding
        self.word_emb_tempo = Embeddings(self.n_token[0], self.emb_sizes[0])
        self.word_emb_chord = Embeddings(self.n_token[1], self.emb_sizes[1])
        self.word_emb_barbeat = Embeddings(self.n_token[2], self.emb_sizes[2])
        self.word_emb_type = Embeddings(self.n_token[3], self.emb_sizes[3])
        self.word_emb_pitch = Embeddings(self.n_token[4], self.emb_sizes[4])
        self.word_emb_duration = Embeddings(self.n_token[5], self.emb_sizes[5])
        self.word_emb_velocity = Embeddings(self.n_token[6], self.emb_sizes[6])

        # pred
        self.proj_tempo = nn.Linear(self.d_model, self.n_token[0])
        self.proj_chord = nn.Linear(self.d_model, self.n_token[1])
        self.proj_barbeat = nn.Linear(self.d_model, self.n_token[2])
        self.proj_type = nn.Linear(self.d_model, self.n_token[3])
        self.proj_pitch = nn.Linear(self.d_model, self.n_token[4])
        self.proj_duration = nn.Linear(self.d_model, self.n_token[5])
        self.proj_velocity = nn.Linear(self.d_model, self.n_token[6])

        # transformer encoder
        self.pos_emb = PositionalEmbedding(self.dim, self.dropout_rate)

        # Encoder
        self.encoder = MusicTransformerEncoder(configs)

        # Decoder
        self.decoder = MusicTransformerDecoder(configs)

        # vqvae
        self.vqvae = VectorQuantizer(self.cb_length, self.e_dim)

        # loss
        self.loss_func = nn.CrossEntropyLoss(reduction='none')


    def emb_input(self, input):
        emb_tempo = self.word_emb_tempo(input[..., 0])
        emb_chord = self.word_emb_chord(input[..., 1])
        emb_barbeat = self.word_emb_barbeat(input[..., 2])
        emb_type = self.word_emb_type(input[..., 3])
        emb_pitch = self.word_emb_pitch(input[..., 4])
        emb_duration = self.word_emb_duration(input[..., 5])
        emb_velocity = self.word_emb_velocity(input[..., 6])

        embs = torch.cat(
            [
                emb_tempo,
                emb_chord,
                emb_barbeat,
                emb_type,
                emb_pitch,
                emb_duration,
                emb_velocity,
            ], dim=-1)

        return embs

    # to be changed
    def forward_output(self, h, y):
        '''
        for training
        '''
        tf_skip_type = self.word_emb_type(y[..., 3])

        # project other
        y_concat_type = torch.cat([h, tf_skip_type], dim=-1)
        y_  = self.project_concat_type(y_concat_type)

        print('y_', y_.shape)
        y_tempo    = self.proj_tempo(y_)
        y_chord    = self.proj_chord(y_)
        y_barbeat  = self.proj_barbeat(y_)
        y_pitch    = self.proj_pitch(y_)
        y_duration = self.proj_duration(y_)
        y_velocity = self.proj_velocity(y_)

        pred = {
            'tempo': y_tempo,
            'chord': y_chord,
            'barbeat': y_barbeat,
            'pitch': y_pitch,
            'duration': y_duration,
            'velocity': y_velocity,
        }

        return pred


    def forward(self, batch_x, batch_y, batch_mask):
        input_emb = self.emb_input(batch_x)

        input_feat = input_emb + self.pos_emb(input)

        # encoder
        enc_feat = self.encoder(input_feat)

        # VQVAE
        hidden_feat, cb_loss, _ = self.vqvae(enc_feat)

        # Decoder
        dec_feat = self.decoder(hidden_feat)

        pred = self.forward_output(dec_feat, batch_y)

        y_type = self.proj_type(enc_feat)
        pred['type'] = y_type

        m_loss = self.loss(pred, batch_y)

        return pred, cb_loss, m_loss


    def loss(self, pred, target, loss_mask):
        loss_tempo = self.compute_loss(
            pred['tempo'], target[..., 0], loss_mask)
        loss_chord = self.compute_loss(
            pred['chord'], target[..., 1], loss_mask)
        loss_barbeat = self.compute_loss(
            pred['barbeat'], target[..., 2], loss_mask)
        loss_type = self.compute_loss(
            pred['type'], target[..., 3], loss_mask)
        loss_pitch = self.compute_loss(
            pred['pitch'], target[..., 4], loss_mask)
        loss_duration = self.compute_loss(
            pred['duration'], target[..., 5], loss_mask)
        loss_velocity = self.compute_loss(
            pred['velocity'], target[..., 6], loss_mask)

        return (loss_tempo + loss_chord + loss_barbeat + loss_type \
               + loss_pitch + loss_duration + loss_velocity) / 7

    def compute_loss(self, predict, target, loss_mask):
        loss = self.loss_func(predict, target)
        loss = loss * loss_mask
        loss = torch.sum(loss) / torch.sum(loss_mask)
        return loss

    # def sampling(self):


