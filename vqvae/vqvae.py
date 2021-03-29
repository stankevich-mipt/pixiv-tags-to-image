"""
Author: Stankevich Andrey, MIPT <stankevich.as@phystech.edu>
"""


import numpy as np
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .quantizer import VectorQuantizer


class VQVAE(nn.Module):

	def __init__(
		self,
		device,
		n_hid,
		n_emb,
		embedding_dim
	):

		super().__init__()

		self.device        = device
		self.n_hid         = n_hid
		self.n_emb         = n_emb 
		self.embedding_dim = embedding_dim

		self.encoder   = Encoder(n_hid=self.n_hid, n_out=self.n_emb)
		self.quantizer = VectorQuantizer(self.embedding_dim, self.n_emb)
		self.decoder   = Decoder(input_channels=self.embedding_dim, n_hid=self.n_hid)


	def encode(self, x):

		enc = self.encoder(x)
		quant, diff, ids = self.quantizer(enc.permute(0, 2, 3, 1))
		quant = quant.permute(0, 3, 1, 2)

		return quant, diff, ids

	def decode(self, x): return self.decoder(x)

	def forward(self, x):
		quant, diff, _ = self.encode(x)
		return torch.sigmoid(self.decode(quant)), diff


class VQVAE2(nn.Module):

	def __init__(
		self,
		device,
		n_hid,
		embedding_dim,
		n_embed
	):

		super().__init__()

		self.device        = device
		self.n_hid         = n_hid 
		self.embedding_dim = embedding_dim
		self.n_embed       = n_embed

		self.encoder_t   = Encoder(
			input_channels=self.n_hid, 
			n_hid=self.n_hid, n_out=self.n_hid, n_groups=1
		)
		self.encoder_b   = Encoder(input_channels=3, n_hid=self.n_hid, n_out=self.n_hid, n_groups=2)

		self.quantize_conv_t = nn.Conv2d(self.n_hid, self.embedding_dim, 1)
		self.quantize_conv_b = nn.Conv2d(self.n_hid + self.embedding_dim, self.embedding_dim, 1) 
		
		self.quantizer_t = VectorQuantizer(self.embedding_dim, self.n_embed)
		self.quantizer_b = VectorQuantizer(self.embedding_dim, self.n_embed)

		self.decoder_t = Decoder(
			input_channels=self.embedding_dim, n_hid=self.n_hid, n_groups=1, n_out=self.n_hid)
		self.decoder_b = Decoder(
			input_channels=self.embedding_dim * 2, n_hid=self.n_hid, n_groups=2)
		
		self.upsample_t = nn.ConvTranspose2d(
			self.embedding_dim, self.embedding_dim, 4, stride=2, padding=1
		)


	def encode(self, x):

		enc_b = self.encoder_b(x)
		enc_t = self.encoder_t(enc_b)
		
		quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)
		quant_t, diff_t, id_t = self.quantizer_t(quant_t)
		quant_t = quant_t.permute(0, 3, 1, 2)
		diff_t = diff_t.unsqueeze(0)
		dec_t = self.decoder_t(quant_t)  

		enc_b   = torch.cat([dec_t, enc_b], 1)
		quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)
		quant_b, diff_b, id_b = self.quantizer_b(quant_b)
		quant_b = quant_b.permute(0, 3, 1, 2)
		diff_b  = diff_b.unsqueeze(0)

		return quant_t, quant_b, diff_t + diff_b, id_t, id_b

	def decode(self, quant_t, quant_b): 
		upsample_t = self.upsample_t(quant_t)
		quant = torch.cat([upsample_t, quant_b], 1)

		return self.decoder_b(quant)

	def forward(self, x):
		quant_t, quant_b, diff, _, _ = self.encode(x)
		return torch.sigmoid(self.decode(quant_t, quant_b)), diff