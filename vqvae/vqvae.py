"""
Author: Stankevich Andrey, MIPT <stankevich.as@phystech.edu>
"""


import numpy as np
import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from quantizer import VectorQuantizer


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
		self.quantizer = VectorQuantizer(self.n_emb, self.embedding_dim)
		self.decoder   = Decoder(n_init=self.n_emb, n_hid=self.n_hid)


	def encode(self, x):

		enc = self.encoder(x)
		quant, diff, ids = self.quantizer(enc.permute(0, 2, 3, 1))
		quant = quant.permute(0, 3, 1, 2)

		return quant, diff, ids

	def decode(self, x): return self.decoder(x)

	def forward(self, x):

		quant, diff, _ = self.encode(x)
		print(quant.shape)
		return self.decode(quant), diff