"""
Author: Stankevich Andrey, MIPT <stankevich.as@phystech.edu>
Borrowed from https://github.com/rosinality/vq-vae-2-pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class VectorQuantizer(nn.Module):

	def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):


		super().__init__()

		self.dim     = dim
		self.n_embed = n_embed
		self.decay   = decay
		self.eps     = eps

		embedding = torch.randn(dim, n_embed)
		self.register_buffer("embedding", embedding)
		self.register_buffer("cluster_size", torch.zeros(n_embed))
		self.register_buffer("embedding_avg", embedding.clone())


	def forward(self, z):

		flatten = z.reshape(-1, self.dim)

		dist = flatten.pow(2).sum(1, keepdim=True) -\
			   2 * flatten @ self.embedding  +\
			   self.embedding.pow(2).sum(0, keepdim=True)

		_, embed_ind = (-dist).max(1)
		embed_onehot  = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)

		embed_ind = embed_ind.view(*z.shape[:-1])
		quantize  = F.embedding(embed_ind, self.embedding.transpose(0, 1))

		if self.training:
			embed_onehot_sum = embed_onehot.sum(0)
			embed_sum        = flatten.transpose(0, 1) @ embed_onehot

			self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1-self.decay)
			self.embedding_avg.data.mul_(self.decay).add_(embed_sum, alpha=1-self.decay)

			
			n = self.cluster_size.sum()

			cluster_size = (
				(self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
			)

			embed_normalized = self.embedding_avg / cluster_size.unsqueeze(0)

			self.embedding.data.copy_(embed_normalized)

		diff     = (quantize.detach() - z).pow(2).mean()
		quantize = z + (quantize - z).detach()


		return quantize, diff, embed_ind 


	def embed_code(self, embed_id):

		return F.embedding(embed_id, self.embedding.transpose(0, 1))

