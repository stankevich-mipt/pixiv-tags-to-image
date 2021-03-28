"""
Author: Stankevich Andrey, MIPT <stankevich.as@phystech.edu>
Implementation is mostly copied from openai DALL-E open-source
repository https://github.com/openai/DALL-E
"""

import torch
import torch.nn as nn

from collections import OrderedDict
from functools import partial


class DecoderBlock(nn.Module):

	def __init__(
		self,
		n_in,
		n_out,
		n_layers
	):

		super().__init__()

		self.n_in  = n_in
		self.n_out = n_out
		self.n_hid = self.n_out // 4

		self.n_layers  = n_layers
		self.post_gain = 1. / float(self.n_layers ** 2)  

		self.id_path = nn.Conv2d(self.n_in, self.n_out, 1) \
			if (self.n_in != self.n_out) else nn.Identity()

		self.res_path = nn.Sequential(
			OrderedDict([
				('relu_1', nn.ReLU()),
				('conv_1', nn.Conv2d(self.n_in, self.n_hid, kernel_size=3, dilation=2, padding=2)),
				('relu_2', nn.ReLU()),
				('conv_2', nn.Conv2d(self.n_hid, self.n_hid, kernel_size=3, dilation=2, padding=2)),
				('relu_3', nn.ReLU()),
				('conv_3', nn.Conv2d(self.n_hid, self.n_hid, kernel_size=3, dilation=2, padding=2)),
				('relu_4', nn.ReLU()),
				('conv_4', nn.Conv2d(self.n_hid, self.n_out, kernel_size=3, dilation=2, padding=2))				
			]))


	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.id_path(x) + self.post_gain * self.res_path(x) 


class Decoder(nn.Module):

	def __init__(
		self, 
		n_init=256,
		n_hid=256,
		n_block_per_group=2,
		input_channels=3,
		output_channels=3,
		
	):

		super().__init__()

		self.group_count       = 4
		self.n_init            = n_init
		self.n_hid             = n_hid
		self.n_block_per_group = n_block_per_group
		self.input_channels    = input_channels
		self.output_channels   = output_channels

		br       = range(self.n_block_per_group)
		n_layers = self.group_count * self.n_block_per_group
		make_b   = partial(DecoderBlock, n_layers=n_layers)

		self.blocks = nn.Sequential(
			OrderedDict([
				('input', nn.Conv2d(self.n_init, self.n_init, 1)),
				('group_1', nn.Sequential(
					OrderedDict([
						*[(f'block_{i + 1}', make_b(
							self.n_init if  i == 0 else 8 * self.n_hid, 8 * self.n_hid)) for i in br
						],
						('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
					])
				)),
				('group_2', nn.Sequential(
					OrderedDict([
						*[(f'block_{i + 1}', make_b(
							8 * self.n_hid if  i == 0 else 4 * self.n_hid, 4 * self.n_hid)) for i in br
						],
						('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
					])
				)),
				('group_3', nn.Sequential(
					OrderedDict([
						*[(f'block_{i + 1}', make_b(
							4 * self.n_hid if  i == 0 else 2 * self.n_hid, 2 * self.n_hid)) for i in br
						],
						('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
					])
				)),
				('group_4', nn.Sequential(
					OrderedDict([
						*[(f'block_{i + 1}', make_b(
							2 * self.n_hid if  i == 0 else 1 * self.n_hid, 1 * self.n_hid)) for i in br
						],
						('upsample', nn.Upsample(scale_factor=2, mode='nearest'))
					])
				)),
				('output', nn.Sequential(
					OrderedDict([
						('relu', nn.ReLU()),
						('conv', nn.Conv2d(1 * self.n_hid, self.output_channels, 1))
					])
				))
			]))

	def forward(self, x: torch.Tensor) -> torch.Tensor:

		if len(x.shape) != 4: raise ValueError(f'input shape {x.shape} is not 4d')
		if x.shape[1] != self.n_init: 
			raise ValueError(f'input has {x.shape[1]} channels but model built for {self.n_init}')
			
		return self.blocks(x)