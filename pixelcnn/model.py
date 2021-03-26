"""
Author: Stankevich Andrey, MIPT <stankevich.as@phystech.edu>
"""

import torch
import torch.nn as nn
from modules import (
	MaskedConv2d,
	GatedResnet,
	DownShiftedConv2d,
	DownRightShiftedConv2d,
	down_shift,
	right_shift
)


class PixelCNN(nn.Module):

	def __init__(self):

		super().__init__()

		self.n_layers = 8
		self.hid_size = 64

		layers = [
			MaskedConv2d('A', 3, self.hid_size, kernel_size=7, stride=1, padding=3),
			nn.BatchNorm2d(self.hid_size),
			nn.CELU()
		]

		for _ in range(1, self.n_layers):
			layers.extend([
				MaskedConv2d('B', self.hid_size, self.hid_size, kernel_size=3, stride=1, padding=1),
				nn.BatchNorm2d(self.hid_size),
				nn.CELU()
			])

		self.rc = nn.Conv2d(self.hid_size, 256, 1)
		self.gc = nn.Conv2d(self.hid_size, 256, 1)
		self.bc = nn.Conv2d(self.hid_size, 256, 1)

		self.pixelcnn = nn.Sequential(*layers)


	def forward(self, x): 
		x = self.pixelcnn(x)
		return self.rc(x), self.gc(x), self.bc(x)


class GatedPixelCNN(nn.Module):

	def __init__(self):

		super().__init__()

		self.hid_size = 64
		self.blocks   = nn.ModuleList([GatedResnet(self.hid_size) for _ in range(6)])

		self.conv_v  = MaskedConv2d('A', 3, self.hid_size, kernel_size=7, padding=3)
		self.conv_h  = MaskedConv2d('A', 3, self.hid_size, kernel_size=7, padding=3)
		
		self.nonlinearity = nn.CELU()

		self.rc = nn.Conv2d(self.hid_size, 256, 1)
		self.bc = nn.Conv2d(self.hid_size, 256, 1)
		self.gc = nn.Conv2d(self.hid_size, 256, 1)
	

	def forward(self, x):

		v_out = self.conv_v(x)
		h_out = self.conv_h(x)

		skip_connections = []

		for b in self.blocks:
			h_out, v_out, skip = b(h_out, v_out)
			skip_connections.append(skip)

		skip_output = 0.
		for value in skip_connections: skip_output += value

		skip_output = self.nonlinearity(skip_output)
		r, g, b = self.rc(skip_output), self.gc(skip_output), self.bc(skip_output)
		return r, g, b
