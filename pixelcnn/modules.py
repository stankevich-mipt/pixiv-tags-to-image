"""
Author: Stankevich Andrey, MIPT <stankevich.as@phystech.edu>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
	def __init__(self, mask_type, *args, **kwargs):
		
		super().__init__(*args, **kwargs)
		assert mask_type in {'A', 'B'}
		
		self.register_buffer('mask', self.weight.data.clone())
		_, _, kH, kW = self.weight.size()
		self.mask.fill_(1.)
		self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0.
		self.mask[:, :, kH // 2 + 1:] = 0.

	def forward(self, x):
		self.weight.data *= self.mask
		return super().forward(x)

def down_shift(x):
	b, c, w, h = x.size()
	return torch.cat([torch.zeros(b, c, 1, h), x[:, :, :w-1, :]], dim=2)

def right_shift(x):
	b, c, w, h = x.size()
	return torch.cat([torch.zeros(b, c, w, 1), x[:, :, :, :h-1]], dim=3)


class DownShiftedConv2d(nn.Conv2d):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.shift_pad = nn.ConstantPad2d(
			(
				int((self.kernel_size[1] - 1) // 2),
				int((self.kernel_size[1] - 1) // 2),
				self.kernel_size[0] - 1, 0
			), 0.
		)

	def forward(self, x):
		x = self.shift_pad(x)
		return right_shift(super().forward(x))


class DownRightShiftedConv2d(nn.Conv2d):

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.shift_pad = nn.ConstantPad2d(
			(self.kernel_size[1] - 1, 0, self.kernel_size[0] - 1, 0), 0.)
	
	def forward(self, x):
		x = self.shift_pad(x)
		return down_shift(super().forward(x))


class GatedResnet(nn.Module):

	def __init__(
		self, 
		channels, 
		kernel_size=3,
		nonlinearity='celu', 
		**kwargs
	):

		super().__init__(**kwargs)

		self.channels    = channels
		self.kernel_size = kernel_size

		if nonlinearity == 'celu': 
			self.nonlinearity = nn.CELU()
		else:
			raise ValueError

		self.v_conv = DownShiftedConv2d(channels, 2 * channels, kernel_size=kernel_size)
		self.v_to_h = nn.Conv2d(2 * channels, 2 * channels, kernel_size=1)
		self.v_fc   = nn.Conv2d(channels, channels, kernel_size=1)

		self.h_conv = DownRightShiftedConv2d(channels, 2 * channels, kernel_size=kernel_size)
		self.h_fc   = nn.Conv2d(channels, channels, kernel_size=1)
		self.h_skip = nn.Conv2d(channels, channels, kernel_size=1)

	def forward(self, h, v):

		h_in = self.nonlinearity(h)
		v_in = self.nonlinearity(v)

		v_out  = self.nonlinearity(self.v_conv(v_in))
		v_to_h = self.nonlinearity(self.v_to_h(v_out))
		
		# gated activation for vertical stack 
		v_out_tanh, v_out_sigma = torch.split(v_out, self.channels, dim=1)
		v_out = torch.tanh(v_out_tanh) * torch.sigmoid(v_out_sigma)
		v_out_ = self.nonlinearity(self.v_fc(v_out))
		v_out_ += v

		# gated activation for horizontal stack
		h_out  = self.nonlinearity(self.h_conv(h_in))
		h_out += v_to_h
		h_out_tanh, h_out_sigma = torch.split(h_out, self.channels, dim=1)
		h_out  = torch.tanh(h_out_tanh) * torch.sigmoid(h_out_sigma)
		h_out_ = self.nonlinearity(self.h_fc(h_out))
		h_skip = self.nonlinearity(self.h_skip(h_out))
		h_out_ += h

		return h_out_, v_out_, h_skip