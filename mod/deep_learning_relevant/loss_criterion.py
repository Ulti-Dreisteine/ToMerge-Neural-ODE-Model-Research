# -*- coding: utf-8 -*-
"""
Created on 2020/5/29

@File: loss_criterion

@Department: AI Lab, Rockontrol, Chengdu

@Author: Luo Lei

@Email: dreisteine262@163.com

@Describe: loss funcs
"""

import torch.nn.functional as f
from torch import nn
import torch


def smape(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
	"""
	SMAPE loss.
	:param y_true: torch.Tensor, true value
	:param y_pred: torch.Tensor, pred value
	:return: torch.tensor, smape
	"""
	numerator = torch.abs(y_pred - y_true)
	denominator = torch.div(torch.add(torch.abs(y_pred), torch.abs(y_true)), 2.0)
	return torch.mean(torch.abs(torch.div(numerator, denominator)))


class FocalLoss(nn.Module):
	def __init__(self):
		super(FocalLoss, self).__init__()

	def forward(self, y_true, y_pred):
		error = torch.norm(y_true - y_pred, dim = 1)
		p = f.softmax(error, dim = 0)
		loss = torch.sum(torch.mul(p, error))
		return loss


def criterion(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
	"""
	Loss criterion.
	:param y_true: torch.Tensor, true value
	:param y_pred: torch.Tensor, pred value
	:return: torch.tensor, loss
	"""
	# l1 = nn.L1Loss()
	# l2 = nn.MSELoss()
	# loss = l1(y_true, y_pred) + l2(y_true, y_pred)
	focal_loss = FocalLoss()
	loss = focal_loss(y_true, y_pred)
	return loss