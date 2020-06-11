# -*- coding: utf-8 -*-
"""
Created on 2020/6/11 4:32 下午

@File: show_figure.py

@Department: AI Lab, Rockontrol, Chengdu

@Author: luolei

@Email: dreisteine262@163.com

@Describe:
"""

import logging

logging.basicConfig(level = logging.INFO)

from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import pandas as pd
import sys, os


def show_all_fields(data: pd.DataFrame, legend_label: str = 'index', overlap: bool = False, fig: Figure = None,
					return_fig: bool = False):
	"""
	显示数据中所有除了'time'以外其他字段值变化
	"""
	try:
		assert data.columns[0] == 'time'
	except:
		raise ValueError('The first col is not "time"')

	fields = list(data.columns)[1:]

	if not overlap:
		fig = plt.figure(figsize = [8, 12])

	for field in fields:
		if not overlap:
			ax = fig.add_subplot(len(fields) // 3 + 1, 3, fields.index(field) + 1)
		else:
			ax = fig.axes[fields.index(field)]

		ax.plot(list(data[field]), linewidth = 0.3)

		if not overlap:
			plt.xticks(fontsize = 6)
			plt.yticks(fontsize = 6)

			if legend_label == 'index':
				plt.legend(['field no. {}'.format(fields.index(field))], loc = 'upper right', fontsize = 6)
			elif legend_label == 'field':
				plt.legend([field], loc = 'lower right', fontsize = 6)
			else:
				raise ValueError('legend_label not in {"index", "field"}')

	plt.tight_layout()

	if return_fig:
		return fig
	else:
		pass



