# -*- coding: utf-8 -*-
"""
Created on 2020/5/29

@File: __init__

@Department: AI Lab, Rockontrol, Chengdu

@Author: Luo Lei

@Email: dreisteine262@163.com

@Describe: initialize and set config
"""

import logging

logging.basicConfig(level = logging.INFO)

import sys
import os

sys.path.append('../')

from mod.config.config_loader import config

proj_dir, proj_cmap = config.proj_dir, config.proj_cmap

env_params = config.env_conf
model_params = config.conf
test_params = config.test_params