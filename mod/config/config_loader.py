# -*- coding: utf-8 -*-
"""
Created on 2019/11/22 上午9:24

@Project -> File: ode-neural-network -> config_loader.py

@Author: luolei

@Describe: project params config loader
"""

import lake.decorator
import lake.dir
import logging
import logging.config
import yaml
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '../config/'))

if len(logging.getLogger().handlers) == 0:
	logging.basicConfig(level = logging.DEBUG)


@lake.decorator.singleton
class ConfigLoader(object):
	def __init__(self):
		self._set_proj_dir()
		self._set_proj_cmap()
		self._load_config()
		self._load_env_config()
		self._load_test_params_config()
		
	def _absolute_path(self, path):
		return os.path.join(os.path.dirname(__file__), path)
	
	def _set_proj_dir(self):
		"""
		Set root dir path.
		"""
		self._proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
	
	def _set_proj_cmap(self):
		"""
		Set color map.
		"""
		self._proj_cmap = {
			'blue': '#1f77b4',
			'orange': '#ff7f0e',
			'green': '#2ca02c',
			'red': '#d62728',
			'purple': '#9467bd',
			'cyan': '#17becf',
			'grey': '#7f7f7f',
			'black': 'k'
		}
	
	def _load_config(self):
		"""Load params config from config.yml"""
		self._config_path = os.path.join(self.proj_dir, 'config/config.yml')
		with open(self._config_path, 'r', encoding = 'utf-8') as f:
			self._conf = yaml.load(f, Loader = yaml.Loader)  # yaml.FullLoader
	
	def _load_env_config(self):
		"""
		Load env params config.
		"""
		config_dir_ = os.path.join(self.proj_dir, 'config/')
		
		# Primarily use params from master.yml if the file exists, else use default.yml, default empty.
		if 'master.yml' in os.listdir(config_dir_):
			print('Use env variables in master.yml.')
			env_config_path_ = os.path.join(config_dir_, 'master.yml')
		else:
			if 'default.yml' in os.listdir(config_dir_):
				print('Use env variables in default.yml.')
				env_config_path_ = os.path.join(config_dir_, 'default.yml')
			else:
				env_config_path_ = None
		
		if env_config_path_ is None:
			self._local_env_conf = {}
		else:
			with open(env_config_path_, 'r', encoding = 'utf-8') as f:
				self._local_env_conf = yaml.load(f, Loader = yaml.Loader)
		
		# Inject online params if available, else use the values from the config files.
		self._env_conf = {}
		if self._local_env_conf is None:
			pass
		else:
			for key in self._local_env_conf.keys():
				if key in os.environ:
					self._env_conf.update({key: os.environ[key]})
				else:
					self._env_conf.update({key: self._local_env_conf[key]})
				
	def _load_test_params_config(self):
		"""
		Load testing params config.
		"""
		config_dir_ = os.path.join(self.proj_dir, 'config/')
		test_config_path_ = os.path.join(config_dir_, 'test_params.yml')
		with open(test_config_path_, 'r', encoding = 'utf-8') as f:
			self._test_params = yaml.load(f, Loader = yaml.Loader)
	
	@property
	def proj_dir(self):
		return self._proj_dir
	
	@property
	def proj_cmap(self):
		return self._proj_cmap
	
	@property
	def conf(self):
		return self._conf
	
	@property
	def env_conf(self):
		return self._env_conf
	
	@property
	def test_params(self):
		return self._test_params
	
	def set_logging(self):
		"""
		Set logging files
		"""
		if 'logs' not in os.listdir(self.proj_dir):
			os.mkdir(os.path.join(self.proj_dir, 'logs/'))
		log_config = self.conf['logging']
		logging.config.dictConfig(log_config)


def update_filename(log_config):
	"""
	Update file name.
	:param log_config: dict, logs config
	"""
	to_log_path = lambda x: os.path.abspath(os.path.join(os.path.dirname(__file__), '../', x))
	if 'filename' in log_config:
		log_config['filename'] = to_log_path(log_config['filename'])
	for key, value in log_config.items():
		if isinstance(value, dict):
			update_filename(value)


config = ConfigLoader()
