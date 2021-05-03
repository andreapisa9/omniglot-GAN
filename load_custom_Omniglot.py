import os
import argparse
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from torchvision.utils import save_image
from torchvision import datasets, transforms
from torchvision.datasets.utils import download_and_extract_archive, check_integrity, list_dir, list_files

from PIL import Image
from typing import Optional, Callable, Tuple, List, Any
from os.path import join


class customOmniglot(datasets.Omniglot):
	
	folder = 'omniglot-py'
	download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
	zips_md5 = {
		'images_background': '68d2efa1b9178cc56df9314c21c6e718',
		'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
	}

	def __init__(self, root: str, label: int, num_labels: int, background: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, mask_mode: bool = False) -> None:
		super(datasets.Omniglot, self).__init__(join(root, self.folder), transform=transform,
									   target_transform=target_transform)
		self.label = label
		self.num_labels = num_labels
		self.background = background
		self.mask_mode = mask_mode

		if download:
			self.download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
							   ' You can use download=True to download it')

		#UNDERSTAND FULLY WHAT ARE THESE COMPOSED OF BEFORE MAKING A MESS
		self.target_folder = join(self.root, self._get_target_folder())
		self._alphabets = list_dir(self.target_folder)

		self._characters: List[str] = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
										   for a in self._alphabets], [])
		#print(self._characters)
		
		self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
								  for idx, character in enumerate(self._characters)]
		
		self._flat_character_images: List[Tuple[str, int]] = sum(self._character_images, [])
		
		self.label_data = []
		self.label_target = []

		if self.mask_mode == False:
			#Load only 1 label
			character_batch = self._character_images[self.label]
			for image_name, idx in character_batch:
				self.label_data.append(image_name)
				self.label_target.append(idx)

			#print(self.label_target)
			print("LabelOmniglot {}".format(self.label))

		else:
			#Load N-1 labels
			#TO FIX
			for i in range(self.num_labels):
				if i != self.label:
					character_batch = self._character_images[i]
					for image_name, idx in character_batch:
						self.label_data.append(image_name)
						self.label_target.append(idx)

			print("LabelOmniglot masked {}".format(self.label))

	def __len__(self) -> int:
		return len(self._flat_character_images)

	def __getitem__(self, index: int) -> Tuple[Any, Any]:
		"""
		Args:
			index (int): Index

		Returns:
			tuple: (image, target) where target is index of the target character class.
		"""
		image_name = self.label_data[self.label]
		character_class = self.label_target[self.label]
		image_path = join(self.target_folder, self._characters[character_class], image_name)
		image = Image.open(image_path, mode='r').convert('L')

		if self.transform:
			image = self.transform(image)

		if self.target_transform:
			character_class = self.target_transform(character_class)

		return image, character_class

	def _check_integrity(self) -> bool:
		zip_filename = self._get_target_folder()
		if not check_integrity(join(self.root, zip_filename + '.zip'), self.zips_md5[zip_filename]):
			return False
		return True

	def download(self) -> None:
		if self._check_integrity():
			print('Files already downloaded and verified')
			return

		filename = self._get_target_folder()
		zip_filename = filename + '.zip'
		url = self.download_url_prefix + '/' + zip_filename
		download_and_extract_archive(url, self.root, filename=zip_filename, md5=self.zips_md5[filename])

	def _get_target_folder(self) -> str:
		return 'images_background' if self.background else 'images_evaluation'
