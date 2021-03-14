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

from PIL import Image
from typing import Optional


class customOmniglot(datasets.Omniglot):
	
	folder = 'omniglot-py'
	download_url_prefix = 'https://github.com/brendenlake/omniglot/raw/master/python'
	zips_md5 = {
		'images_background': '68d2efa1b9178cc56df9314c21c6e718',
		'images_evaluation': '6b91aef0f799c5bb55b94e3f2daec811'
	}

	def __init__(self, root: str, background: bool = True, transform: Optional[Callable] = None, target_transform: Optional[Callable] = None, download: bool = False, mask_mode = False) -> None:
		super(Omniglot, self).__init__(join(root, self.folder), transform=transform,
									   target_transform=target_transform)
		self.background = background
		self.mask_mode = mask_mode

		if download:
			self.download()

		if not self._check_integrity():
			raise RuntimeError('Dataset not found or corrupted.' +
							   ' You can use download=True to download it')

		self.target_folder = join(self.root, self._get_target_folder())
		self._alphabets = list_dir(self.target_folder)
		self._characters: List[str] = sum([[join(a, c) for c in list_dir(join(self.target_folder, a))]
										   for a in self._alphabets], [])
		self._character_images = [[(image, idx) for image in list_files(join(self.target_folder, character), '.png')]
								  for idx, character in enumerate(self._characters)]
		self._flat_character_images: List[Tuple[str, int]] = sum(self._character_images, [])

		self.label_data = []
		self.label_target = []

		if self.mask_mode == False:
			#Load only 1 label
			for i, (d,l) in enumerate(zip(self.data,self.targets)):
				if(int(l) == self.label):
					self.label_data.append(d)
					self.label_target.append(l)

			print("LabelOmniglot {}".format(self.label))

		else:
			#Load N-1 labels
			for i, (d,l) in enumerate(zip(self.data,self.targets)):
				if(int(l) != self.label):
					self.label_data.append(d)
					self.label_target.append(l)

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
		image_name, character_class = self._flat_character_images[index]
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
