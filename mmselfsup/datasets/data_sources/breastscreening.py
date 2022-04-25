import os.path as osp
import pickle

import numpy as np
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..builder import DATASOURCES
from ..utils import check_integrity, download_and_extract_archive
from .base import BaseDataSource
from .breast_util import load_mammogram_img, read_h5
from PIL import Image

@DATASOURCES.register_module()
class NYUBreastScreening(BaseDataSource):
	""" NYU breast cancer screening datasets:
	 consists of ffdm and ultrasound exams acuqired on the same day. 
		
	us_select_index_logic 
		choices={"all", ("random", int), ("first", int)}
		  "all" selects all exams
		  ("random", N) selects N exams at random
		  ("first", N) selects the first N exams at random 
		
	"""
	ffdm_crop_size = (2944, 1920) 
	# channel=3 if use ImageNet pretrained weights
	us_select_index_logic = ('random', 50)
	# Use self.color_type insteadly 
	# # channel=3 if use ImageNet pretrained weights
	# us_shape = (3, 256, 256)

	def load_ffdm_img(self, img_prefix, accession_number, view, index, 
		filename, best_center, horizontal_flip, crop_method):
		"""Load a ffdm img.
		Args:
			img_prefix
			filename
			horizontal_flip
			best_center
			view
			crop_method

		Return:

		"""

		# step #1: load image np

		img_path = osp.join(img_prefix, filename + ".hdf5")

		# option 1: load best center
		if crop_method == "best_center":
			img_np = load_mammogram_img(
				img_path, self.ffdm_crop_size, view, best_center, horizontal_flip)
		elif crop_method == "no_crop":
			img_np = read_h5(img_path)

		# step #2: transform to pil image 
		return Image.fromarray(img_np / img_np.max())

	def load_us_slices(self, img_prefix, accession_number, indices, filenames):
		"""Load Ultrasound image slices.
		Args:
			image_prefix
			acn
			filenames

			we use self.color_type to determine 
			the number of channels of the us images. 
		Return:

		"""
		if self.us_select_index_logic == "all":
			pass
		elif isinstance(self.us_select_index_logic, (list, tuple)) and self.us_select_index_logic[1] < len(indices):
			method, count = self.us_select_index_logic	    
			if method == "first":
				indices = indices[:count]
			elif method == "random":
				indices = np.random.choice(indices, size=count, replace=False)
			else:
				raise ValueError(f"invalid select_index_logic = {self.us_select_index_logic}")

		elif isinstance(self.us_select_index_logic, (list, tuple)) and self.us_select_index_logic[0] in ("random", "first"):
			pass
		else:
			raise ValueError(f"invalid select_index_logic = {self.us_select_index_logic}")

		output = []
		for idx in indices:
			img_np = np.load(osp.join(img_prefix, accession_number, f"{idx}.npy"))
			img_pil = Image.fromarray(img_np.astype("uint8")).convert(
				"RGB" if self.color_type=='color' else "L") 
			output.append(img_pil)

		return output

	def load_annotations(self):
		"""Load datalist pickle file.

		"""
		if not self.test_mode:
			with open(osp.join(self.data_prefix, 'train'), "rb") as f:
				data_infos = pickle.load(f)
		else:
			with open(osp.join(self.data_prefix, 'val'), "rb") as f:
				data_infos = pickle.load(f)

		return data_infos

	def get_sample(self, idx):
		"""Get paired FFDM image and US slices by index.

		Args:
			idx (int): Index of data.
		Returns:
			FFDM image:  PIL image normalized 
			US slices: list of PIL images 
		"""

		# loal FFDM:
		assert self.data_infos[idx].get('ffdm', None) is not None
		img_ffdm = self.load_ffdm_img(**self.data_infos[idx]['ffdm'])

		# loal US:
		assert self.data_infos[idx].get('us', None) is not None
		imgs_us = self.load_us_slices(**self.data_infos[idx]['us'])

		return img_ffdm, imgs_us

	def get_img(self, idx, modal=None):
		assert modal is not None 
		if modal=='ffdm':
			return self.load_ffdm_img(**self.data_infos[idx]['ffdm'])
		elif modal=='us':
			return self.load_us_slices(**self.data_infos[idx]['us'])

	def get_gt_labels(self):
		"""Get all ground-truth labels (categories).

		Returns:
		    list[int]: categories for all images.
		"""

		gt_labels = np.array([data['malignant'] \
			for data in self.data_infos])
		return gt_labels

	def get_token_labels(self):

		def one_hot_encoder(label_indices):
			temp = [0] * 371
			for index in label_indices:
				temp[index] = 1
			return np.array(temp, dtype=np.long)

		noisy_token_labels = np.stack(
			(one_hot_encoder(data['noisy_token_indicies']) \
				for data in self.data_infos)
			)
		return noisy_token_labels

	def get_biopsied_labels(self):
		"""Get all labels regarding whether the sample is biopsed.

		Returns:
		    list[int]: categories for all images.
		"""

		gt_labels = np.array([data['biopsied'] \
			for data in self.data_infos])
		return gt_labels

	