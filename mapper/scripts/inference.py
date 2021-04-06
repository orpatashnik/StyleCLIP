import os
from argparse import Namespace

import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import time

from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from mapper.datasets.latents_dataset import LatentsDataset

from mapper.options.test_options import TestOptions
from mapper.styleclip_mapper import StyleCLIPMapper


def run(test_opts):
	out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
	os.makedirs(out_path_results, exist_ok=True)

	# update test options with options used during training
	ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
	opts = ckpt['opts']
	opts.update(vars(test_opts))
	opts = Namespace(**opts)

	net = StyleCLIPMapper(opts)
	net.eval()
	net.cuda()

	test_latents = torch.load(opts.latents_test_path)
	dataset = LatentsDataset(latents=test_latents.cpu(),
										 opts=opts)
	dataloader = DataLoader(dataset,
	                        batch_size=opts.test_batch_size,
	                        shuffle=False,
	                        num_workers=int(opts.test_workers),
	                        drop_last=True)

	if opts.n_images is None:
		opts.n_images = len(dataset)
	
	global_i = 0
	global_time = []
	for input_batch in tqdm(dataloader):
		if global_i >= opts.n_images:
			break
		with torch.no_grad():
			input_cuda = input_batch.cuda().float()
			tic = time.time()
			result_batch = run_on_batch(input_cuda, net, test_opts.couple_outputs)
			toc = time.time()
			global_time.append(toc - tic)

		for i in range(opts.test_batch_size):
			im_path = str(global_i).zfill(5)
			if test_opts.couple_outputs:
				couple_output = torch.cat([result_batch[2][i].unsqueeze(0), result_batch[0][i].unsqueeze(0)])
				torchvision.utils.save_image(couple_output, os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
			else:
				torchvision.utils.save_image(result_batch[0][i], os.path.join(out_path_results, f"{im_path}.jpg"), normalize=True, range=(-1, 1))
			torch.save(result_batch[1][i].detach().cpu(), os.path.join(out_path_results, f"latent_{im_path}.pt"))

			global_i += 1

	stats_path = os.path.join(opts.exp_dir, 'stats.txt')
	result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
	print(result_str)

	with open(stats_path, 'w') as f:
		f.write(result_str)


def run_on_batch(inputs, net, couple_outputs=False):
	w = inputs
	with torch.no_grad():
		w_hat = w + 0.1 * net.mapper(w)
		x_hat, w_hat = net.decoder([w_hat], input_is_latent=True, return_latents=True, randomize_noise=False, truncation=1)
		result_batch = (x_hat, w_hat)
		if couple_outputs:
			x, _ = net.decoder([w], input_is_latent=True, randomize_noise=False, truncation=1)
			result_batch = (x_hat, w_hat, x)
	return result_batch


if __name__ == '__main__':
	test_opts = TestOptions().parse()
	run(test_opts)
