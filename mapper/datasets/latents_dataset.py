import torch
from torch.utils.data import Dataset


class LatentsDataset(Dataset):

	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):

		return self.latents[index]

class StyleSpaceLatentsDataset(Dataset):

	def __init__(self, latents, opts):
		padded_latents = []
		for latent in latents:
			latent = latent.cpu()
			if latent.shape[2] == 512:
				padded_latents.append(latent)
			else:
				padding = torch.zeros((latent.shape[0], 1, 512 - latent.shape[2], 1, 1))
				padded_latent = torch.cat([latent, padding], dim=2)
				padded_latents.append(padded_latent)
		self.latents = torch.cat(padded_latents, dim=2)
		self.opts = opts

	def __len__(self):
		return len(self.latents)

	def __getitem__(self, index):
		return self.latents[index]
