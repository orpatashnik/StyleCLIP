from torch.utils.data import Dataset


class LatentsDataset(Dataset):

	def __init__(self, latents, opts):
		self.latents = latents
		self.opts = opts

	def __len__(self):
		return self.latents.shape[0]

	def __getitem__(self, index):

		return self.latents[index]
