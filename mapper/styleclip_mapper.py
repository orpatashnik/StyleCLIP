import torch
from torch import nn
from mapper import latent_mappers
from models.stylegan2.model import Generator


def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class StyleCLIPMapper(nn.Module):

	def __init__(self, opts):
		super(StyleCLIPMapper, self).__init__()
		self.opts = opts
		# Define architecture
		self.mapper = self.set_mapper()
		self.decoder = Generator(self.opts.stylegan_size, 512, 8)
		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))
		# Load weights if needed
		self.load_weights()

	def set_mapper(self):
		if self.opts.mapper_type == 'SingleMapper':
			mapper = latent_mappers.SingleMapper(self.opts)
		elif self.opts.mapper_type == 'LevelsMapper':
			mapper = latent_mappers.LevelsMapper(self.opts)
		else:
			raise Exception('{} is not a valid mapper'.format(self.opts.mapper_type))
		return mapper

	def load_weights(self):
		if self.opts.checkpoint_path is not None:
			print('Loading from checkpoint: {}'.format(self.opts.checkpoint_path))
			ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
			self.mapper.load_state_dict(get_keys(ckpt, 'mapper'), strict=True)
			self.decoder.load_state_dict(get_keys(ckpt, 'decoder'), strict=True)
		else:
			print('Loading decoder weights from pretrained!')
			ckpt = torch.load(self.opts.stylegan_weights)
			self.decoder.load_state_dict(ckpt['g_ema'], strict=False)

	def forward(self, x, resize=True, latent_mask=None, input_code=False, randomize_noise=True,
	            inject_latent=None, return_latents=False, alpha=None):
		if input_code:
			codes = x
		else:
			codes = self.mapper(x)

		if latent_mask is not None:
			for i in latent_mask:
				if inject_latent is not None:
					if alpha is not None:
						codes[:, i] = alpha * inject_latent[:, i] + (1 - alpha) * codes[:, i]
					else:
						codes[:, i] = inject_latent[:, i]
				else:
					codes[:, i] = 0

		input_is_latent = not input_code
		images, result_latent = self.decoder([codes],
		                                     input_is_latent=input_is_latent,
		                                     randomize_noise=randomize_noise,
		                                     return_latents=return_latents)

		if resize:
			images = self.face_pool(images)

		if return_latents:
			return images, result_latent
		else:
			return images
