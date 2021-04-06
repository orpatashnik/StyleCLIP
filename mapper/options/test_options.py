from argparse import ArgumentParser


class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		# arguments for inference script
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to model checkpoint')
		self.parser.add_argument('--couple_outputs', action='store_true', help='Whether to also save inputs + outputs side-by-side')

		self.parser.add_argument('--mapper_type', default='LevelsMapper', type=str, help='Which mapper to use')
		self.parser.add_argument('--no_coarse_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_medium_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_fine_mapper', default=False, action="store_true")
		self.parser.add_argument('--stylegan_size', default=1024, type=int)


		self.parser.add_argument('--test_batch_size', default=2, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--latents_test_path', default=None, type=str, help="The latents for the validation")
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--n_images', type=int, default=None, help='Number of images to output. If None, run on all data')

	def parse(self):
		opts = self.parser.parse_args()
		return opts