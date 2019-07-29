import os
class Config(object):
	def __init__(self):
		self.num_epochs = 150
		self.init_lr = 1e-4
		self.batch_size = 2

		self.dpi=(72.0, 72.0)

		self.n_label = 16

		self.train_path = 'dataset/train'
		self.train_data_path = 'dataset/train_data'
		self.val_path = 'dataset/val'
		self.test_path = 'dataset/test'
		self.seed = 7
		self.augment = True



		self.size_train = (400, 400)

		# image channel-wise mean to subtract, the order is BGR
		self.img_channel_mean = [103.939, 116.779, 123.68]

	def check_folder(self, log_dir):
		if not os.path.exists(log_dir):
			os.makedirs(log_dir)
		return log_dir