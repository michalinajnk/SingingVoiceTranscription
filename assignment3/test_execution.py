from tqdm import tqdm
from dataset import get_data_loader, move_data_to_device
from hparams import Hparams




if __name__ == "__main__":
	train_loader = get_data_loader(split='train', args=Hparams.args)
	for data in tqdm(train_loader):
		x, onset, offset, octave, pitch_class = move_data_to_device(data, 'cuda')
		assert list(x.shape) == [8, 250, 256]
		assert list(onset.shape) == list(offset.shape) == list(octave.shape) == list(pitch_class.shape) == [8, 250]
		break
	print('Congrats!')
