import matplotlib.pyplot as plt
import numpy as np
import torch

from assignment3.dataset import move_data_to_device


class Visualizer:
	def __init__(self, model, data_loader, device):
		self.model = model
		self.data_loader = data_loader
		self.device = device

	def visualize_results(self):
		"""
		Visualize the model's predictions and the ground truth annotations for a segment of audio.

		Args:
		- model: The trained model.
		- data_loader: DataLoader object for loading data.
		- device: The device on which the model and data are loaded (CPU or GPU).
		"""
		self.model.eval()  # Set the model to evaluation mode

		# Fetch one batch of data
		batch = next(iter(self.data_loader))
		x, onset_gt, offset_gt, octave_gt, pitch_class_gt = move_data_to_device(batch, self.device)

		# Get model predictions
		with torch.no_grad():
			onset_pred, offset_pred, octave_pred, pitch_class_pred = self.model(x)

		# Convert predictions to CPU for visualization
		onset_pred = torch.sigmoid(onset_pred).cpu().numpy()
		offset_pred = torch.sigmoid(offset_pred).cpu().numpy()
		octave_pred = torch.argmax(octave_pred, dim=2).cpu().numpy()
		pitch_class_pred = torch.argmax(pitch_class_pred, dim=2).cpu().numpy()

		# Convert ground truths to CPU for visualization
		onset_gt = onset_gt.cpu().numpy()
		offset_gt = offset_gt.cpu().numpy()
		octave_gt = octave_gt.cpu().numpy()
		pitch_class_gt = pitch_class_gt.cpu().numpy()

		# Select one example from the batch for visualization
		example_idx = 0  # Can be changed to visualize other examples
		time_steps = np.arange(x.shape[2])  # Assuming time dimension is at axis 2

		# Plotting the results
		plt.figure(figsize=(15, 10))

		plt.subplot(4, 1, 1)
		plt.plot(time_steps, onset_gt[example_idx], label='Ground Truth')
		plt.plot(time_steps, onset_pred[example_idx], label='Prediction', linestyle='--')
		plt.title('Onset')
		plt.xlabel('Time Steps')
		plt.ylabel('Onset Probability')
		plt.legend()

		plt.subplot(4, 1, 2)
		plt.plot(time_steps, offset_gt[example_idx], label='Ground Truth')
		plt.plot(time_steps, offset_pred[example_idx], label='Prediction', linestyle='--')
		plt.title('Offset')
		plt.xlabel('Time Steps')
		plt.ylabel('Offset Probability')
		plt.legend()

		plt.subplot(4, 1, 3)
		plt.plot(time_steps, octave_gt[example_idx], label='Ground Truth')
		plt.plot(time_steps, octave_pred[example_idx], label='Prediction', linestyle='--')
		plt.title('Octave')
		plt.xlabel('Time Steps')
		plt.ylabel('Octave Class')
		plt.legend()

		plt.subplot(4, 1, 4)
		plt.plot(time_steps, pitch_class_gt[example_idx], label='Ground Truth')
		plt.plot(time_steps, pitch_class_pred[example_idx], label='Prediction', linestyle='--')
		plt.title('Pitch Class')
		plt.xlabel('Time Steps')
		plt.ylabel('Pitch Class')
		plt.legend()

		plt.tight_layout()
		plt.show()
