'''
This file is only used in the ipynb file.
'''

import os
import sys
import torch

class Hparams:
    args = {
        
        'save_model_dir': r'clr1e-3',
        'device':  'cuda',
        'dataset_root': r'/content/SingingVoiceTranscription/assignment3/data_mini',
        'sampling_rate': 16000, # Please keep the sampling rate unchanged
        'sample_length': 5,  # in second
        'num_workers': 8,  # Number of additional thread for data loading. When running on laptop, set to 0.
        'annotation_path': r'/content/SingingVoiceTranscription/assignment3/data_mini/annotations.json',
        'frame_size': 0.02,
        'batch_size': 16,
    }
