import math
import torch
import torchaudio
from torch.utils.data import Dataset

from utils import read_json, ls, jpath

class MyDataset(Dataset):
    def __init__(self, dataset_root, split, sampling_rate, annotation_path, sample_length, frame_size, song_fns=None):
        '''
        This dataset returns an audio clip of a specific duration in the training loop, with its "__getitem__" function.
        '''
        self.dataset_root = dataset_root
        self.split = split
        self.dataset_path = jpath(self.dataset_root, self.split)
        self.sampling_rate = sampling_rate
        self.annotation_path = annotation_path
        self.all_annotations = read_json(self.annotation_path)
        self.duration = {}
        if song_fns is None:
            self.song_fns = ls(self.dataset_path)
            self.song_fns.sort()
        else:
            self.song_fns = song_fns
        self.index = self.index_data(sample_length)
        self.sample_length = sample_length
        self.frame_size = frame_size
        self.frame_per_sec = int(1 / self.frame_size)
        self.audio_cache = {}  # Cache for audio data
        self.annotation_cache = {}  # Cache for annotations

    def index_data(self, sample_length):
        '''
        Prepare the index for the dataset, i.e., the audio file name and starting time of each sample
        '''
        index = []
        for song_fn in self.song_fns:
            if song_fn.startswith('.'):  # Ignore any hidden file
                continue
            duration = self.all_annotations[song_fn][-1][1]
            num_seg = math.ceil(duration / sample_length)
            for i in range(num_seg):
                index.append([song_fn, i * sample_length])
            self.duration[song_fn] = duration
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        '''
        Return spectrogram and 4 labels of an audio clip
        The audio filename and the start time of this sample is specified by "audio_fn" and "start_sec"
        '''
        audio_fn, start_sec = self.index[idx]
        end_sec = start_sec + self.sample_length

        # Check if the audio and annotations are already cached
        if audio_fn in self.audio_cache:
            audio = self.audio_cache[audio_fn]
            sr = self.sampling_rate  # Audio is already resampled
        else:
            audio_fp = jpath(self.dataset_path, audio_fn, 'Mixture.mp3')
            audio, sr = torchaudio.load(audio_fp)
            if audio.shape[0] > 1:  # Convert stereo to mono
                audio = torch.mean(audio, dim=0, keepdim=True)
            if sr != self.sampling_rate:
                resample = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sampling_rate)
                audio = resample(audio)
            self.audio_cache[audio_fn] = audio  # Cache the loaded and processed audio

        # Calculate the start and end frames for the audio clip
        start_frame = int(start_sec * self.sampling_rate)
        end_frame = int(end_sec * self.sampling_rate)
        audio_clip = audio[:, start_frame:end_frame]

        # Compute mel spectrogram
        hop_length = int(self.sampling_rate * self.frame_size)
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sampling_rate,
            n_fft=2048,
            hop_length=hop_length,
            n_mels=256,
            f_min=50,
            f_max=8000
        )
        mel_spectrogram = mel_spectrogram_transform(audio_clip)

        # Check if annotations are already cached
        if audio_fn in self.annotation_cache:
            onset_roll, offset_roll, octave_roll, pitch_roll = self.annotation_cache[audio_fn]
        else:
            duration = self.duration[audio_fn]
            onset_roll, offset_roll, octave_roll, pitch_roll = self.get_labels(self.all_annotations[audio_fn], duration)
            self.annotation_cache[audio_fn] = (onset_roll, offset_roll, octave_roll, pitch_roll)  # Cache the annotations

        # Extract the desired clip of annotations
        start_frame_index = int(start_sec * self.frame_per_sec)
        end_frame_index = int(end_sec * self.frame_per_sec)
        onset_clip = onset_roll[start_frame_index:end_frame_index]
        offset_clip = offset_roll[start_frame_index:end_frame_index]
        octave_clip = octave_roll[start_frame_index:end_frame_index]
        pitch_class_clip = pitch_roll[start_frame_index:end_frame_index]

        return mel_spectrogram, onset_clip, offset_clip, octave_clip, pitch_class_clip

    def get_labels(self, annotation_data, duration):
        '''
        This function reads annotation from file, and then converts annotation from note-level to frame-level
        Because we will be using frame-level labels in training.
        '''
        frame_num = math.ceil(duration * self.frame_per_sec)

        octave_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)
        pitch_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)
        onset_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)
        offset_roll = torch.zeros(size=(frame_num + 1,), dtype=torch.long)

        # Code to convert annotations to frame-level labels goes here...
        # ...

        return onset_roll, offset_roll, octave_roll, pitch_roll

    def get_octave_and_pitch_class_from_pitch(self, pitch, note_start=36):
        '''
        Convert MIDI pitch number to octave and pitch_class
        pitch: int, range [36 (octave 0, pitch_class 0), 83 (octave 3, pitch 11)]
                pitch = 0 means silence
        return: octave, pitch_class.
                if no pitch or pitch out of range, output: 0, 0
        '''
        if pitch == 0:
            return 4, 12

        t = pitch - note_start
        octave = t // 12
        pitch_class = t % 12

        if pitch < note_start or pitch > 83:
            return 0, 0
        else:
            return octave + 1, pitch_class + 1
