import math
import torch
import torchaudio
import soundfile as sf  # Using soundfile for memory-mapped file access
from torch.utils.data import Dataset, DataLoader
from utils import read_json, ls, jpath
from concurrent.futures import ThreadPoolExecutor
import os
def move_data_to_device(data, device='cuda'):
    return [i.to(device) if isinstance(i, torch.Tensor) else i for i in data]

def load_audio_segment(file_path, start_sec, duration_sec, sample_rate):
    frames_to_read = int(duration_sec * sample_rate)
    start_frame = int(start_sec * sample_rate)

    with sf.SoundFile(file_path) as f:
        f.seek(start_frame)
        audio = f.read(frames_to_read)
    return audio


def load_audio_segment_torchaudio(file_path, start_sec, duration_sec, sample_rate):
    # Load audio with torchaudio
    waveform, sr = torchaudio.load(file_path, normalize=True)

    # Calculate start and end frames
    start_frame = int(start_sec * sample_rate)
    end_frame = start_frame + int(duration_sec * sample_rate)

    # Check if the desired frames exceed the waveform length
    if end_frame > waveform.shape[1]:
        end_frame = waveform.shape[1]

    # Slice the waveform to get the segment
    segment = waveform[:, start_frame:end_frame]
    return segment

def get_data_loader(split, args, fns=None):
    """
    Create a DataLoader for the dataset with optimized settings.
    """
    dataset = MyDataset(
        dataset_root=args['dataset_root'],
        split=split,
        sampling_rate=args['sampling_rate'],
        annotation_path=args['annotation_path'],
        sample_length=args['sample_length'],
        frame_size=args['frame_size'],
        song_fns=fns,
        device='cuda'
    )

    # Optimized DataLoader configuration
    data_loader = DataLoader(
        dataset,
        batch_size=args['batch_size'],
        num_workers=args['num_workers'],  # Set to the number of CPU cores for parallel data loading
        pin_memory=True,
        shuffle=False,
        collate_fn=collate_fn,
        persistent_workers=True,  # Keep workers alive between epochs
        pin_memory_device='cuda'
    )
    return data_loader

def calculate_no_frames(sample_rate, segment_duration, hop_length):
    return int(sample_rate * segment_duration / hop_length)

def collate_fn(batch):
    '''
    Group different components into separate tensors and pad samples to the maximum length in the batch.
    '''
    inp, onset, offset, octave, pitch = [], [], [], [], []
    max_frame_num = max(sample[0].shape[0] for sample in batch)

    for sample in batch:
        inp.append(torch.nn.functional.pad(sample[0], (0, 0, 0, max_frame_num - sample[0].shape[0]), mode='constant', value=0))
        onset.append(torch.nn.functional.pad(sample[1], (0, max_frame_num - sample[1].shape[0]), mode='constant', value=0))
        offset.append(torch.nn.functional.pad(sample[2], (0, max_frame_num - sample[2].shape[0]), mode='constant', value=0))
        octave.append(torch.nn.functional.pad(sample[3], (0, max_frame_num - sample[3].shape[0]), mode='constant', value=0))
        pitch.append(torch.nn.functional.pad(sample[4], (0, max_frame_num - sample[4].shape[0]), mode='constant', value=0))

    return torch.stack(inp), torch.stack(onset), torch.stack(offset), torch.stack(octave), torch.stack(pitch)


class MyDataset(Dataset):
    def __init__(self, dataset_root, split, sampling_rate, annotation_path, sample_length, frame_size, device='cuda', song_fns=None):
        self.dataset_root = dataset_root
        self.split = split
        self.dataset_path = jpath(self.dataset_root, self.split)
        self.sampling_rate = sampling_rate
        self.annotation_path = annotation_path
        self.all_annotations = read_json(self.annotation_path)
        self.sample_length = sample_length
        self.frame_size = frame_size
        self.frame_per_sec = int(1 / self.frame_size)
        self.device =device
        self.audio_cache = {}  # Cache for audio data
        self.annotation_cache = {}  # Cache for annotations
        self.mel_cache = {}


        if song_fns is None:
            self.song_fns = ls(self.dataset_path)
            self.song_fns.sort()
        else:
            self.song_fns = song_fns

        self.audio_paths = {fn: self.find_wav_file(fn) for fn in self.song_fns if fn.isdigit()}
        self.index = self.index_data(sample_length)

        # Preload and preprocess all data
        self.preload_data()

    def preload_data(self):
        for fn in self.song_fns:
            audio_path = self.audio_paths.get(fn)
            if audio_path:
                full_audio = self.load_full_audio(audio_path)
                self.audio_cache[fn] = full_audio
                annotations = self.load_and_cache_annotations(fn)
                mel_spectrogram = self.get_mel_spectrogram(full_audio, fn, self.sampling_rate)

    def initialize_audio_paths(self, dataset_root, song_fns):
        audio_paths = {}
        with ThreadPoolExecutor(max_workers=10) as executor:  # Adjust the number of workers to your environment
            future_to_fn = {executor.submit(find_wav_file, os.path.join(dataset_root, fn)): fn for fn in song_fns}
            for future in concurrent.futures.as_completed(future_to_fn):
                fn = future_to_fn[future]
                result = future.result()
                if result:
                    audio_paths[fn] = result
        return audio_paths

    def find_wav_file(self, audio_fn):
        audio_dir = jpath(self.dataset_path, audio_fn)

        # Check if the directory name itself is numeric
        if not os.path.isdir(audio_dir) or not audio_fn.isdigit():
            return None  # Skip non-numeric directories or non-existent paths

        for file in os.listdir(audio_dir):
            if file.lower().endswith('.wav'):
                return os.path.join(audio_dir, file)
        return None

    def load_and_cache_annotations(self, audio_fn):
        if audio_fn in self.annotation_cache:
            return self.annotation_cache[audio_fn]

        annotation_data = self.all_annotations.get(audio_fn, [])
        duration = self.all_annotations[audio_fn][-1][1]
        full_annotations = self.get_labels(annotation_data, duration)
        self.annotation_cache[audio_fn] = full_annotations
        return full_annotations


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
        return index

    def __len__(self):
        return len(self.index)

    def load_full_audio(self, file_path):
        with sf.SoundFile(file_path) as f:
            audio = f.read(dtype='float32')
        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        if audio_tensor.ndim > 1:
            audio_tensor = torch.mean(audio_tensor, dim=1)
        return audio_tensor.to(self.device)

    def get_mel_spectrogram(self, audio,fn, sample_rate):
        # Create the MelSpectrogram transform configured to operate on the device
        mel_spectrogram_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=2048,
            hop_length=int(sample_rate * self.frame_size),
            n_mels=256,
            f_min=50,
            f_max=8000
        ).to(self.device)  # Ensure the transform is on the correct device

        # Apply the transform
        mel_spectrogram = mel_spectrogram_transform(audio)
        mel_spectrogram = mel_spectrogram.transpose(1, 0)  # Adjust dimensions as needed
        self.mel_cache[fn] = mel_spectrogram
        return mel_spectrogram

    def __getitem__(self, idx):
        audio_fn, start_sec = self.index[idx]
        mel_spectrogram = self.mel_cache[audio_fn]
        annotations = self.annotation_cache[audio_fn]

        start_frame = int(start_sec * self.sampling_rate)
        end_frame = start_frame + int(self.sample_length * self.sampling_rate)


        mel_segment = mel_spectrogram[:, start_frame:end_frame] #[:250, :]
        onset_clip = annotations['onset_roll'][start_frame:end_frame]
        offset_clip = annotations['offset_roll'][start_frame:end_frame]
        octave_clip = annotations['octave_roll'][start_frame:end_frame]
        pitch_class_clip = annotations['pitch_roll'][start_frame:end_frame]

        return mel_spectrogram, onset_clip, offset_clip, octave_clip, pitch_class_clip

    def get_labels(self, annotation_data, duration):
        '''
        This function reads annotation from file, and then converts annotation from note-level to frame-level
        Because we will be using frame-level labels in training.
        '''
        frame_num = math.ceil(duration * self.frame_per_sec)

        # Initialize frame-level label vectors with the correct size
        onset_roll = torch.zeros(size=(frame_num,), dtype=torch.long)
        offset_roll = torch.zeros(size=(frame_num,), dtype=torch.long)
        octave_roll = torch.zeros(size=(frame_num,), dtype=torch.long)
        pitch_roll = torch.zeros(size=(frame_num,), dtype=torch.long)

        # Iterate over each note in the annotation data
        for note in annotation_data:
            start_time, end_time, pitch = note

            # Convert times to frame indices
            start_frame = int(max(start_time * self.frame_per_sec, 0))
            end_frame = int(min(end_time * self.frame_per_sec, frame_num - 1))

            # Set onset and offset markers
            if start_frame < frame_num:
                onset_roll[start_frame] = 1
            if end_frame >=0 and end_frame < frame_num:
                offset_roll[end_frame] = 1

            # Convert pitch to octave and pitch class
            octave, pitch_class = self.get_octave_and_pitch_class_from_pitch(pitch)
            frame_range = range(max(start_frame, 0), min(end_frame + 1, frame_num))

            octave_roll[frame_range] = octave
            pitch_roll[frame_range] = pitch_class

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
