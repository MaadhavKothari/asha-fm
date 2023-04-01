import math
import numpy as np
import librosa
import librosa.display

DRUM_AUDIO_PATH = '../raw_data/audio/13.wav'
audio_path = '../raw_data/audio/13.wav'
AUDIO_FILE_NAME = '13'
RAW_AUDIO_PATH = '../raw_data/audio/13.wav'

class AudioProcessor:
    def __init__(self, audio_path):
        self.audio_path = audio_path
        self.FRAME_RATE = 60
        self.HOP_WINDOW = 512
        self.image_time_gap = 1 # seconds, play with this
        self.image_frame_gap = self.image_time_gap * self.frame_rate
        
    def get_audio_params(audio_path):
        # Load song with 22050 sample rate
        y, sr = librosa.load(audio_path, sr=22050)
        audio_duration = librosa.get_duration(y, sr=sr)
        
        total_frames_float = audio_duration * FRAME_RATE
        total_frames = math.ceil(total_frames_float)
        
        sample_rate = round(total_frames_float / audio_duration * HOP_WINDOW)
        
        return total_frames, sample_rate, audio_duration

    def get_onset_info(audio_path, sample_rate):
        # Load
        y, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Onset strengths and normalize
        onset_strengths = librosa.onset.onset_strength(y=y, sr=sample_rate, aggregate=np.median)
        onset_strengths = librosa.util.normalize(onset_strengths)
        
        # Onset timestamps and frames
        onset_times = librosa.times_like(onset_strengths, sr=sample_rate)
        onset_frames = onset_times * FRAME_RATE
        
        onset_info = np.concatenate([
            onset_frames.reshape((-1, 1)),
            onset_times.reshape((-1, 1)),
            onset_strengths.reshape((-1, 1))
        ], axis=1)
        
        # Beat times
        beat_times = librosa.beat.beat_track(y=y, sr=sample_rate, units='time')[1]
        
        return y, onset_info, beat_times

    def create_onset_features(onset_info, beat_times, lin_decay_time=0.25, exp_decay_rate=0.25, decay_magnification=True):
        
        fixed_decay_frames = int(lin_decay_time * FRAME_RATE)
        
        # Create column of zeroes as default value for both Linear and Exp Decay
        onset_info = np.concatenate([onset_info, np.zeros((onset_info.shape[0], 1)), np.zeros((onset_info.shape[0], 1))], axis=1)
        
        # for each row
        for i in range(onset_info.shape[0]):
            # Skip the first value, makes life easy
            if i == 0:
                onset_info[i, 3] = 0
        
            
            ## LINEAR
            # If the timestamp is in beat_times, it's a peak
            if onset_info[i, 1] in beat_times:
                onset_info[i, 3] = onset_info[i, 2] # Linear Column
                
                # Decay Params
                decay_factor = 1
                if decay_magnification:
                    decay_factor *= (onset_info[i, 2] + 1)
                
                decay_frames = fixed_decay_frames * decay_factor
                lin_decay_val = onset_info[i, 2] / decay_frames
            
            # Check if the previous value is zero or less than the decay_val -> 0
            if onset_info[i - 1, 3] == 0. or abs(onset_info[i -1, 3]) < lin_decay_val:
                pass
            
            # If previous value > 0, needs decay
            elif onset_info[i - 1, 3] > 0: 
                onset_info[i, 3] = onset_info[i - 1, 3] - lin_decay_val
            
            
            # EXPONENTIAL
            # If the timestamp is in beat_times, it's a peak
            if onset_info[i, 1] in beat_times:
                onset_info[i, 4] = onset_info[i, 2] # Exp Column
            
            # Set current to zero if previous is zero or small number
            elif onset_info[i - 1, 4] == 0 or onset_info[i - 1, 4] < 0.005: 
                pass
            
            # If previous value > 0, needs decay
            elif onset_info[i - 1, 4] != 0: 
                    onset_info[i, 4] = onset_info[i - 1, 4] * (1 - exp_decay_rate)
        
        return onset_info