from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

from pyannote.audio import Audio
from pyannote.core import Segment

#import librosa

import torch

#from WavLM import WavLM, WavLMConfig

#REMINDER: using the .local faster_whisper

model_size = "large-v3"

# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_waveformsize, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

audio_name = "testaudio.wav"
segments, info = model.transcribe(audio_name, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

#decode audio with librosa:
#waveform1, sr = librosa.load(audio_name)

#decode audio with faster-whisper:
sr1 = 16000
waveform1 = decode_audio(audio_name, sr)

#pyannote Audio class (pyannote/audio/core/io.py)
#mono parameter: in case of multi-channel audio, turn into mono audio by randmoly choosing with 'random' or averaging 
#all channels with 'downmix' 
audio = Audio(mono='downmix')

#EMBEDDING
# load the pre-trained checkpoints
model_checkpoint_loc = 'WavLM-Large.pt'
checkpoint = torch.load(model_checkpoint_loc) #should be .pt file
cfg = WavLMConfig(checkpoint['cfg'])
model = WavLM(cfg)
model.load_state_dict(checkpoint['model'])
model.eval()

#SEGMENTING
def segment_embed(trans_segment, waveform, sr):
    start = trans_segment.start
    end = trans_segment.end
    clip = Segment(start, end)
    audio_segment, sr = audio.crop(waveform, clip)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(audio_segment, audio_segment.shape)
    rep = model.extract_features(wav_input_16khz)[0]
    return rep #as a tensor

for segment in segments:
    print(segment.text)
    embd = segment_embed(segment, audio_name, sr1)
    print(embd)


'''
#SEGMENTING
def segment_audio(trans_segment, waveform, sr):
    start = trans_segment.start
    end = trans_segment.end
    clip = Segment(start, end)
    audio_segment, sr = audio.crop(waveform, clip)
    return audio_segment #as a tensor

for segment in segments:
    print(segment.text)
    audio_segment = segment_audio(segment, audio_name, sr)
    print(audio_segment)

'''










    