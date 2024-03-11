from faster_whisper import WhisperModel
from segment import segment_audio, segment_video
from embed import load_model, embed_audio
#from embed import embed_audio
import tensorflow as tf
import torchaudio, torch
import os
from WavLM import WavLM, WavLMConfig

#REMINDER: using the .local faster_whisper

model_size = "large-v3"

# Run on GPU with FP16
#model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
model = WhisperModel(model_size, device="cpu", compute_type="int8")

audioin = "testaudio.wav"

segments, info = model.transcribe(audioin, beam_size=5)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

#--- PARAMS ---#
#sampling rate
sr = 16000

#loading feature extraction model
feat_model, cfg = load_model()

#SEGMENT AND EMBED AUDIO
for segment in segments:
    print(segment.text)
    seg = segment_audio(segment, audioin, sr, save_into_file=True)
    print(seg)
    emb = embed_audio(feat_model, cfg, seg)
    print(emb)
    break

#SEGMENT VIDEO
videoin = 'marta.mp4'

for segment in segments:
  segment_video(segment, videoin)
  break



    
    
    