from faster_whisper import WhisperModel
from segment import segmentation
from embed import load_model, embed_audio
#from embed import embed_audio
import tensorflow as tf
import torchaudio, torch
import os

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
#segment number initialization
seg_num = 0

model, cfg = load_model()

for segment in segments:
    #print(segment.text)
    #calling segmentation creates 
    seg = segmentation(segment, audioin, sr, seg_num)
    seg_num+=1
    seg1= torch.randn(1,30000)
    emb = embed_audio(model, cfg, seg)
    print(emb)
    break



    
    
    