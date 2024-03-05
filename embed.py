#import librosa
import torch
from WavLM import WavLM, WavLMConfig

#audio_name = "testaudio.wav"

#decode audio with librosa:
#waveform1, sr = librosa.load(audio_name)

#decode audio with faster-whisper:
#sr1 = 16000
#waveform1 = decode_audio(audio_name, sr1)

def load_model():
    # load the pre-trained checkpoints
    model_checkpoint_loc = 'WavLM-Large.pt'
    checkpoint = torch.load(model_checkpoint_loc) #should be .pt file
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, cfg

def embed_audio(model, cfg, audio_segment):
    #audio_segment = segment_audio(trans_segment, waveform, sr)
    if cfg.normalize:
        wav_input_16khz = torch.nn.functional.layer_norm(audio_segment, audio_segment.shape)
    rep = model.extract_features(wav_input_16khz)[0]
    return rep


