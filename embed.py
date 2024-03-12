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
  '''
  Loads feature extraction model, configuration and weights.
  '''
  # load the pre-trained checkpoints
  model_checkpoint_loc = 'WavLM-Base+.pt' #WavLM-Large consumed way too much RAM
  checkpoint = torch.load(model_checkpoint_loc) #should be .pt file
  cfg = WavLMConfig(checkpoint['cfg'])
  model = WavLM(cfg)
  model.load_state_dict(checkpoint['model'])
  model.eval()
  return model, cfg

def embed_audio(model, cfg, audio_segment, last_layer=False):
  '''
  Generates embeddings for the given audio file or segment. Extracts the
  representation of the last layer (last_layer=True), or the representation from
  each layer and performs a weighted sum (last_layer=False).
  '''

  #if last_layer:
  #if cfg.normalize:
  wav_input_16khz = torch.nn.functional.layer_norm(audio_segment, audio_segment.shape)
  rep = model.extract_features(wav_input_16khz)[0]

  if last_layer == True:
    if cfg.normalize:
      wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz , wav_input_16khz.shape)
    rep, layer_results = model.extract_features(wav_input_16khz,
                                                output_layer=model.cfg.encoder_layers,
                                                ret_layer_results=True)[0]
    layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

  return rep