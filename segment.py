#from faster_whisper import WhisperModel
#from faster_whisper.audio import decode_audio
from pyannote.audio import Audio
from pyannote.core import Segment
import torch, torchaudio
import ffmpeg
import os
import time, subprocess
#import librosa

#DEPENDENCIES: faster-whisper, pyannote.audio, pyannote.core 

#decode audio with librosa:
#waveform1, sr = librosa.load(audio_name)

#decode audio with faster-whisper:
#sr1 = 16000
#waveform1 = decode_audio(audio_name, sr1)

#pyannote Audio class (pyannote/audio/core/io.py)
#mono parameter: in case of multi-channel audio, turn into mono audio by randmoly choosing with 'random' or averaging 
#all channels with 'downmix' 
audio = Audio(mono='downmix') #add parsing arg (for nemo we need mono)

def seconds_to_hms(seconds):
  '''
  Given an int/float number of seconds it returns a string with the format HH:MM:SS.
  '''
  return time.strftime('%H:%M:%S', time.gmtime(seconds))


def segment_audio(trans_segment,
            waveform,
            sr,
            save_into_file=False):
  '''
  Given a start and end timestamp from the transcription, it crops an audio file
  or segment returned as a torch tensor. Moreover, if save_into_file=True, it
  saves the segment into a .wav and .pt file. For default, turned off as it
  consumes storage memory.
  '''
  #kathleen needs: .wav segs, .pt segs

  start = trans_segment.start
  end = trans_segment.end
  clip = Segment(start, end)
  audio_segment, sr = audio.crop(waveform, clip) #as a tensor

  if save_into_file:
    #extracting file name and extension
    name, extension = os.path.splitext(waveform)
    if not os.path.exists("%s_audio_segments" % (name)):
      os.makedirs("%s_audio_segments" % (name))
    
    #getting timestamp in hh:mm:ss
    astart = seconds_to_hms(start)
    aend = seconds_to_hms(end)

    #saving segment into .wav
    torchaudio.save("./%s_audio_segments/%s_%s_%s%s" % (name, name, astart, aend, extension),
                        audio_segment, sr, format="wav")

    #saving segment into .pt 
    #TODO: remove pt file so torchaudio is not used on worker as well
    torch.save(audio_segment, "./%s_audio_segments/%s_%s_%s%s" % (name, name, astart, 
                                                            aend, extension),
                 _use_new_zipfile_serialization=False)

  return audio_segment

def segment_video(trans_segment,
            video):
  '''
  Given a start and end timestamp from the transcription, it crops video file
  saves the segment into a .mp4 file.
  '''
  #kathleen needs: .mp4 segs

  #getting timestamps and formatting for ffmpeg
  vstart = seconds_to_hms(trans_segment.start)
  vend = seconds_to_hms(trans_segment.end)

  #extracting filename and extension
  name, extension = os.path.splitext(video)
  if not os.path.exists("%s_video_segments" % (name)):
      os.makedirs("%s_video_segments" % (name))
  
  outfile = "./%s_video_segments/%s_%s_%s%s" % (name, name, vstart, vend, extension)
  
  run_ffmpeg = ['ffmpeg', '-ss', vstart, '-to', vend, '-i', video, '-c', 'copy', outfile]
  subprocess.call(run_ffmpeg)
    