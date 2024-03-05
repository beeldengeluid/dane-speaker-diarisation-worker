#from faster_whisper import WhisperModel
#from faster_whisper.audio import decode_audio
from pyannote.audio import Audio
from pyannote.core import Segment
import torch, torchaudio
#import ffmpeg
import os
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
audio = Audio(mono='downmix')

def segmentation(trans_segment, 
            waveform, 
            #video, 
            sr, 
            seg_num,
            segment_audio=True, 
            segment_video=False):
    
    if segment_audio:
        #segmenting
        start = trans_segment.start
        end = trans_segment.end
        clip = Segment(start, end)
        audio_segment, sr = audio.crop(waveform, clip) #as a tensor
        
        #extracting file name and extension 
        name, extension = os.path.splitext(waveform)
        if not os.path.exists("%s_segments" % (name)):
            os.makedirs("%s_segments" % (name))
        
        #saving segment into file 
        torchaudio.save("./%s_segments/%s_seg_%x%s" % (name, name, seg_num, extension),
                         audio_segment, sr, format="wav")

        return audio_segment
    
    #if segment_video:
        #ffmpeg -i video -ss 00:01:00 -to 00:02:00 -c copy output_segmented_video.mp4




    