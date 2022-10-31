from __future__ import unicode_literals
import argparse
import spleeter
import itertools

from pydub import AudioSegment
from pydub.silence import split_on_silence, detect_nonsilent
from pydub.utils import make_chunks
from pathlib import Path
from termcolor import colored
import os
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import numpy as np
import sounddevice as sd
import soundfile as sf
import ffmpeg
import math
from pydub.playback import play
import tensorflow as tf

# process arguments
parser = argparse.ArgumentParser(description='Audio Dataset Processor')

parser.add_argument('--file_path', type=Path,
                    help='Path to the source file')

parser.add_argument(
            "--cleanup_vocals_files", 
            action="store_true", 
            help="Delete the temp vocals files after run")

#file_path = Path("J:\\bbcDownloads\\4_20_21-dira\\4_20_21-dira.mp3")
p = parser.parse_args()
file_path =  p.file_path


def split_on_silence_with_min_clip_length(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,
                     seek_step=1, min_clip_length = 5000):
    """
    Returns list of audio segments from splitting audio_segment on silent sections
    audio_segment - original pydub.AudioSegment() object
    min_silence_len - (in ms) minimum length of a silence to be used for
        a split. default: 1000ms
    silence_thresh - (in dBFS) anything quieter than this will be
        considered silence. default: -16dBFS
    keep_silence - (in ms or True/False) leave some silence at the beginning
        and end of the chunks. Keeps the sound from sounding like it
        is abruptly cut off.
        When the length of the silence is less than the keep_silence duration
        it is split evenly between the preceding and following non-silent
        segments.
        If True is specified, all the silence is kept, if False none is kept.
        default: 100ms
    seek_step - step size for interating over the segment in ms

    min_clip_length - (in ms) minimum length of a generated clip. default: 5000ms
    """

    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    if isinstance(keep_silence, bool):
        keep_silence = len(audio_segment) if keep_silence else 0

    output_ranges = [
        [ start - keep_silence, end + keep_silence ]
        for (start,end)
            in detect_nonsilent(audio_segment, min_silence_len, silence_thresh, seek_step)
    ]

    modified_output_ranges = []
    num_chunks = len(output_ranges)
    i=0
    while i < num_chunks:
        start_chunk = output_ranges[i]
        end_chunk = output_ranges[i]
        while ((end_chunk[1] - start_chunk[0]) < min_clip_length) and i < num_chunks:
            # expand range to get to min clip length
            i+=1
            if(i >= num_chunks):
                break
            end_chunk = output_ranges[i]
        modified_output_ranges.append([start_chunk[0], end_chunk[1]])
        i+=1

    for range_i, range_ii in pairwise(modified_output_ranges):
        last_end = range_i[1]
        next_start = range_ii[0]
        if next_start < last_end:
            range_i[1] = (last_end+next_start)//2
            range_ii[0] = range_i[1]

    return [
        audio_segment[ max(start,0) : min(end,len(audio_segment)) ]
        for start,end in modified_output_ranges
    ]

def main():
    fileName = file_path.stem

    parentFolder = file_path.parent.absolute()

    outputPath = os.path.join(parentFolder , "output-8")
    vocalsPath = os.path.join(outputPath , fileName, "vocals.flac")

    chunksPath = os.path.join(outputPath , fileName, "chunks-withMin")

    if not os.path.exists(outputPath):
        os.mkdir(outputPath)

    if not os.path.exists(os.path.join(outputPath , fileName)):
        os.mkdir(os.path.join(outputPath , fileName))

    # suppress tensorflow warnings and only show errors
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    metadata = ffmpeg.probe(file_path)
    
    
    audio_adapter = AudioAdapter.default()
    if not os.path.exists(vocalsPath):
        # Separate vocals from accompaniment
        block_duration = 30
        num_split_blocks = math.ceil(float(metadata['format']['duration'])/block_duration)
        # num_split_blocks = 10
        # all_audio_segments = AudioSegment.empty()
        separator = Separator('spleeter:2stems')
        allVocals = np.random.uniform(-1,1,44100)
        for i in range(num_split_blocks):
            waveform, _ = audio_adapter.load(
                        file_path,
                        offset=i*block_duration,
                        duration=block_duration,
                        sample_rate=44100,
                    )
            
            prediction = separator.separate(waveform)
            vocals = prediction["vocals"]
            #sd.play(vocals, 44100)

            if(i == 0):
                allVocals = vocals
            else:
                allVocals = np.concatenate((allVocals, vocals), axis=0)
            
        # AudionSegment has an overload that can be initialized with a bytestring, but it
        # results in corrupt audio when writing file output. To get around this, we use
        # soundFile to write out the vocals file and then use the file-based initializtion
        # code in AudioSegment
        sf.write(vocalsPath, allVocals, 44100)
    
    # Split audio into sections by silence
    all_audio_segments = AudioSegment.from_file(vocalsPath)

    # from the itertools documentation
    def pairwise(iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = itertools.tee(iterable)
        next(b, None)
        return zip(a, b)

    # split on silence
    audio_chunks = split_on_silence_with_min_clip_length(all_audio_segments, 
        # must be silent for at least half a second
        min_silence_len=500,

        # consider it silent if quieter than -40 dBFS
        silence_thresh=-40,
        
        # clips should be at least 5 seconds long
        min_clip_length = 5000
    )

    if not os.path.exists(chunksPath):
        os.mkdir(chunksPath)

    for chunk_num, chunk in enumerate(audio_chunks):
        # play(chunk)
        if(chunk.duration_seconds <=30):
            out_file = "{0}/{1}.mp3".format(chunksPath, chunk_num)
            chunk.export(out_file, format="mp3")
        else:
            #try splitting again, with more aggressive pattern
            audio_chunks2 = split_on_silence_with_min_clip_length(chunk, 
                # switch to 200ms silence
                min_silence_len=200,

                # consider it silent if quieter than -40 dBFS
                silence_thresh=-40,

                # clips should be at least 5 seconds long
                min_clip_length = 5000
            )
            for chunk_num2, chunk2 in enumerate(audio_chunks2):
                if(chunk2.duration_seconds <=30):
                    out_file2 = "{0}/{1}_{2}.mp3".format(chunksPath, chunk_num, chunk_num2)
                    chunk2.export(out_file2, format="mp3")
                else:
                    # give up and just split by time
                    audio_chunks3 = make_chunks(chunk2, 30000)
                    for chunk_num3, chunk3 in enumerate(audio_chunks3):
                        out_file3 = "{0}/{1}_{2}_{3}.mp3".format(chunksPath, chunk_num, chunk_num2, chunk_num3)
                        chunk3.export(out_file3, format="mp3")
    
    if(p.cleanup_vocals_files):
        os.remove(vocalsPath)
    
if __name__ == "__main__":
    main()