'''
General utility functions.
'''

from mido import MidiFile, MidiTrack, Message
from midi2audio import FluidSynth
from pydub import AudioSegment
import time
import tracemalloc
import os
from consonance.params import *

def simple_time_and_memory_tracker(method):

    # ### Log Level
    # 0: Nothing
    # 1: Print Time and Memory usage of functions
    LOG_LEVEL = 1

    def method_with_trackers(*args, **kw):
        ts = time.time()
        tracemalloc.start()
        result = method(*args, **kw)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        te = time.time()
        duration = te - ts
        if LOG_LEVEL > 0:
            output = f"{method.__qualname__} executed in {round(duration, 2)} seconds, using up to {round(peak / 1024**2,2)}MB of RAM"
            print(output)
        return result

    return method_with_trackers

def is_directory_empty(directory: str) -> bool:
    """
    Check if the specified directory is empty.
    """
    return not any(os.scandir(directory))

def convert_notes_to_midi(note_string, position_to_midi):   # position_to_midi is the dictionary
    # Initialize an empty list to store the MIDI values
    midi_values = []

    # Split the string into pairs of characters representing notes
    notes = [note_string[i:i+2] for i in range(0, len(note_string), 2)]

    # Convert each note to its corresponding MIDI value using the dictionary
    for note in notes:
        if note in position_to_midi:
            midi_values.append(position_to_midi[note])
        else:
            print(f"Warning: {note} not found in the mapping dictionary.")

    return midi_values

# Example usage
position_to_midi = {
    'A4': 69, 'B4': 71, 'C5': 72, 'D5': 74, 'E5': 76, 'F5': 77, 'G5': 79,
    'A5': 81, 'B5': 83, 'C6': 84, 'D6': 86, 'E6': 88, 'F6': 89, 'G6': 91,
    'A6': 93, 'B6': 95, 'C7': 96
}

# note_string = 'A4G5B6C7G6' <--- a beautiful tune
midi_values = convert_notes_to_midi(note_string, position_to_midi)
print(midi_values)

def quarter_note_conversion(note_list):
    midi_tuple = []
    for note in note_list:
        midi_tuple.append((note, 480))

    return midi_tuple

midi_tuple = quarter_note_conversion(midi_values)


def create_midi(notes, output_file='output.mid', ticks_per_beat=480):
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # Set a default instrument (optional)
    track.append(Message('program_change', program=0, time=0))

    for pitch, duration in notes:
        track.append(Message('note_on', note=pitch, velocity=64, time=0))
        track.append(Message('note_off', note=pitch, velocity=64, time=duration))

    mid.save(output_file)

# Example usage
# notes = [(69, 480), (79, 480), (95, 480), (96, 480), (91, 480)]
create_midi(midi_tuple, output_file='my_music.mid')


def convert_midi_to_audio(input_midi: str, output_format: str):
    """
    Convert a MIDI file to the specified audio format.

    Parameters:
    - input_midi: Path to the input MIDI file.
    - output_format: Desired output audio format (e.g., "mp3", "wav").

    The output file will be saved with the same name as the input MIDI file but with the specified audio format.
    """
    # Ensure the output format is lowercase
    output_format = output_format.lower()

    # Check if the output format is supported
    supported_formats = ["mp3", "wav", "ogg", "flac", "aac", "wma"]
    if output_format not in supported_formats:
        raise ValueError(f"Unsupported output format: {output_format}. Supported formats are: {supported_formats}")

    # Create the output file name by replacing the .mid extension with the desired format
    output_file = input_midi.replace('.mid', f'.{output_format}')

    # Temporary WAV file to store the intermediate audio
    temp_wav_file = "temp_output.wav"

    # Convert MIDI to WAV using FluidSynth
    fs = FluidSynth()
    fs.midi_to_audio(input_midi, temp_wav_file)

    # Load the temporary WAV file using pydub
    audio = AudioSegment.from_wav(temp_wav_file)

    # Export the audio in the desired format
    audio.export(output_file, format=output_format)

    # Optionally, delete the temporary WAV file if you want to clean up
    # import os
    # os.remove(temp_wav_file)

    return output_file

convert_midi_to_audio('my_music.mid', 'mp3')
