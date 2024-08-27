from mido import MidiFile, MidiTrack, Message
from midi2audio import FluidSynth
from pydub import AudioSegment
from consonance.params import *

from mido import MidiFile, MidiTrack, Message
import os
from pydub import AudioSegment

def note_mapping(note):
    """
    Map the note to a tuple of (MIDI pitch, duration).
    """
    duration_map = {
        'whole': 1920,
        'half': 960,
        'quarter': 480,
        'eighth': 240,
        '16th': 120
    }

    # Extract the note and duration from the note string
    note_parts = note.split('_')
    pitch = note_parts[0]
    duration = note_parts[1]

    # Mapping note names to MIDI pitch values
    pitch_map = {
        'B3': 59, 'C4': 60, 'D4': 62, 'E4': 64, 'F4': 65, 'G4': 67, 'A4': 69, 'B4': 71,
        'C5': 72, 'D5': 74, 'E5': 76, 'F5': 77, 'G5': 79, 'A5': 81, 'B5': 83,
        'C6': 84, 'D6': 86
    }

    midi_pitch = pitch_map[pitch]
    midi_duration = duration_map[duration]

    return midi_pitch, midi_duration

def create_music_files(note_string, media_type):
    """
    Creates a MIDI file and optionally converts it to a specified media type (wav or mp3).
    If the media type is MIDI or MP3, it also creates a WAV file.

    Args:
    note_string (str): Space-separated string of notes and durations, e.g., 'B3_whole D4_quarter'.
    media_type (str): The desired output format, 'midi', 'wav', or 'mp3'.

    Returns:
    dict: A dictionary containing the file paths of the created files.
    """

    # Split the note string into individual notes
    notes = note_string.split()

    # Map each note to its corresponding (MIDI pitch, duration) tuple
    y_pred = [note_mapping(note) for note in notes]

    # 1. Create MIDI file
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    for note_pitch, duration in y_pred:
        track.append(Message('note_on', note=note_pitch, velocity=64, time=0))
        track.append(Message('note_off', note=note_pitch, velocity=64, time=int(duration)))

    midi_file_path = 'output.mid'
    midi.save(midi_file_path)

    output_files = {'midi': midi_file_path}

    # 2. Convert to WAV or MP3 if needed
    if media_type != 'midi':
        audio_segment = AudioSegment.from_file(midi_file_path, format="mid")

        if media_type == 'mp3':
            mp3_file_path = 'output.mp3'
            audio_segment.export(mp3_file_path, format='mp3')
            output_files['user_format'] = mp3_file_path
        elif media_type == 'wav':
            wav_file_path = 'output.wav'
            audio_segment.export(wav_file_path, format='wav')
            output_files['user_format'] = wav_file_path

    # 3. Create a WAV file if media_type is 'midi' or 'mp3'
    if media_type == 'midi' or media_type == 'mp3':
        wav_file_path = 'output.wav'
        audio_segment = AudioSegment.from_file(midi_file_path, format="mid")
        audio_segment.export(wav_file_path, format='wav')
        output_files['wav'] = wav_file_path

    return output_files


# def create


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
