from mido import MidiFile, MidiTrack, Message
from midi2audio import FluidSynth
from pydub import AudioSegment
from consonance.params import *

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
