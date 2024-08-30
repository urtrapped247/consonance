from mido import MidiFile, MidiTrack, Message, bpm2tempo, MetaMessage
from midi2audio import FluidSynth
from pydub import AudioSegment
import os
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

def adjust_for_key(midi_note, key):
    key_adjustments = {
        "G Major": {77: 78},  # F5 -> F#5
        "D Major": {77: 78, 79: 80},  # F5 -> F#5, C6 -> C#6
        "A Major": {77: 78, 79: 80, 84: 85},  # F5 -> F#5, C6 -> C#6, G6 -> G#6
        "F Major": {71: 70},  # B4 -> Bb4
        "Bb Major": {71: 70, 76: 75},  # B4 -> Bb4, E5 -> Eb5
        "Eb Major": {71: 70, 76: 75, 81: 80},  # B4 -> Bb4, E5 -> Eb5, A5 -> Ab5
        "A Minor": {},  # A Minor has no accidental adjustments
        "E Minor": {77: 78},  # F5 -> F#5
        "B Minor": {77: 78, 79: 80},  # F5 -> F#5, C6 -> C#6
        "D Minor": {71: 70},  # B4 -> Bb4
        "G Minor": {71: 70, 76: 75},  # B4 -> Bb4, E5 -> Eb5
        "C Major": {}  # C Major has no accidental adjustments
    }
    return key_adjustments.get(key, {}).get(midi_note, midi_note)

# def create_music_files(note_string, media_type):
#     """
#     Creates a MIDI file and optionally converts it to a specified media type (wav or mp3).
#     If the media type is MIDI or MP3, it also creates a WAV file.

#     Args:
#     note_string (str): Space-separated string of notes and durations, e.g., 'B3_whole D4_quarter'.
#     media_type (str): The desired output format, 'midi', 'wav', or 'mp3'.

#     Returns:
#     dict: A dictionary containing the file paths of the created files.
#     """

#     # Split the note string into individual notes
#     notes = note_string.split()

#     # Map each note to its corresponding (MIDI pitch, duration) tuple
#     y_pred = [note_mapping(note) for note in notes]

#     # 1. Create MIDI file
#     midi = MidiFile()
#     track = MidiTrack()
#     midi.tracks.append(track)

#     for note_pitch, duration in y_pred:
#         track.append(Message('note_on', note=note_pitch, velocity=64, time=0))
#         track.append(Message('note_off', note=note_pitch, velocity=64, time=int(duration)))

#     midi_file_path = 'output.mid'
#     midi.save(midi_file_path)

#     output_files = {'midi': midi_file_path}

#     # 2. Convert to WAV or MP3 if needed
#     if media_type != 'midi':
#         audio_segment = AudioSegment.from_file(midi_file_path, format="mid")

#         if media_type == 'mp3':
#             mp3_file_path = 'output.mp3'
#             audio_segment.export(mp3_file_path, format='mp3')
#             output_files['user_format'] = mp3_file_path
#         elif media_type == 'wav':
#             wav_file_path = 'output.wav'
#             audio_segment.export(wav_file_path, format='wav')
#             output_files['user_format'] = wav_file_path

#     # 3. Create a WAV file if media_type is 'midi' or 'mp3'
#     if media_type == 'midi' or media_type == 'mp3':
#         wav_file_path = 'output.wav'
#         audio_segment = AudioSegment.from_file(midi_file_path, format="mid")
#         audio_segment.export(wav_file_path, format='wav')
#         output_files['wav'] = wav_file_path

#     print("\n🥝 output_files: 🥝 ", output_files, "\n")
#     return output_files

def create_music_files(note_string, media_type, tempo_bpm=120, key_signature="C Major"):
    """
    Creates a MIDI file and optionally converts it to a specified media type (wav or mp3).
    If the media type is MIDI or MP3, it also creates a WAV file.
    Supports tempo and key signature adjustments.

    Args:
    note_string (str): Space-separated string of notes and durations, e.g., 'B3_whole D4_quarter'.
    media_type (str): The desired output format, 'midi', 'wav', or 'mp3'.
    tempo_bpm (int): Tempo in beats per minute. Default is 120 BPM.
    key_signature (str): The key signature for the music, e.g., 'C Major', 'G Major'.

    Returns:
    dict: A dictionary containing the file paths of the created files.
    """
    soundfont_path = os.path.join(os.getcwd(), "consonance/utils/FluidR3_GM.sf2")

    # Split the note string into individual notes
    notes = note_string.split()

    # Map each note to its corresponding (MIDI pitch, duration) tuple
    y_pred = [note_mapping(note) for note in notes]
    print("\n🥝 note_mapping y_pred: 🥝 ", y_pred, "\n")

    # 1. Create MIDI file
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    # Set the tempo
    tempo = bpm2tempo(tempo_bpm)  # Converts BPM to microseconds per beat
    # track.append(Message('set_tempo', tempo=tempo))
    track.append(MetaMessage('set_tempo', tempo=tempo))


    for note_pitch, duration in y_pred:
        # Adjust for key signature
        adjusted_pitch = adjust_for_key(note_pitch, key_signature)
        # Calculate the adjusted time based on the tempo
        adjusted_duration = int(duration * (120 / tempo_bpm))
        track.append(Message('note_on', note=adjusted_pitch, velocity=64, time=0))
        track.append(Message('note_off', note=adjusted_pitch, velocity=64, time=adjusted_duration))

    midi_file_path = 'output.mid'
    midi.save(midi_file_path)

    output_files = {'midi': midi_file_path}

    # 2. Convert MIDI to WAV or MP3 using FluidSynth
    if media_type in ['wav', 'mp3']:
        wav_file_path = 'output.wav'
        fs = FluidSynth(soundfont_path)
        fs.midi_to_audio(midi_file_path, wav_file_path)
        output_files['wav'] = wav_file_path

        if media_type == 'mp3':
            mp3_file_path = 'output.mp3'
            os.system(f"ffmpeg -i {wav_file_path} -codec:a libmp3lame -qscale:a 2 {mp3_file_path}")
            output_files['user_format'] = mp3_file_path

    # 3. If media_type is 'midi', create WAV as well
    elif media_type == 'midi':
        wav_file_path = 'output.wav'
        fs = FluidSynth(soundfont_path)
        fs.midi_to_audio(midi_file_path, wav_file_path)
        output_files['wav'] = wav_file_path

    return output_files
