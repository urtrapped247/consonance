# for XML generation
# this needs to be updated!! This is pre-single-note code

from music21 import stream, note, duration, pitch, metadata, clef
import os
import random
import warnings
from music21.musicxml import m21ToXml

# for XML --> PNG
import subprocess
import platform

# Suppress annoying MusicXMLWarning
warnings.filterwarnings("ignore", category=m21ToXml.MusicXMLWarning)

def generate_random_note():
    '''
    A function to generate a random note based on a predefined list
    of pitches and durations.
    '''

    # Define a list of possible pitches and durations
    pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5']
    durations = ['whole', 'half', 'quarter', 'eighth', '16th']

    # Select a random pitch and duration
    selected_pitch = random.choice(pitches)
    selected_duration = random.choice(durations)

    # Create and return a music21 note
    n = note.Note()
    n.pitch = pitch.Pitch(selected_pitch)
    n.duration = duration.Duration(selected_duration)
    return n

def generate_synthetic_musicxml(num_samples=10, output_folder='../../raw_data/musicxml_files'):
    '''
    A function to create a folder of MusicXML files.
    '''
    # check for output - Use this for .py
    output_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'musicxml_files')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    #for our notebook since __file__ seems to not work
    # current_dir = os.getcwd()
    # output_folder = os.path.abspath(os.path.join(current_dir, output_folder))
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    synthetic_data = []

    for i in range(num_samples):
        s = stream.Stream()
        for _ in range(1):  # Generate 21 random note per sheet
            n = generate_random_note()
            s.append(n)
        s.write('musicxml', fp=f'{output_folder}/sheet_{i}.musicxml')
        synthetic_data.append(s)

    return synthetic_data

    # for i in range(num_samples):
    #     m = stream.Measure()
    #     total_duration = 0.0    # we are using a 4/4 measure

    #     # build out the measure
    #     while total_duration < 4.0:
    #         n = generate_random_note()
    #         if total_duration + n.duration.quarterLength <= 4.0:
    #             m.append(n)
    #             total_duration += n.duration.quarterLength
    #         else:
    #             break

    #     # create the score and append the measure
    #     s = stream.Score()
    #     s.append(m)
    #     s.metadata = metadata.Metadata()
    #     s.metadata.title = ''

    #     # write the score to MusicXML
    #     s.write('musicxml', fp=f'{output_folder}/sheet_{i}.musicxml')
    #     synthetic_data.append(s)

    # return synthetic_data


# get the right path for musescore based on system
def get_musescore_path():
    '''
    A simple function to determine the path for musescore depending on the system platform.
    This assumes the program was stored in the default directory.
    '''
    system = platform.system()
    if system == 'Windows':
        return r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'  # Update this path if necessary
    elif system == 'Darwin':  # macOS
        return '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
    elif system == 'Linux':
        return '/usr/bin/musescore4'  # Update this path if necessary
    else:
        raise ValueError("Unsupported operating system")

def convert_musicxml_to_png(input_folder='../../raw_data/musicxml_files', output_folder='../../raw_data/sheet_images'):
    output_folder = os.path.join(os.path.dirname(__file__), os.pardir, 'musicxml_files')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # current_dir = os.getcwd()
    # input_folder = os.path.abspath(os.path.join(current_dir, input_folder))
    # output_folder = os.path.abspath(os.path.join(current_dir, output_folder))
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)

    musescore_path = get_musescore_path() #change this line manually if musescore is installed in a different directory

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.musicxml'):
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name.replace('.musicxml', '.png'))
            result = subprocess.run([musescore_path, input_path, '-o', output_path], stderr=subprocess.PIPE)
            if result.returncode != 0:
                # Handle or log the error if needed
                print(f"Error processing {file_name}: {result.stderr.decode('utf-8')}")

    return None

if __name__ == '__main__':
    generate_synthetic_musicxml()
    convert_musicxml_to_png()
