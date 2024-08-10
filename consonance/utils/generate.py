import csv
import os
import random
import warnings
from music21 import stream, note, duration, pitch
from music21.musicxml import m21ToXml

# Suppress annoying MusicXMLWarning
warnings.filterwarnings("ignore", category=m21ToXml.MusicXMLWarning)

def generate_random_note():
    '''
    A function to generate a random note based on a predefined list
    of pitches and durations.
    '''
    pitches = ['C4', 'D4', 'E4', 'F4', 'G4', 'A4', 'B4', 'C5', 'D5', 'E5', 'F5', 'G5']
    durations = ['whole', 'half', 'quarter', 'eighth', '16th']

    selected_pitch = random.choice(pitches)
    selected_duration = random.choice(durations)

    n = note.Note()
    n.pitch = pitch.Pitch(selected_pitch)
    n.duration = duration.Duration(selected_duration)

    return n, selected_pitch, selected_duration

def pitch_to_label(pitch):
    pitch_dict = {'C4': -7, 'D4': -6, 'E4': -5, 'F4': -4, 'G4': -3, 'A4': -2, 'B4': -1,
                  'C5': 0, 'D5': 1, 'E5': 2, 'F5': 3, 'G5': 4}
    return pitch_dict[pitch]

def generate_synthetic_single_musicxml(num_samples=10, output_folder='../raw_data/musicxml_files', label_file='../raw_data/labels.csv'):
    '''
    A function to create musicXML files with a single note of music, along with a label file.
    '''
    # get CWD and create folders if needed
    current_dir = os.getcwd()
    output_folder = os.path.abspath(os.path.join(current_dir, output_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    label_file = os.path.abspath(os.path.join(current_dir, label_file))

    # create music files and labels
    with open(label_file, 'w', newline='') as csvfile:
        label_writer = csv.writer(csvfile)
        label_writer.writerow(['filename', 'label'])
        for i in range(num_samples):
            s = stream.Stream()
            n, selected_pitch, selected_duration = generate_random_note()
            s.append(n)
            filename = f'note_{selected_pitch}_{selected_duration}_{i}.musicxml'
            s.write('musicxml', fp=os.path.join(output_folder, filename))
            label = pitch_to_label(selected_pitch)
            label_writer.writerow([filename.replace('.musicxml', '.png'), label])

    return None



# get the right path for musescore based on system
def get_musescore_path():
    system = platform.system()
    if system == 'Windows':
        return r'C:\Program Files\MuseScore 4\bin\MuseScore4.exe'  # Update this path if necessary
    elif system == 'Darwin':  # macOS
        return '/Applications/MuseScore 4.app/Contents/MacOS/mscore'
    elif system == 'Linux':
        return '/usr/bin/musescore4'  # Update this path if necessary
    else:
        raise ValueError("Unsupported operating system")
    

def convert_musicxml_to_png(input_folder='../raw_data/musicxml_files', output_folder='../raw_data/sheet_images'):
    current_dir = os.getcwd()
    input_folder = os.path.abspath(os.path.join(current_dir, input_folder))
    output_folder = os.path.abspath(os.path.join(current_dir, output_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    musescore_path = get_musescore_path()

    for file_name in os.listdir(input_folder):
        if file_name.endswith('.musicxml'):
            input_path = os.path.join(input_folder, file_name)
            output_filename = file_name.replace('.musicxml', '.png')
            output_path = os.path.join(output_folder, output_filename)
            result = subprocess.run([musescore_path, input_path, '-o', output_path], stderr=subprocess.PIPE)
            if result.returncode != 0:
                print(f"Error processing {file_name}: {result.stderr.decode('utf-8')}")

            # Check if the file has a '-1' suffix and rename it
            generated_filename = output_filename.replace('.png', '-1.png')
            generated_path = os.path.join(output_folder, generated_filename)
            if os.path.exists(generated_path):
                os.rename(generated_path, output_path)

    return None

# if __name__ == '__main__':
#     generate_synthetic_musicxml()
#     convert_musicxml_to_png()