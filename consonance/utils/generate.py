import csv
import os
import random
import warnings
from music21 import stream, note, duration, pitch
from music21.musicxml import m21ToXml

# Suppress annoying MusicXMLWarning
warnings.filterwarnings("ignore", category=m21ToXml.MusicXMLWarning)

def pitch_to_label(pitch):
    pitch_dict = {
    'B3': 0, 'C4': 1, 'D4': 2, 'E4': 3, 'F4': 4, 'G4': 5, 'A4': 6, 'B4': 7, 'C5': 8,
    'D5': 9, 'E5': 10, 'F5': 11, 'G5': 12, 'A5': 13, 'B5': 14, 'C6': 15, 'D6': 16}
    return pitch_dict[pitch]

def duration_to_label(duration):
    duration_dict = {
        "whole": 17, "half": 18, "quarter": 19, "eighth": 20, "16th": 21}
    return duration_dict[duration]
    
def generate_random_note(last_note, remaining_beats):
    '''
    A function to generate a random note ensuring that it is not the same as the last note
    and that the note fits within the remaining beats of the measure.
    '''
    pitches = list(pitch_dict.keys())
    durations = list(duration_dict.keys())

    # Filter out the last selected pitch
    available_pitches = [p for p in pitches if p != last_note]

    # Map duration names to their respective beat values
    duration_to_beats = {
        "whole": 4.0, "half": 2.0, "quarter": 1.0, "eighth": 0.5, "16th": 0.25
    }

    # Filter durations to fit within the remaining beats
    available_durations = [d for d in durations if duration_to_beats[d] <= remaining_beats]

    # Just a safety check if somehow the last_note wasn't excluded
    if not available_pitches:
        available_pitches = pitches  # Revert to all pitches if something goes wrong

    selected_pitch = random.choice(available_pitches)
    selected_duration = random.choice(available_durations)

    n = note.Note()
    n.pitch = pitch.Pitch(selected_pitch)
    n.duration = duration.Duration(selected_duration)

    return n, selected_pitch, selected_duration, duration_to_beats[selected_duration]

# This might be outdated, but will keep for now
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

# This is the new code to generate a whole sheet of music insteaf of a single note
def generate_music(num_sheets=5, num_notes=150, output_folder='../raw_data/musicxml_files', label_file='../raw_data/labels.csv'):
    '''
    A function to create musicXML files with a specified number of random notes,
    and generate a label file with the note values, ensuring no immediate repetition
    and that each measure has the exact number of beats.
    '''
    current_dir = os.getcwd()
    output_folder = os.path.abspath(os.path.join(current_dir, output_folder))
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    label_file = os.path.abspath(os.path.join(current_dir, label_file))

    with open(label_file, 'w', newline='') as csvfile:
        label_writer = csv.writer(csvfile)
        label_writer.writerow(['filename', 'labels'])

        for i in range(num_sheets):
            s = stream.Stream()
            pitch_values = []
            duration_values = []
            last_note = None
            remaining_beats = 4.0  # Assuming a 4/4 time signature

            for _ in range(num_notes):
                n, selected_pitch, selected_duration, note_beats = generate_random_note(last_note, remaining_beats)

                s.append(n)
                pitch_values.append(pitch_dict[selected_pitch])
                duration_values.append(duration_dict[selected_duration])

                last_note = selected_pitch
                remaining_beats -= note_beats

                if remaining_beats == 0:
                    remaining_beats = 4.0  # Reset for the next measure

            filename = f'music_{i}.musicxml'
            s.write('musicxml', fp=os.path.join(output_folder, filename))
            label_writer.writerow([filename.replace('.musicxml', '.png'), pitch_values])
            label_writer.writerow([filename.replace('.musicxml', '.png'), duration_values])

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
