import os
from tqdm import tqdm

from music21 import meter, note, stream, pitch, duration
from random import randint
import pandas as pd

pitches = [
    'C4', 'D4', 'E4', 'F4', 'G4', 'A4', 
    'B4', 'C5', 'D5', 'E5', 'F5', 'G5'
]
durations = {
    1: 'whole', 0.5: 'half', 0.25: 'quarter', 
    # 0.125: 'eighth', 0.0625: '16th'
}

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

def generate_scores(n):
    data = []
    for i in tqdm(range(n)):
        pp, dd = [], []
        s = stream.Stream()
        s.append(meter.TimeSignature('4/4'))
        for _ in range(1):
            m = stream.Measure()
            s.append(m)
            t = 0
            pn = None
            while t < 4:
                n = note.Note()
                
                # Select a different pitch each time
                p = pitches[randint(0, len(pitches) - 1)]
                while p == pn:
                    p = pitches[randint(0, len(pitches) - 1)]
                n.pitch = pitch.Pitch(p)
                pn = p

                # Select a duration so that the total is <= 4
                d = list(durations.keys())[randint(0, len(durations) - 1)]
                while t + d > 4:
                    d = list(durations.keys())[randint(0, len(durations) - 1)]
                n.duration = duration.Duration(durations[d])
                t += d
                
                m.append(n)
                pp.append(p)
                dd.append(durations[d])
        fname = ("".join(pp)).ljust(30, '*')
        fpath = os.path.join(base_dir, f'data/{fname}.png')
        s.write('musicxml.png', fpath)
        data.append({
            'image': f'data/{"".join(pp)}-1.png', 
            'pitches': " ".join(pp), 
            'durations': " ".join(dd)
        })
    return data


if __name__ == '__main__':
    data = generate_scores(1024)
    pd.DataFrame(data).to_csv(
        os.path.join(base_dir, 'data/dataset.csv'), 
        index=False
    )
    

