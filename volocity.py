"""Functions to handle Volocity cell tracks"""
import pandas as pd


def read_tracks(path, condition=None, sample=None, min_track_length=5):
    """Reads a Pandas DataFrame from Volocity files"""
    with open(path, 'r') as volocity_file:
        lines = volocity_file.readlines()

    for i, line in enumerate(lines):
        if 'Centroid X' in line:
            words = line.split('\t')
            for j, word in enumerate(words):
                if 'Name' in word:
                    condition_id = j
                if 'Track ID' in word:
                    index_track_id = j
                if 'Timepoint' in word:
                    index_time = j
                if 'Centroid X' in word:
                    data_begin = i + 1
                    index_X = j

    tracks = pd.DataFrame()
    for i, line in enumerate(lines[data_begin:]):
        words = line.split('\t')

        try:
            tracks.loc[i, 'Track_ID'] = float(words[index_track_id])
            tracks.loc[i, 'Time'] = float(words[index_time])
            tracks.loc[i, 'X'] = float(words[index_X])
            tracks.loc[i, 'Y'] = float(words[index_X+1])
            tracks.loc[i, 'Z'] = float(words[index_X+2])
        except ValueError:
            pass

    tracks['Source'] = 'Volocity'

    if condition != None:
        tracks['Condition'] = condition

    if sample != None:
        tracks['Sample'] = sample

    for track_id, track in tracks.groupby('Track_ID'):
        if track.__len__() < min_track_length:
            tracks = tracks[tracks['Track_ID'] != track_id]

    return tracks.dropna()


if __name__ == '__main__':
    """Illustrates the analysis of Volocity data"""
    import matplotlib.pyplot as plt
    import motility

    tracks = read_tracks('Examples/Volocity_example.txt')
    tracks = motility.analyze(tracks)
    motility.plot_motility(tracks)
    motility.plot_joint_motility(tracks)
