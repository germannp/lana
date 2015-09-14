"""Functions to handle Volocity cell tracks"""
import pandas as pd


def read_tracks_excel(path, condition=None, sample=None, time_step=20,
    min_track_length=5):
    """Read tracks from excel file into pandas DataFrame"""
    tracks = pd.read_excel(path).reset_index()

    tracks['Track_ID'] = tracks['Track ID']
    tracks['Time'] = (tracks['Timepoint'] - 1)/60*time_step
    tracks['X'] = tracks['Centroid X (µm)']
    tracks['Y'] = tracks['Centroid Y (µm)']
    tracks['Z'] = tracks['Centroid Z (µm)']

    to_drop = [col
        for col in ['Name', 'Track ID', 'Timepoint', 'index',
            'Centroid X (µm)', 'Centroid Y (µm)', 'Centroid Z (µm)']
        if col in tracks.columns]
    tracks = tracks.drop(to_drop, 1)

    tracks['Source'] = 'Volocity'

    if condition != None:
        tracks['Condition'] = condition

    if sample != None:
        tracks['Sample'] = sample

    for track_id, track in tracks.groupby('Track_ID'):
        if track.__len__() < min_track_length:
            tracks = tracks[tracks['Track_ID'] != track_id]

    print('Read {} tracks with {} seconds time step.'.format(
        len(tracks['Track_ID'].unique()), time_step))

    return tracks.sort('Time')


def read_tracks_txt(path, condition=None, sample=None, time_step=20,
    min_track_length=5):
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
            tracks.loc[i, 'Time'] = (float(words[index_time]) - 1)/60*time_step
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

    print('Read {} tracks with {} seconds time step.'.format(
        len(tracks['Track_ID'].unique()), time_step))

    return tracks.dropna().sort('Time')


if __name__ == '__main__':
    """Illustrates the analysis of Volocity data"""
    import matplotlib.pyplot as plt
    import motility

    # tracks = read_tracks_txt('Examples/Volocity_example.txt', sample='Movie 1')
    # motility.plot(tracks)
    # motility.joint_plot(tracks)
    # motility.lag_plot(tracks)

    tracks = read_tracks_excel('Examples/Volocity_example.xlsx')
    print(tracks[tracks.Track_ID == 4474])
    # print(tracks)
    # motility.plot(tracks)
    # motility.joint_plot(tracks)
    # motility.lag_plot(tracks)
