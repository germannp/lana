"""Handle tracks in excel files from imaris"""
import pandas as pd


def read_tracks(path, condition=None, sample=None, min_track_length=5):
    """Read tracks from excel file"""
    tracks = pd.read_excel(path, sheetname='Position', skiprows=1)
    old_columns = tracks.columns 

    tracks['Track_ID'] = tracks['TrackID']
    tracks['X'] = tracks['Position X']
    tracks['Y'] = tracks['Position Y']
    tracks['Z'] = tracks['Position Z']

    tracks = tracks.drop(['ID', 'Category', 'Collection', 'TrackID', 
        'Unit', 'Position X', 'Position Y', 'Position Z'], 1)

    if condition != None:
        tracks['Condition'] = condition

    if sample != None:
        tracks['Sample'] = sample

    for track_id, track in tracks.groupby('Track_ID'):
        if track.__len__() < min_track_length:
            tracks = tracks[tracks['Track_ID'] != track_id]

    return tracks


if __name__ == '__main__':
    """Illustrates loading of Imaris tracks"""
    import lana
    tracks = read_tracks('Examples/Imaris_example.xls', sample='Movie 1')
    tracks = lana.analyze_motility(tracks)
    # lana.plot_joint_motility(tracks)
    # lana.plot_motility(tracks)
    lana.lag_plot(tracks)
    print(tracks[tracks['Track_ID'] == 1000000093])
    
    # import matplotlib.pyplot as plt
    # tracks.set_index(['Time', 'Track_ID'])['Turning Angle'].unstack().plot(subplots=True, sharey=True, layout=(-1,6))
    # plt.show()
