"""Handle tracks in excel files from imaris"""
import pandas as pd


def read_tracks(
        path, condition=None, sample=None, tissue=None, time_step=20,
        min_track_length=5):
    """Read tracks from excel file into pandas DataFrame"""
    tracks = pd.read_excel(path, sheetname='Position', skiprows=1)

    tracks['Track_ID'] = tracks['TrackID']
    tracks['Time'] = (tracks['Time'] - 1) / 60 * time_step
    tracks['X'] = tracks['Position X']
    tracks['Y'] = tracks['Position Y']
    tracks['Z'] = tracks['Position Z']

    tracks = tracks.drop([
        'ID', 'Category', 'Collection', 'TrackID', 'Unit', 'Position X',
        'Position Y', 'Position Z'], 1)

    tracks['Source'] = 'Imaris'

    if condition != None:
        tracks['Condition'] = condition

    if sample != None:
        tracks['Sample'] = sample

    if tissue != None:
        tracks['Tissue'] = tissue

    for track_id, track in tracks.groupby('Track_ID'):
        if track.__len__() < min_track_length:
            tracks = tracks[tracks['Track_ID'] != track_id]

    print(
        'Read {} tracks with {} seconds time step.'.format(
            len(tracks['Track_ID'].unique()), time_step))

    return tracks.sort_values('Time')


if __name__ == '__main__':
    """Illustrates loading of Imaris tracks"""
    from lana import motility

    tracks = read_tracks('Examples/Imaris_example.xls', sample='Movie 1')
    motility.plot_dr(tracks)

    # motility.joint_plot(tracks)
    # motility.plot(tracks)
    # motility.lag_plot(tracks)
    # print(tracks[tracks['Track_ID'] == 1000000093])

    # summary = motility.summarize(tracks)
    # motility.plot_summary(summary)

    # import matplotlib.pyplot as plt
    # tracks.set_index(['Time', 'Track_ID'])['Turning Angle'].unstack().plot(
    #     subplots=True, sharey=True, layout=(-1, 6))
    # plt.show()
