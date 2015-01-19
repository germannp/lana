"""Handle tracks in excel files from imaris"""
import pandas as pd


def read_tracks(path, sample=None):
    tracks = pd.read_excel(path, sheetname='Position', skiprows=1)
    old_columns = tracks.columns 

    tracks['Track_ID'] = tracks['TrackID']
    tracks['X'] = tracks['Position X']
    tracks['Y'] = tracks['Position Y']
    tracks['Z'] = tracks['Position Z']

    tracks = tracks.drop(['ID', 'Category', 'Collection', 'TrackID', 
    	'Unit', 'Position X', 'Position Y', 'Position Z'], 1)

    if sample != None:
    	tracks['Sample'] = sample

    return tracks


if __name__ == '__main__':
    """Illustrates loading of Imaris tracks"""
    import lana
    tracks = read_tracks('Examples/Imaris_example.xls', sample='Movie 1')
    tracks = lana.analyze_tracks(tracks)
    lana.plot_joint_motility(tracks)
