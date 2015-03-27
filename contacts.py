"""Analyze and plot contacts within lymph nodes"""
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D


def by_distance(Tcell_tracks, n_Tcells=10, n_DCs=1, n_iter=10, ln_volume=0.125,
    contact_radius=10):
    """Identify contacts by distance"""
    contacts = pd.DataFrame()
    for n in range(n_iter):
        DCs = pd.DataFrame({'X': [10, -10], 'Y': [10, -10], 'Z': [10, -10]})
        if n_Tcells < Tcell_tracks['Track_ID'].unique().__len__():
            tracks = Tcell_tracks[Tcell_tracks['Track_ID'].isin(
                np.random.choice(Tcell_tracks['Track_ID'].unique(), n_Tcells))]

        free_Tcells = list(tracks['Track_ID'].unique())
        for time, positions in tracks.groupby('Time'):
            for dc, tcell in itertools.product(DCs.index, free_Tcells):
                position = positions[positions['Track_ID'] == tcell]
                distance = np.linalg.norm(
                    DCs[['X','Y','Z']].loc[dc] - position[['X','Y','Z']])
                if distance < contact_radius:
                    free_Tcells.remove(tcell)
            contacts.loc[time, 'Run '+str(n)] = n_Tcells - free_Tcells.__len__()

    sns.set(style="white")
    plt.xlabel('Time [h]')
    plt.ylabel('# of Contacts')
    plt.plot(contacts.index/60, contacts.median(axis=1))
    plt.fill_between(contacts.index/60, contacts.min(axis=1), contacts.max(axis=1), alpha=0.2)
    plt.show()



if __name__ == '__main__':
    from remix import silly_tracks

    by_distance(silly_tracks())
