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


def by_distance(Tcell_tracks, n_Tcells=10, n_DCs=2, n_iter=10, ln_volume=0.125,
    contact_radius=10):
    """Identify contacts by distance"""
    contacts = pd.DataFrame()
    max_index = 0
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
            contacts.loc[max_index, 'Time'] = time
            contacts.loc[max_index, 'Run'] = n
            contacts.loc[max_index, 'Contacts'] = n_Tcells-free_Tcells.__len__()
            max_index += 1

    contacts['Parameters'] = '{} T cells, {} DCs'.format(n_Tcells, n_DCs)

    return contacts


def plot_over_time(contacts):
    """Plot contacts over time"""
    sns.set(style="white")
    plt.xlabel('Time [h]')
    plt.ylabel('# of Contacts')

    for label, _contacts in contacts.groupby('Parameters'):
        plt.plot(contacts.groupby('Time').median()['Contacts'], label=label)
        plt.fill_between(contacts['Time'].unique(),
            contacts.groupby('Time').min()['Contacts'],
            contacts.groupby('Time').max()['Contacts'], alpha=0.2)

    plt.legend()
    plt.show()


if __name__ == '__main__':
    from remix import silly_tracks

    plot_over_time(by_distance(silly_tracks()))
