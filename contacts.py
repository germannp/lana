"""Analyze and plot contacts within lymph nodes"""
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial as spatial
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D

from utils import equalize_axis3d


def by_distance(Tcell_tracks, n_Tcells=10, n_DCs=10, n_iter=10,
    ln_volume=1000, contact_radius=10):
    """Identify contacts by distance"""
    contacts = pd.DataFrame()
    max_index = 0
    for n in range(n_iter):
        r = (3*ln_volume/(4*np.pi))**(1/3)*np.random.rand(n_DCs)**(1/3)
        theta = np.random.rand(n_DCs)*2*np.pi
        phi = np.arccos(2*np.random.rand(n_DCs) - 1)
        DCs = pd.DataFrame({
            'X': r*np.sin(theta)*np.sin(phi),
            'Y': r*np.cos(theta)*np.sin(phi),
            'Z': r*np.cos(phi)})
        DC_tree = spatial.cKDTree(DCs)

        if n_Tcells < Tcell_tracks['Track_ID'].unique().__len__():
            T_cells = Tcell_tracks[Tcell_tracks['Track_ID'].isin(
                np.random.choice(Tcell_tracks['Track_ID'].unique(), n_Tcells,
                replace=False))]

        free_Tcells = list(T_cells['Track_ID'].unique())
        for time, T_cell_positions in T_cells.sort('Time').groupby('Time'):
            if free_Tcells != []:
                T_cell_tree = spatial.cKDTree(
                    T_cell_positions[T_cell_positions['Track_ID'].isin(free_Tcells)]
                    [['X', 'Y', 'Z']])
                current_contacts = DC_tree.query_ball_tree(
                    T_cell_tree, contact_radius)
                T_cells_in_contact = set([T_cell
                for CD_list in current_contacts
                for T_cell in CD_list])
                for T_cell in sorted(T_cells_in_contact, reverse=True):
                    del free_Tcells[T_cell]
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

    # by_distance(silly_tracks())
    plot_over_time(by_distance(silly_tracks()))
