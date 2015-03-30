"""Analyze and plot contacts within lymph nodes"""
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial as spatial


def by_distance(tracks, n_Tcells=10, n_DCs=50, n_iter=10,
    ln_volume=0.125e9, contact_radius=10):
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

        if n_Tcells < tracks['Track_ID'].unique().__len__():
            Tcell_tracks = tracks[tracks['Track_ID'].isin(
                np.random.choice(tracks['Track_ID'].unique(), n_Tcells,
                replace=False))]
        else:
            n_Tcells = ['Track_ID'].unique().__len__()
            Tcell_tracks = tracks

        free_Tcells = list(Tcell_tracks['Track_ID'].unique())
        for time, positions in Tcell_tracks.sort('Time').groupby('Time'):
            if free_Tcells != []:
                Tcell_tree = spatial.cKDTree(
                    positions[positions['Track_ID'].isin(free_Tcells)]
                    [['X', 'Y', 'Z']])
                current_contacts = DC_tree.query_ball_tree(
                    Tcell_tree, contact_radius)
                Tcells_in_contact = set([Tcell
                for CD_list in current_contacts
                for Tcell in CD_list])
                for Tcell in sorted(Tcells_in_contact, reverse=True):
                    del free_Tcells[Tcell]
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

    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    import motility
    from remix import silly_tracks

    tracks = silly_tracks(25, 100)
    motility.plot_tracks(tracks, ln_volume=5e6)
    plot_over_time(by_distance(tracks, ln_volume=5e6))
