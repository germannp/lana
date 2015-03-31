"""Analyze and plot contacts within lymph nodes"""
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial as spatial


def by_distance(tracks, n_Tcells=[10,20], n_DCs=[50,100], n_iter=10,
    ln_volume=0.125e9, contact_radius=10):
    """Simulate contacts by distance"""
    if max(n_Tcells) > tracks['Track_ID'].unique().__len__():
        print('Error: max. n_Tcells is larger than # of given tracks.')
        return

    print('Simulating contacts {} times'.format(n_iter))
    contacts = pd.DataFrame()
    max_index = 0
    for n in range(n_iter):
        r = (3*ln_volume/(4*np.pi))**(1/3)*np.random.rand(max(n_DCs))**(1/3)
        theta = np.random.rand(max(n_DCs))*2*np.pi
        phi = np.arccos(2*np.random.rand(max(n_DCs)) - 1)
        DCs = pd.DataFrame({
            'X': r*np.sin(theta)*np.sin(phi),
            'Y': r*np.cos(theta)*np.sin(phi),
            'Z': r*np.cos(phi)})
        DC_subsets = dict(zip(n_DCs, [np.random.choice(DCs.index, n,
            replace=False) for n in n_DCs]))
        DC_tree = spatial.cKDTree(DCs)

        T_tracks = tracks[tracks['Track_ID'].isin(
            np.random.choice(tracks['Track_ID'].unique(), max(n_Tcells),
            replace=False))]

        free_Tcells = dict()
        for combo in itertools.product(n_Tcells, n_DCs):
            free_Tcells[combo] = set(np.random.choice(T_tracks['Track_ID'].unique(),
                combo[0], replace=False).flat)

        for time, positions in T_tracks.sort('Time').groupby('Time'):
            left_Tcells = set()
            for Tcells in free_Tcells.values():
                left_Tcells = left_Tcells | Tcells
            if left_Tcells != set():
                Tcell_tree = spatial.cKDTree(
                    positions[positions['Track_ID'].isin(left_Tcells)]
                    [['X', 'Y', 'Z']])
                current_contacts = DC_tree.query_ball_tree(
                    Tcell_tree, contact_radius)
                for cell_numbers in free_Tcells:
                    for DC in DC_subsets[cell_numbers[1]]:
                        free_Tcells[cell_numbers] = free_Tcells[cell_numbers] \
                            - set(current_contacts[DC])

            for cell_numbers in free_Tcells:
                contacts.loc[max_index, 'Time'] = time
                contacts.loc[max_index, 'Run'] = n
                contacts.loc[max_index, 'Contact Radius'] = contact_radius
                contacts.loc[max_index, 'Cell Numbers'] = \
                    '{} T cells, {} DCs'.format(*cell_numbers)
                contacts.loc[max_index, 'Contacts'] = \
                    cell_numbers[0] - len(free_Tcells[cell_numbers])
                max_index += 1

        print('  Run {} done.'.format(n + 1))

    return contacts


def plot_over_time(contacts):
    """Plot contacts over time"""
    sns.set(style="white")
    plt.xlabel('Time [h]')
    plt.ylabel('# of Contacts')

    for i, (label, _contacts) in enumerate(contacts.groupby('Cell Numbers')):
        color = sns.color_palette(n_colors=i+1)[-1]
        stats = _contacts.groupby('Time')['Contacts'].describe().unstack()
        plt.plot(stats.index/60, stats['50%'], label=label, color=color)
        plt.fill_between(stats.index/60, stats['25%'], stats['75%'], alpha=0.2, color=color)

    plt.legend(loc='lower right')
    plt.show()


if __name__ == '__main__':
    import motility
    from remix import silly_tracks

    tracks = silly_tracks(25, 120)
    # motility.plot_tracks(tracks, ln_volume=5e6)
    contacts = by_distance(tracks, ln_volume=5e6)
    plot_over_time(contacts)
