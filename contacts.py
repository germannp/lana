"""Analyze and plot contacts within lymph nodes"""
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial as spatial


def find(tracks, n_Tcells=[10,20], n_DCs=[50,100], n_iter=10,
    ln_volume=0.125e9, contact_radius=10):
    """Simulate contacts within radius"""
    print('Simulating contacts {} times'.format(n_iter))

    if type(n_Tcells) == int:
        n_Tcells = [n_Tcells]
    else:
        print('  Warning: Using subsets of T cells introduces bias.')

    if type(n_DCs) == int:
        n_DCs = [n_DCs]
    else:
        print('  Warning: Using subsets of T cells introduces bias.')

    if max(n_Tcells) > tracks['Track_ID'].unique().__len__():
        print('  Error: max. n_Tcells is larger than # of given tracks.')
        return

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


def plot(contacts):
    """Plot accumulation and final number of contacts"""
    sns.set(style="white")
    over_time_ax = plt.subplot2grid((1,3), (0,0), colspan=2)
    final_ax = plt.subplot2grid((1,3), (0,2), sharey=over_time_ax)

    over_time_ax.set_title('Accumulation over Time')
    over_time_ax.set_xlabel('Time [h]')
    over_time_ax.set_ylabel('# of Contacts')
    over_time_ax.set_ylim([0, contacts['Contacts'].max()])

    final_ax.set_title('Final Contacts')
    final_ax.set_xlabel('Density')

    for i, (label, _contacts) in enumerate(contacts.groupby('Cell Numbers')):
        color = sns.color_palette(n_colors=i+1)[-1]
        stats = _contacts.groupby('Time')['Contacts'].describe().unstack()
        over_time_ax.plot(stats.index/60, stats['50%'], label=label, color=color)
        over_time_ax.fill_between(stats.index/60, stats['25%'], stats['75%'],
            alpha=0.2, color=color)

        final_contacts = _contacts[_contacts['Time'] == _contacts['Time'].max()]
        sns.kdeplot(final_contacts['Contacts'], color=color, vertical=True,
            shade=True, legend=False, ax=final_ax)

    handles, labels = over_time_ax.get_legend_handles_labels()
    final_contacts = contacts[contacts['Time'] == contacts['Time'].max()]
    final_medians = final_contacts.groupby('Cell Numbers')['Contacts'].median()
    final_medians = final_medians.reset_index(drop=True)
    order = final_medians.order(ascending=False).index.values
    over_time_ax.legend([handles[i] for i in order], [labels[i] for i in order],
        loc='lower right')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import motility
    from remix import silly_tracks

    tracks = silly_tracks(25, 120)
    # motility.plot_tracks(tracks, ln_volume=5e6)
    contacts = find(tracks, ln_volume=5e6)
    plot(contacts)
