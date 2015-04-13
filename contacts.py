"""Analyze and plot contacts within lymph nodes"""
import itertools

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.spatial as spatial
import mpl_toolkits.mplot3d.axes3d as p3
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, PathPatch
from matplotlib.ticker import MaxNLocator

from utils import equalize_axis3d
from utils import track_identifiers


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
        print('  Warning: Using subsets of DCs introduces bias.')

    if max(n_Tcells) > tracks['Track_ID'].unique().__len__():
        print('  Error: max. n_Tcells is larger than # of given tracks.')
        return

    n_rows = len(n_Tcells)*len(n_DCs)*n_iter*len(tracks['Time'].unique())
    contacts = pd.DataFrame(index=np.arange(n_rows),
        columns=('Time', 'Run', 'Contact Radius', 'Cell Numbers', 'Contacts'))
    contacts[['Time', 'Contact Radius', 'Run', 'Contacts']] = \
        contacts[['Time', 'Contact Radius', 'Run', 'Contacts']].astype(float)
    max_index = 0
    for n in range(n_iter):
        ln_r = (3*ln_volume/(4*np.pi))**(1/3)
        r = ln_r*np.random.rand(max(n_DCs))**(1/3)
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
            positions = positions[positions['Track_ID'].isin(left_Tcells)]
            positions = positions[np.linalg.norm(positions[['X', 'Y', 'Z']],
                axis=1) < (ln_r + contact_radius)]
            if positions.__len__() != 0:
                Tcell_tree = spatial.cKDTree(positions[['X', 'Y', 'Z']])
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


def plot(contacts, parameters='Cell Numbers'):
    """Plot accumulation and final number of contacts"""
    sns.set(style='white')

    n_parameter_sets = len(contacts[parameters].unique())
    gs = gridspec.GridSpec(n_parameter_sets,2)
    final_ax = plt.subplot(gs[:,0])
    ax0 = plt.subplot(gs[1])

    final_ax.set_title('Final State')
    final_ax.set_ylabel('Percentage of Final Contacts')
    ax0.set_title('Dynamics')

    for i, (label, _contacts) in enumerate(contacts.groupby(parameters)):
        n = _contacts['Run'].max() + 1
        label = '  ' + label + ' (n = {:.0f})'.format(n)
        final_ax.text(i*2 - 0.5, 0, label, rotation=90, va='bottom')

        if i == 0:
            ax = ax0
        else:
            ax = plt.subplot(gs[2*i+1], sharex=ax0, sharey=ax0)

        if i < n_parameter_sets - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_ylim([0,100])
            ax.set_xlabel('Time [h]')

        total_contacts = _contacts.groupby(['Time', 'Contacts']).count()['Run'].unstack().fillna(0)
        total_contacts = total_contacts[total_contacts.columns[::-1]].cumsum(axis=1)
        color = sns.color_palette(n_colors=i+1)[-1]

        for n_contacts in sorted(set(_contacts['Contacts'].unique()) - set([0])):
            ax.fill_between(total_contacts.index/60, 0,
                total_contacts[n_contacts]/n*100,
                color=color, alpha=1/contacts['Contacts'].max())

            percentage = total_contacts[n_contacts].iloc[-1]/n*100
            try:
                next_percentage = total_contacts[n_contacts + 1].iloc[-1]/n*100
            except:
                if n_contacts == _contacts['Contacts'].max():
                    next_percentage = 0
                else:
                    next_percentage = percentage
            final_ax.bar(i*2, percentage, color=color,
                alpha=1/contacts['Contacts'].max())
            percentage_diff = percentage - next_percentage
            if percentage_diff > 3:
                final_ax.text(i*2 + 0.38, percentage - percentage_diff/2 - 0.5,
                    int(n_contacts), ha='center', va='center')

    final_ax.set_xlim(left=-0.8)
    final_ax.set_xticks([])
    final_ax.set_ylim([0,100])

    plt.tight_layout()
    plt.show()


def plot_situation(tracks, n_DCs=100, ln_volume=0.125e9, zoom=1):
    """Plot some tracks, DCs and volume"""
    sns.set_style('white')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1, projection='3d')

    choice = np.random.choice(tracks['Track_ID'].unique(), 6*3)
    tracks = tracks[tracks['Track_ID'].isin(choice)]
    for _, track in tracks.groupby(track_identifiers(tracks)):
        ax.plot(track['X'].values, track['Y'].values, track['Z'].values)

    r = (3*ln_volume/(4*np.pi))**(1/3)*np.random.rand(n_DCs)**(1/3)
    theta = np.random.rand(n_DCs)*2*np.pi
    phi = np.arccos(2*np.random.rand(n_DCs) - 1)
    DCs = pd.DataFrame({
        'X': r*np.sin(theta)*np.sin(phi),
        'Y': r*np.cos(theta)*np.sin(phi),
        'Z': r*np.cos(phi)})
    ax.scatter(DCs['X'], DCs['Y'], DCs['Z'], color='y')

    r = (3*ln_volume/(4*np.pi))**(1/3)
    for i in ['x', 'y', 'z']:
        circle = Circle((0, 0), r, fill=False, linewidth=2)
        ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)

    equalize_axis3d(ax, zoom)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import motility
    from remix import silly_tracks

    tracks = silly_tracks(25, 120)
    contacts = find(tracks, ln_volume=5e6)
    plot(contacts)

    # contacts = pd.read_csv('16h_contacts.csv')
    # plot(contacts)
