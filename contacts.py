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


def _find_by_distance(tracks, DCs, contact_radius, tcz_radius):
    """Find contacts among T-cell tracks and DC positions"""
    free_T_cells = set(tracks['Track_ID'].unique())
    DC_tree = spatial.cKDTree(DCs)
    contacts = pd.DataFrame()
    max_index = 0
    for time, positions in tracks.sort('Time').groupby('Time'):
        positions = positions[positions['Track_ID'].isin(free_T_cells)]
        positions = positions[np.linalg.norm(positions[['X', 'Y', 'Z']],
            axis=1) < (tcz_radius + contact_radius)]
        if positions.__len__() != 0:
            T_cell_tree = spatial.cKDTree(positions[['X', 'Y', 'Z']])
            new_contacts = DC_tree.query_ball_tree(
                T_cell_tree, contact_radius)
            for DC, DC_contacts in enumerate(new_contacts):
                for T_cell in DC_contacts:
                    contacts.loc[max_index, 'Time'] = time
                    contacts.loc[max_index, 'Track_ID'] = \
                        positions.iloc[T_cell]['Track_ID']
                    contacts.loc[max_index, 'X'] = DCs.loc[DC, 'X']
                    contacts.loc[max_index, 'Y'] = DCs.loc[DC, 'Y']
                    contacts.loc[max_index, 'Z'] = DCs.loc[DC, 'Z']
                    max_index += 1
                    try:
                        free_T_cells.remove(
                            positions.iloc[T_cell]['Track_ID'])
                    except KeyError:
                        print('  Warning: T cell binding two DCs.')

    if len(contacts) != 0:
        n_twice_bound = \
            contacts['Track_ID'].duplicated().sum()
        n_twice_bound_at_same_time = \
            contacts[['Track_ID', 'Time']]\
            .duplicated().sum()
        assert n_twice_bound == n_twice_bound_at_same_time,\
            'T cells were in contacts at different times.'

    return contacts


def find_pairs(tracks, n_Tcells=[10,20], n_DCs=[25,50], n_iter=10,
    tcz_volume=0.125e9/100, contact_radius=10):
    """Simulate ensemble of pair-wise T cell/DC contacts within radius"""
    print('\nSimulating pair-wise contacts {} times'.format(n_iter))

    if type(n_Tcells) == int:
        n_Tcells = [n_Tcells]
    if type(n_DCs) == int:
        n_DCs = [n_DCs]
    if type(contact_radius) != list:
        contact_radius = [contact_radius]
    if max(n_Tcells) > tracks['Track_ID'].unique().__len__():
        print('Max. n_Tcells is larger than # of given tracks.')
        return

    pairs = pd.DataFrame()
    for n_run in range(n_iter):
        for cr, nT, nDC in itertools.product(contact_radius, n_Tcells, n_DCs):
            T_tracks = tracks[tracks['Track_ID'].isin(
                np.random.choice(tracks['Track_ID'].unique(), nT,
                replace=False))]

            tcz_radius = (3*tcz_volume/(4*np.pi))**(1/3)
            r = tcz_radius*np.random.rand(nDC)**(1/3)
            theta = np.random.rand(nDC)*2*np.pi
            phi = np.arccos(2*np.random.rand(nDC) - 1)
            DCs = pd.DataFrame({
                'X': r*np.sin(theta)*np.sin(phi),
                'Y': r*np.cos(theta)*np.sin(phi),
                'Z': r*np.cos(phi)})

            run_pairs = _find_by_distance(T_tracks, DCs, cr, tcz_radius)
            run_pairs['Run'] = n_run
            run_pairs['Cell Numbers'] = \
                '{} T cells, {} DCs'.format(nT, nDC)
            run_pairs['Contact Radius'] = cr
            pairs = pairs.append(run_pairs)

        print('  Run {} done.'.format(n_run+1))

    # Save duration and number of runs for analysis
    pairs.reset_index(drop=True, inplace=True)
    max_index = pairs.index.max()
    pairs.loc[max_index + 1, 'Time'] = tracks['Time'].max()
    pairs.loc[max_index + 1, 'Run'] = n_iter - 1

    return pairs


def find_pairs_and_triples(CD4_tracks, CD8_tracks, n_CD4=20, n_CD8=10, n_DCs=50,
    CD8_delay=[0, 15], n_iter=10, tcz_volume=0.125e9/100, contact_radius=10):
    """Simulate ensemble of triple contacts allowing CD4/DC and CD8/DC pairs"""
    print('\nSimulating triple contacts allowing CD4/DC & CD8/DC pairs {} times'
        .format(n_iter))

    if type(n_CD4) == int:
        n_CD4 = [n_CD4]
    if type(n_CD8) == int:
        n_CD8 = [n_CD8]
    if type(n_DCs) == int:
        n_DCs = [n_DCs]
    if type(CD8_delay) != list:
        CD8_delay = [CD8_delay]
    if type(contact_radius) != list:
        contact_radius = [contact_radius]
    if max(n_CD4) > CD4_tracks['Track_ID'].unique().__len__():
        print('Max. n_CD4 is larger than # of given CD4+ tracks.')
        return
    if max(n_CD8) > CD8_tracks['Track_ID'].unique().__len__():
        print('Max. n_CD8 is larger than # of given CD8+ tracks.')
        return

    CD4_pairs = pd.DataFrame()
    CD8_pairs = pd.DataFrame()
    triples = pd.DataFrame()
    for n_run in range(n_iter):
        for cr, n4, n8, nDC, delay in itertools.product(contact_radius, n_CD4,
            n_CD8, n_DCs, CD8_delay):
            tcz_radius = (3*tcz_volume/(4*np.pi))**(1/3)
            r = tcz_radius*np.random.rand(nDC)**(1/3)
            theta = np.random.rand(nDC)*2*np.pi
            phi = np.arccos(2*np.random.rand(nDC) - 1)
            DCs = pd.DataFrame({
                'X': r*np.sin(theta)*np.sin(phi),
                'Y': r*np.cos(theta)*np.sin(phi),
                'Z': r*np.cos(phi)})

            T_tracks = CD4_tracks[CD4_tracks['Track_ID'].isin(
                np.random.choice(CD4_tracks['Track_ID'].unique(), n4,
                replace=False))]
            run_CD4_pairs = _find_by_distance(T_tracks, DCs, cr, tcz_radius)
            run_CD4_pairs['Run'] = n_run
            run_CD4_pairs['Cell Numbers'] = \
                '{} CD4+ T cells, {} CD8+ T cells, {} DCs'.format(n4, n8, nDC)
            run_CD4_pairs['Contact Radius'] = cr
            run_CD4_pairs['CD8 Delay'] = delay
            CD4_pairs = CD4_pairs.append(run_CD4_pairs)

            T_tracks = CD8_tracks[CD8_tracks['Track_ID'].isin(
                np.random.choice(CD8_tracks['Track_ID'].unique(), n8,
                replace=False))]
            T_tracks.loc[:, 'Time'] = T_tracks['Time'] + delay # gives warning
            run_CD8_pairs = _find_by_distance(T_tracks, DCs, cr, tcz_radius)
            run_CD8_pairs['Run'] = n_run
            run_CD8_pairs['Cell Numbers'] = \
                '{} CD4+ T cells, {} CD8+ T cells, {} DCs'.format(n4, n8, nDC)
            run_CD8_pairs['Contact Radius'] = cr
            run_CD8_pairs['CD8 Delay'] = delay
            CD8_pairs = CD8_pairs.append(run_CD8_pairs)

            run_triples = pd.DataFrame()
            for _, pair in run_CD8_pairs.iterrows():
                try:
                    pair_triples = run_CD4_pairs[
                        np.isclose(run_CD4_pairs['X'], pair['X']) &
                        np.isclose(run_CD4_pairs['Y'], pair['Y']) &
                        np.isclose(run_CD4_pairs['Z'], pair['Z'])]
                except KeyError:
                    pair_triples = []
                if len(pair_triples) > 0:
                    pair_triples['CD4 Track_ID'] = pair_triples['Track_ID']
                    pair_triples['CD8 Track_ID'] = pair['Track_ID']
                    pair_triples['Time Between Contacts'] = pair['Time']\
                        - pair_triples['Time']
                    pair_triples['Run'] = n_run
                    pair_triples['Cell Numbers'] = \
                        '{} CD4+ T cells, {} CD8+ T cells, {} DCs'.format(n4, n8, nDC)
                    pair_triples['Contact Radius'] = cr
                    pair_triples['CD8 Delay'] = \
                        '{} minutes between CD4 and CD8 injection'.format(delay)
                    pair_triples['Priming'] = 'Not Required'
                    pair_triples.drop('Track_ID', axis=1, inplace=True)
                    for idx, triple in pair_triples.iterrows():
                        if triple['Time'] < pair['Time']:
                            pair_triples.loc[idx, 'Time'] = pair['Time']
                    run_triples = run_triples.append(pair_triples)
            triples = triples.append(run_triples)

            assert len(run_triples) <= len(run_CD4_pairs)*len(run_CD8_pairs), \
                'More triples found than possible.'
            # TODO: Assert distance between CD4 and CD8 < 2*contact_radius

        print('  Run {} done.'.format(n_run+1))

    # Save duration and number of runs for analysis
    for df, tracks in zip([CD4_pairs, CD8_pairs, triples],
        [CD4_tracks, CD8_tracks, CD4_tracks]):
        df.reset_index(drop=True, inplace=True)
        max_index = df.index.max()
        df.loc[max_index + 1, 'Time'] = tracks['Time'].max()
        df.loc[max_index + 1, 'Run'] = n_iter - 1

    return pd.Panel({'CD4-DC-Pairs': CD4_pairs, 'CD8-DC-Pairs': CD8_pairs,
        'Triples': triples})


def plot_details(contacts, tracks):
    """Plot distances over time and time within contact radius"""
    sns.set(style='white')
    distance_ax = plt.subplot(1,2,1)
    duration_ax = plt.subplot(1,2,2)

    distance_ax.set_xlabel('Time [min]')
    distance_ax.set_ylabel(r'Distance [$\mu$m]')

    duration_ax.set_xlabel('Time within Contact Radius [min]')
    duration_ax.set_ylabel('Density of Contacts')

    for i, (radius, _contacts) in enumerate(contacts.groupby('Contact Radius')):
        color = sns.color_palette(n_colors=i+1)[-1]
        distances = pd.Series()
        durations = []
        for _, contact in _contacts.dropna().iterrows():
            track = tracks[tracks['Track_ID'] == contact['Track_ID']]
            track = track[['Time', 'X', 'Y', 'Z']]
            track = track[track['Time'] <= contact['Time'] + 20]
            track = track[track['Time'] >= contact['Time'] - 10]
            distance = pd.Series(
                np.linalg.norm(
                    track[['X', 'Y', 'Z']].astype(float)
                    - contact[['X', 'Y', 'Z']].astype(float), axis=1),
                    track['Time'] - contact['Time'])
            time_step = track['Time'].diff().mean()
            distances = distances.append(distance)
            durations.append(distance[distance <= radius].size*time_step)

        distances.index = np.round(distances.index, 5) # Handle non-integer 'Times'
        distats = distances.groupby(distances.index).describe().unstack()
        distance_ax.plot(distats.index, distats['50%'], color=color)
        distance_ax.fill_between(distats.index, distats['25%'], distats['75%'],
            color=color, alpha=0.2)
        distance_ax.fill_between(distats.index, distats['min'], distats['max'],
            color=color, alpha=0.2)

        sns.distplot(durations, bins=np.arange(20 + 1), kde=False, ax=duration_ax,
            color=color)

    plt.tight_layout()
    plt.show()


def plot_numbers(contacts, parameters='Cell Numbers'):
    """Plot accumulation and final number of contacts"""
    sns.set(style='white')

    n_parameter_sets = len(contacts[parameters].unique()) - 1 # nan for t_end
    gs = gridspec.GridSpec(n_parameter_sets,2)
    final_ax = plt.subplot(gs[:,0])
    ax0 = plt.subplot(gs[1])

    final_ax.set_title('Final State')
    final_ax.set_ylabel('Percentage of Final Contacts')
    ax0.set_title('Dynamics')

    final_sum = contacts.groupby(parameters).count()['Time']
    order = list(final_sum.order().index.values)

    for label, _contacts in contacts.groupby(parameters):
        i = order.index(label)
        n_runs = contacts['Run'].max() + 1
        label = '  ' + str(label) + ' (n = {:.0f})'.format(n_runs)
        final_ax.text(i*2 - 0.5, 0, label, rotation=90, va='bottom')

        if i == 0:
            ax = ax0
        else:
            ax = plt.subplot(gs[2*i+1], sharex=ax0, sharey=ax0)

        if i < n_parameter_sets - 1:
            plt.setp(ax.get_xticklabels(), visible=False)
        else:
            ax.set_xlabel('Time [h]')

        color = sns.color_palette(n_colors=i+1)[-1]

        accumulation = _contacts[['Run', 'Time']].pivot_table(
            columns='Run', index='Time', aggfunc=len, fill_value=0).cumsum()
        runs_with_n_contacts = accumulation.apply(lambda x: x.value_counts(), axis=1).fillna(0)
        runs_with_n_contacts = runs_with_n_contacts[runs_with_n_contacts.columns[::-1]]
        runs_with_geq_n_contacts = runs_with_n_contacts.cumsum(axis=1)
        runs_with_geq_n_contacts.loc[contacts['Time'].max(), :] = \
            runs_with_geq_n_contacts.iloc[-1]

        for n_contacts in [n for n in runs_with_geq_n_contacts.columns if n > 0]:
            ax.fill_between(runs_with_geq_n_contacts[n_contacts].index/60, 0,
                runs_with_geq_n_contacts[n_contacts].values/n_runs*100,
                color=color, alpha=1/runs_with_n_contacts.columns.max())

            percentage = runs_with_geq_n_contacts[n_contacts].iloc[-1]/n_runs*100
            final_ax.bar(i*2, percentage, color=color,
                alpha=1/runs_with_n_contacts.columns.max())

            if n_contacts == runs_with_geq_n_contacts.columns.max():
                next_percentage = 0
            else:
                next_n = next(n for n in runs_with_geq_n_contacts.columns[::-1]
                    if n > n_contacts)
                next_percentage = runs_with_geq_n_contacts[next_n].iloc[-1]/n_runs*100

            percentage_diff = percentage - next_percentage
            if percentage_diff > 3:
                final_ax.text(i*2 + 0.38, percentage - percentage_diff/2 - 0.5,
                    int(n_contacts), ha='center', va='center')

    final_ax.set_xlim(left=-0.8)
    final_ax.set_xticks([])
    final_ax.set_ylim([0,100])
    ax.set_ylim([0,100])

    plt.tight_layout()
    plt.show()


def plot_situation(tracks, n_DCs=50, tcz_volume=0.125e9/100, zoom=1):
    """Plot some T cell tracks, DC positions and T cell zone volume"""
    sns.set_style('white')

    gs = gridspec.GridSpec(1,3)
    space_ax = plt.subplot(gs[:,:-1], projection='3d')
    time_ax = plt.subplot(gs[:,-1])

    n_tracks = 6*3
    space_ax.set_title('{} T Cell Tracks & {} DCs'.format(n_tracks, n_DCs))
    choice = np.random.choice(tracks['Track_ID'].unique(), n_tracks)
    chosen_tracks = tracks[tracks['Track_ID'].isin(choice)]
    for _, track in chosen_tracks.groupby(track_identifiers(tracks)):
        space_ax.plot(track['X'].values, track['Y'].values, track['Z'].values)

    r = (3*tcz_volume/(4*np.pi))**(1/3)*np.random.rand(n_DCs)**(1/3)
    theta = np.random.rand(n_DCs)*2*np.pi
    phi = np.arccos(2*np.random.rand(n_DCs) - 1)
    DCs = pd.DataFrame({
        'X': r*np.sin(theta)*np.sin(phi),
        'Y': r*np.cos(theta)*np.sin(phi),
        'Z': r*np.cos(phi)})
    space_ax.scatter(DCs['X'], DCs['Y'], DCs['Z'], color='y')

    r = (3*tcz_volume/(4*np.pi))**(1/3)
    for i in ['x', 'y', 'z']:
        circle = Circle((0, 0), r, fill=False, linewidth=2)
        space_ax.add_patch(circle)
        art3d.pathpatch_2d_to_3d(circle, z=0, zdir=i)

    time_ax.set_xlabel('Time within Lymph Node [h]')
    time_ax.set_ylabel('Number of T Cells')

    residence_time = lambda track: track['Time'].diff().mean()/60*len(
        track[np.linalg.norm(track[['X', 'Y', 'Z']], axis=1) < r])
    residence_times = [residence_time(track)
        for _, track in tracks.groupby('Track_ID')]

    sns.distplot(residence_times, kde=False, ax=time_ax)

    equalize_axis3d(space_ax, zoom)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import motility
    from remix import silly_tracks

    tracks = silly_tracks(25, 180)
    tracks['Time'] = tracks['Time']/3
    # plot_situation(tracks)

    # pairs = find_pairs(tracks)
    # plot_numbers(pairs)
    # plot_details(pairs, tracks)

    triples = find_pairs_and_triples(tracks, tracks)
    plot_numbers(triples['CD8-DC-Pairs'], parameters='CD8 Delay')
    plot_numbers(triples['Triples'], parameters='CD8 Delay')
