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
    if 'Appearance Time' in DCs.columns:
        available_DCs = pd.DataFrame()
    else:
        DC_tree = spatial.cKDTree(DCs[['X', 'Y', 'Z']])
        available_DCs = DCs
    free_T_cells = set(tracks['Track_ID'].unique())
    contacts = pd.DataFrame()
    max_index = 0
    for time, positions in tracks.sort('Time').groupby('Time'):
        if 'Appearance Time' not in DCs.columns:
            pass
        elif len(DCs[DCs['Appearance Time'] <= time]) == 0:
            continue
        elif len(available_DCs) != len(DCs[DCs['Appearance Time'] <= time]):
            available_DCs = DCs[DCs['Appearance Time'] <= time].reset_index()
            DC_tree = spatial.cKDTree(available_DCs[['X', 'Y', 'Z']])
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
                    contacts.loc[max_index, 'X'] = available_DCs.loc[DC, 'X']
                    contacts.loc[max_index, 'Y'] = available_DCs.loc[DC, 'Y']
                    contacts.loc[max_index, 'Z'] = available_DCs.loc[DC, 'Z']
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


def find_pairs(tracks, T_cell_ns=(10, 20), DC_ns=(10, 50), min_distances=(0,),
    min_dist_stds=(0,), contact_radii=(10,), tcz_volume=0.125e9/100, n_iter=10):
    """Simulate ensemble of pair-wise T cell/DC contacts within radius"""
    print('\nSimulating pair-wise contacts {} times'.format(n_iter))
    assert max(T_cell_ns) < tracks['Track_ID'].unique().__len__(),\
        'Max. T_cell_ns is larger than # of given tracks.'

    pairs = pd.DataFrame()
    for n_run in range(n_iter):
        for min_dist, min_std, cr, nT, nDC in itertools.product(min_distances,
            min_dist_stds, contact_radii, T_cell_ns, DC_ns):
            T_tracks = tracks[tracks['Track_ID'].isin(
                np.random.choice(tracks['Track_ID'].unique(), nT,
                replace=False))].copy()
            if min_std != 0:
                # Such noise makes contacts seem to be none!
                for id, track in T_tracks.groupby('Track_ID'):
                    T_tracks.loc[T_tracks['Track_ID'] == id, 'X'] += \
                        np.random.randn()*min_std
                    T_tracks.loc[T_tracks['Track_ID'] == id, 'Y'] += \
                        np.random.randn()*min_std
                    T_tracks.loc[T_tracks['Track_ID'] == id, 'Z'] += \
                        np.random.randn()*min_std

            tcz_radius = (3*tcz_volume/(4*np.pi))**(1/3)
            ratio = (min_dist/tcz_radius)**3
            r = tcz_radius*(ratio + (1 - ratio)*np.random.rand(nDC))**(1/3)
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
            run_pairs['Minimal Initial Distance'] = min_dist
            run_pairs['Std. of Initial Position'] = min_std
            pairs = pairs.append(run_pairs)

        print('  Run {} done.'.format(n_run+1))

    # Save duration and number of runs for analysis
    pairs.reset_index(drop=True, inplace=True)
    max_index = pairs.index.max()
    pairs.loc[max_index + 1, 'Time'] = tracks['Time'].max()
    pairs.loc[max_index + 1, 'Run'] = n_iter - 1

    return pairs


def find_pairs_and_triples(CD4_tracks, CD8_tracks, CD4_ns=(10,), CD8_ns=(10,),
    DC_ns=(50,), CD8_delays=(0,), contact_radii=(10,), licensing_factors=(1, 2, 4),
    tcz_volume=0.125e9/100, n_iter=10):
    """Simulate ensemble of triple contacts allowing CD4/DC and CD8/DC pairs"""
    print('\nSimulating triple contacts allowing CD4/DC & CD8/DC pairs {} times'
        .format(n_iter))
    assert max(CD4_ns) < CD4_tracks['Track_ID'].unique().__len__(),\
        'Max. CD4_ns is larger than # of given CD4+ tracks.'
    assert max(CD8_ns) < CD8_tracks['Track_ID'].unique().__len__(),\
        'Max. CD8_ns is larger than # of given CD8+ tracks.'

    CD4_pairs = pd.DataFrame()
    CD8_pairs = pd.DataFrame()
    triples = pd.DataFrame()
    max_index = 0
    for n_run in range(n_iter):
        for cr, lic_fac, n4, n8, nDC, delay in itertools.product(contact_radii,
            licensing_factors, CD4_ns, CD8_ns, DC_ns, CD8_delays):
            # Create DCs
            tcz_radius = (3*tcz_volume/(4*np.pi))**(1/3)
            r = tcz_radius*np.random.rand(nDC)**(1/3)
            theta = np.random.rand(nDC)*2*np.pi
            phi = np.arccos(2*np.random.rand(nDC) - 1)
            DCs = pd.DataFrame({
                'X': r*np.sin(theta)*np.sin(phi),
                'Y': r*np.cos(theta)*np.sin(phi),
                'Z': r*np.cos(phi)})

            # Find CD4-DC-Pairs
            T_tracks = CD4_tracks[CD4_tracks['Track_ID'].isin(
                np.random.choice(CD4_tracks['Track_ID'].unique(), n4,
                replace=False))]
            run_CD4_pairs = _find_by_distance(T_tracks, DCs, cr, tcz_radius)
            run_CD4_pairs['Run'] = n_run
            run_CD4_pairs['Cell Numbers'] = \
                '{} CD4+ T cells, {} CD8+ T cells, {} DCs'.format(n4, n8, nDC)
            run_CD4_pairs['Contact Radius'] = cr
            run_CD4_pairs['Licensing Factor'] = lic_fac
            run_CD4_pairs['CD8 Delay'] = delay
            CD4_pairs = CD4_pairs.append(run_CD4_pairs)

            # Find CD8-DC-Pairs
            T_tracks = CD8_tracks[CD8_tracks['Track_ID'].isin(
                np.random.choice(CD8_tracks['Track_ID'].unique(), n8,
                replace=False))].copy()
            T_tracks['Time'] = T_tracks['Time'] + delay
            run_CD8_pairs = _find_by_distance(T_tracks, DCs, cr, tcz_radius)
            run_CD8_pairs['Run'] = n_run
            run_CD8_pairs['Cell Numbers'] = \
                '{} CD4+ T cells, {} CD8+ T cells, {} DCs'.format(n4, n8, nDC)
            run_CD8_pairs['Contact Radius'] = cr
            run_CD8_pairs['Licensing Factor'] = lic_fac
            run_CD8_pairs['CD8 Delay'] = delay
            CD8_pairs = CD8_pairs.append(run_CD8_pairs)

            # Find pairs among CD8s and DCs licensed by CD4s
            if lic_fac != 1:
                for idx, DC in DCs.iterrows():
                    try:
                        DC_contacts = run_CD4_pairs[
                            np.isclose(run_CD4_pairs['X'], DC['X']) &
                            np.isclose(run_CD4_pairs['Y'], DC['Y']) &
                            np.isclose(run_CD4_pairs['Z'], DC['Z'])]
                        DCs.loc[idx, 'Appearance Time'] = DC_contacts['Time'].min()
                    except KeyError:
                        continue
                DCs = DCs.dropna().reset_index(drop=True)
                lic_CD8_pairs = _find_by_distance(T_tracks, DCs, cr*lic_fac,
                    tcz_radius)
                lic_CD8_pairs['Run'] = n_run
                lic_CD8_pairs['Cell Numbers'] = \
                    '{} CD4+ T cells, {} CD8+ T cells, {} DCs'.format(n4, n8, nDC)
                lic_CD8_pairs['Contact Radius'] = cr
                lic_CD8_pairs['CD8 Delay'] = delay
                run_CD8_pairs = run_CD8_pairs.append(lic_CD8_pairs)
                try:
                    run_CD8_pairs = run_CD8_pairs.sort('Time').drop_duplicates(
                        'Track_ID')
                except KeyError:
                    pass

            # Check for triples
            run_triples = pd.DataFrame() # For assertion (and evlt. performance)
            for _, pair in run_CD8_pairs.iterrows():
                try:
                    pair_triples = run_CD4_pairs[
                        np.isclose(run_CD4_pairs['X'], pair['X']) &
                        np.isclose(run_CD4_pairs['Y'], pair['Y']) &
                        np.isclose(run_CD4_pairs['Z'], pair['Z'])]
                    closest_CD4_pair = pair_triples.loc[
                        (pair_triples['Time'] - pair['Time']).abs().idxmin(), :]
                except (KeyError, ValueError):
                    continue
                run_triples.loc[max_index, 'CD8 Track_ID'] = pair['Track_ID']
                run_triples.loc[max_index, 'CD4 Track_ID'] = closest_CD4_pair['Track_ID']
                run_triples.loc[max_index, 'Time'] = pair['Time']
                run_triples.loc[max_index, 'Time Between Contacts'] = pair['Time']\
                    - closest_CD4_pair['Time']
                run_triples.loc[max_index, 'Run'] = n_run
                run_triples.loc[max_index, 'Cell Numbers'] = \
                    '{} CD4+ T cells, {} CD8+ T cells, {} DCs'.format(n4, n8, nDC)
                run_triples.loc[max_index, 'Contact Radius'] = cr
                run_triples.loc[max_index, 'Licensing Factor'] = lic_fac
                run_triples.loc[max_index, 'CD8 Delay'] = \
                    '{} min. between injections, priming not required'.format(delay)
                max_index += 1
            try:
                n_triples_of_run = len(run_triples)
            except KeyError:
                n_triples_of_run = 0
            try:
                n_DC8_pairs_of_run = len(run_CD8_pairs)
            except KeyError:
                n_DC8_pairs_of_run = 0
            assert n_triples_of_run <= n_DC8_pairs_of_run, \
                'More triples found than possible.'
            for _, triple in run_triples.iterrows():
                CD8_position = CD8_tracks[
                    (CD8_tracks['Track_ID'] == triple['CD8 Track_ID']) &
                    (CD8_tracks['Time'] == triple['Time'])][['X', 'Y', 'Z']]
                CD4_contact_time = triple['Time'] - triple['Time Between Contacts']
                CD4_position = CD4_tracks[
                    (CD4_tracks['Track_ID'] == triple['CD4 Track_ID']) &
                    np.isclose(CD4_tracks['Time'], CD4_contact_time)][['X', 'Y', 'Z']]
                distance = np.linalg.norm(CD4_position.values - CD8_position.values)
                assert  distance <= cr*(1 + lic_fac), 'Triple too far apart.'
            triples = triples.append(run_triples)

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


def plot_details(contacts, tracks, parameters='Contact Radius'):
    """Plot distances over time and time within contact radius"""
    sns.set(style='ticks')
    figure, axes = plt.subplots(ncols=3, figsize=(12,6))

    axes[0].set_xlabel('Time [min]')
    axes[0].set_ylabel(r'Distance [$\mu$m]')

    axes[1].set_xlabel('Time within Contact Radius [min]')
    axes[1].set_ylabel('Number of Contacts')

    axes[2].set_xlabel('Contact Time [h]')
    axes[2].set_ylabel('Distance from Origin')

    for i, (cond, cond_contacts) in enumerate(contacts.groupby(parameters)):
        if len(cond_contacts['Contact Radius'].dropna().unique()) != 1:
            raise ValueError('Condition with more than one contact radius')
        radius = cond_contacts['Contact Radius'].max()
        color = sns.color_palette(n_colors=i+1)[-1]
        distances = pd.Series()
        durations = []
        for _, contact in cond_contacts.dropna().iterrows():
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
        axes[0].plot(distats.index, distats['50%'], color=color)
        axes[0].fill_between(distats.index, distats['25%'], distats['75%'],
            color=color, alpha=0.2)
        axes[0].fill_between(distats.index, distats['min'], distats['max'],
            color=color, alpha=0.2)

        sns.distplot(durations, bins=np.arange(20 + 1), kde=False, ax=axes[1],
            color=color)

        axes[2].scatter(cond_contacts['Time']/60,
            np.linalg.norm(cond_contacts.dropna()[['X', 'Y', 'Z']], axis=1),
            color=color, label=cond)
        axes[2].legend(loc=4)

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_numbers(contacts, parameters='Cell Numbers'):
    """Plot accumulation and final number of T cells in contact with DC"""
    T_cells_in_contact = contacts.drop_duplicates(['Track_ID', 'Run', parameters])

    sns.set(style='ticks')

    n_parameter_sets = len(T_cells_in_contact[parameters].unique()) - 1 # nan for t_end
    gs = gridspec.GridSpec(n_parameter_sets,2)
    final_ax = plt.subplot(gs[:,0])
    ax0 = plt.subplot(gs[1])

    final_ax.set_title('Final State')
    final_ax.set_ylabel('Percentage of Final T Cells in Contact')
    ax0.set_title('Dynamics')

    final_sum = T_cells_in_contact.groupby(parameters).count()['Time']
    order = list(final_sum.order().index.values)

    for label, _contacts in T_cells_in_contact.groupby(parameters):
        i = order.index(label)
        n_runs = T_cells_in_contact['Run'].max() + 1
        label = '  ' + str(label) + ' (n = {:.0f})'.format(n_runs)
        final_ax.text(i*2 - 0.5, 0, label, rotation=90, va='bottom')

        if i == 0:
            ax = ax0
            ax.set_yticks([0, 50, 100])
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
        runs_with_geq_n_contacts.loc[T_cells_in_contact['Time'].max(), :] = \
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
    final_ax.set_yticks([0, 25, 50, 75, 100])
    final_ax.set_ylim([0,100])
    ax.set_ylim([0,100])

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_triples(pairs_and_triples, parameters='CD8 Delay'):
    """Plot # of CD8+ T cells in contact and times between 1st and 2nd contact"""
    CD8_in_contact = pairs_and_triples['Triples'].drop_duplicates(
        ['CD8 Track_ID', 'Run', parameters])

    sns.set(style='ticks')

    final_ax = plt.subplot(1,2,1)
    timing_ax = plt.subplot(1,2,2)

    final_ax.set_title('Final State')
    final_ax.set_ylabel('Percentage of Final CD8+ T Cells in Contacts')
    timing_ax.set_title('Time Between Contacts')
    timing_ax.set_yticks([])

    final_sum = CD8_in_contact.groupby(parameters).count()['Time']
    order = list(final_sum.order().index.values)

    for label, _triples in CD8_in_contact.groupby(parameters):
        i = order.index(label)
        n_runs = CD8_in_contact['Run'].max() + 1
        label = '  ' + str(label) + ' (n = {:.0f})'.format(n_runs)
        final_ax.text(i*2 - 0.5, 0, label, rotation=90, va='bottom')

        color = sns.color_palette(n_colors=i+1)[-1]

        accumulation = _triples[['Run', 'Time']].pivot_table(
            columns='Run', index='Time', aggfunc=len, fill_value=0).cumsum()
        runs_with_n_contacts = accumulation.apply(lambda x: x.value_counts(), axis=1).fillna(0)
        runs_with_n_contacts = runs_with_n_contacts[runs_with_n_contacts.columns[::-1]]
        runs_with_geq_n_contacts = runs_with_n_contacts.cumsum(axis=1)
        runs_with_geq_n_contacts.loc[CD8_in_contact['Time'].max(), :] = \
            runs_with_geq_n_contacts.iloc[-1]

        for n_contacts in [n for n in runs_with_geq_n_contacts.columns if n > 0]:
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

        bins = np.arange(CD8_in_contact['Time Between Contacts'].min(),
            CD8_in_contact['Time Between Contacts'].max(), 15)/60
        sns.distplot(_triples['Time Between Contacts']/60, kde=False, bins=bins,
            norm_hist=True, color=color, ax=timing_ax, axlabel='Time [h]')

    final_ax.set_xlim(left=-0.8)
    final_ax.set_xticks([])
    final_ax.set_yticks([0,25,50,75,100])
    final_ax.set_ylim([0,100])

    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_triples_vs_pairs(triples, parameters='Cell Numbers'):
    """Scatter plot pure CD8-DC-Pairs vs Triples per run"""
    pairs = triples['CD8-DC-Pairs']
    triples = triples['Triples']

    contact_numbers = pd.DataFrame()
    max_index = 0
    for run, par in itertools.product(range(int(pairs['Run'].max()) + 1),
        pairs[parameters].dropna().unique()):
        contact_numbers.loc[max_index, 'Run'] = run
        contact_numbers.loc[max_index, 'Parameter'] = par
        CD8_in_triples = set(triples[(triples['Run'] == run) &
            (triples[parameters] == par)]['CD8 Track_ID'])
        contact_numbers.loc[max_index, '# CD8 in Triples'] = \
            len(CD8_in_triples)
        CD8_in_pairs = set(pairs[(pairs['Run'] == run) &
            (pairs[parameters] == par)]['Track_ID'])
        contact_numbers.loc[max_index, '# CD8 in Pairs'] = \
            len(CD8_in_pairs.difference(CD8_in_triples))
        max_index += 1

    sns.set(style='ticks')
    # sns.set_style({"xtick.direction": "in","ytick.direction": "in"})
    igure, axes = plt.subplots(ncols=2, figsize=(11,5.5))
    axes[0].set_xlabel('# CD8 in Triples')
    axes[0].set_ylabel('# CD8 in Pairs')
    axes[1].set_xlabel('arctan of # Triples/# Pairs')
    axes[1].set_ylabel('Numbers of Simulations')
    legend = []
    for i, (par, numbers) in enumerate(contact_numbers.groupby('Parameter')):
        color = sns.color_palette(n_colors=i+1)[-1]
        axes[0].scatter(numbers['# CD8 in Triples'] + np.random.rand(len(numbers))/2,
            numbers['# CD8 in Pairs'] + np.random.rand(len(numbers))/2,
            color=color)
        ratios = np.arctan(numbers['# CD8 in Triples']/numbers['# CD8 in Pairs'])
        sns.distplot(ratios, hist=True, kde=False, color=color, ax=axes[1],
            bins=np.arange(21)*np.pi/40)
        legend.append(par)
    axes[0].legend(legend)
    sns.despine()
    plt.tight_layout()
    plt.show()


def plot_situation(tracks, n_tracks=6*3, n_DCs=50, tcz_volume=0.125e9/100,
    min_distance=0, zoom=1):
    """Plot some T cell tracks, DC positions and T cell zone volume"""
    sns.set_style('ticks')

    gs = gridspec.GridSpec(1,3)
    space_ax = plt.subplot(gs[:,:-1], projection='3d')
    time_ax = plt.subplot(gs[:,-1])

    space_ax.set_title('{} T Cell Tracks & {} DCs'.format(n_tracks, n_DCs))

    n_conditions = len(tracks['Condition'].unique())
    palette = itertools.cycle(sns.color_palette())
    for i, (cond, cond_tracks) in enumerate(tracks.groupby('Condition')):
        choice = np.random.choice(cond_tracks['Track_ID'].unique(),
            n_tracks/n_conditions)
        chosen_tracks = cond_tracks[cond_tracks['Track_ID'].isin(choice)]
        for _, track in chosen_tracks.groupby(track_identifiers(chosen_tracks)):
            if n_conditions > 1:
                color = sns.color_palette(n_colors=i+1)[-1]
            else:
                color = next(palette)
            space_ax.plot(track['X'].values, track['Y'].values, track['Z'].values,
                color=color)

    tcz_radius = (3*tcz_volume/(4*np.pi))**(1/3)
    ratio = (min_distance/tcz_radius)**3
    r = tcz_radius*(ratio + (1 - ratio)*np.random.rand(n_DCs))**(1/3)
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

    def _residence_time(track): return track['Time'].diff().mean()/60*len(
        track[np.linalg.norm(track[['X', 'Y', 'Z']], axis=1) < r])

    for cond, cond_tracks in tracks.groupby('Condition'):
        residence_times = [_residence_time(track)
            for _, track in cond_tracks.groupby('Track_ID')]

        sns.distplot(residence_times, kde=False, norm_hist=True, ax=time_ax,
            label=cond)

    time_ax.legend()
    sns.despine(ax=time_ax)
    equalize_axis3d(space_ax, zoom)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    import motility
    from remix import silly_tracks

    tracks = silly_tracks(25, 180)
    tracks['Time'] = tracks['Time']/3
    plot_situation(tracks, n_tracks=10, n_DCs=200, min_distance=60)

    # pairs = find_pairs(tracks)
    # plot_details(pairs, tracks)
    # plot_numbers(pairs)

    pairs_and_triples = find_pairs_and_triples(tracks, tracks)
    # plot_numbers(pairs_and_triples['CD8-DC-Pairs'], parameters='CD8 Delay')
    # plot_numbers(pairs_and_triples['Triples'], parameters='CD8 Delay')
    plot_triples(pairs_and_triples, parameters='Licensing Factor')
    plot_triples_vs_pairs(pairs_and_triples, parameters='Licensing Factor')
