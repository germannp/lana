"""Analyze and plot cell motility from tracks"""
from collections import OrderedDict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering

from utils import equalize_axis3d
from utils import track_identifiers


def _uniquize_tracks(tracks):
    """Cluster tracks, if not unique"""
    if 'Time' not in tracks.columns:
        return

    tracks['Orig. Index'] = tracks.index
    if not tracks.index.is_unique:
        tracks.reset_index(drop=True, inplace=True)

    if 'Track_ID' in tracks.columns:
        max_track_id = tracks['Track_ID'].max()
    else:
        max_track_id = 0

    for identifiers, track in tracks.groupby(track_identifiers(tracks)):
        if sum(track['Time'].duplicated()) != 0:
            n_clusters = track['Time'].value_counts().max()
            track = track.copy()
            index = track.index
            if 'Track_ID' in track.columns:
                tracks.loc[index, 'Orig. Track_ID'] = track['Track_ID']

            clusters = AgglomerativeClustering(n_clusters).fit(
                track[['X', 'Y', 'Z']])
            track.loc[:, 'Cluster'] = clusters.labels_

            if sum(track[['Cluster', 'Time']].duplicated()) != 0:
                clusters = AgglomerativeClustering(n_clusters).fit(
                    track[['Orig. Index']])
                track.loc[:, 'Cluster'] = clusters.labels_

            if sum(track[['Cluster', 'Time']].duplicated()) == 0:
                tracks.loc[index, 'Track_ID'] = max_track_id+1+clusters.labels_
                max_track_id += n_clusters
                pd.set_option('display.max_rows', 1000)
                print('  Warning: Split non-unique track {} by clustering.'
                    .format(identifiers))
            else:
                tracks.drop(index, inplace=True)
                print('  Warning: Delete non-unique track {}.'
                    .format(identifiers))


def _split_at_skip(tracks):
    """Split tracks if timestep is missing in the original DataFrame"""
    if 'Time' not in tracks.columns:
        return

    if not tracks.index.is_unique:
        tracks.reset_index(drop=True, inplace=True)

    if 'Track_ID' in tracks.columns:
        max_track_id = tracks['Track_ID'].max()
    else:
        max_track_id = 0

    for criterium, track in tracks.groupby(track_identifiers(tracks)):
        timesteps = track['Time'].diff()
        skips = np.round((timesteps - timesteps.min())/timesteps.min())
        if skips.max() > 0:
            index = track.index
            if 'Track_ID' in track.columns:
                tracks.loc[index, 'Orig. Track_ID'] = track['Track_ID']
            skip_sum = skips.fillna(0).cumsum()
            tracks.loc[index, 'Track_ID'] = max_track_id + 1 + skip_sum
            max_track_id += max(skip_sum) + 1
            print('  Warning: Split track {} with non-uniform timesteps.'
                .format(criterium))


def _analyze(tracks, uniform_timesteps=True, min_length=6):
    """Calculate velocity, turning angle & plane angle"""
    if 'Displacement' in tracks.columns and tracks['Displacement'].isnull().sum() == 0:
        return
    else:
        print('\nAnalyzing tracks')

    if 'Time' not in tracks.columns:
        print('  Warning: no time given, using index!')
        tracks['Time'] = tracks.index
        if not tracks.index.is_unique: # For inplace analysis!
            tracks.reset_index(drop=True, inplace=True)
    else:
        _uniquize_tracks(tracks)
        if uniform_timesteps:
            _split_at_skip(tracks)

    for _, track in tracks.groupby(track_identifiers(tracks)):
        if track.__len__() < min_length:
            tracks.drop(track.index, inplace=True)
        else:
            tracks.loc[track.index, 'Track Time'] = \
                np.round(track['Time'] - track['Time'].iloc[0], 4)

            if 'Z' in track.columns:
                positions = track[['X', 'Y', 'Z']]
            else:
                positions = track[['X', 'Y']]

            tracks.loc[track.index, 'Displacement'] = \
                np.linalg.norm(positions - positions.iloc[0], axis=1)

            dr = positions.diff()
            dr_norms = np.linalg.norm(dr, axis=1)

            tracks.loc[track.index, 'Velocity'] = dr_norms/track['Time'].diff()

            dot_products = np.sum(dr.shift(-1)*dr, axis=1)
            norm_products = dr_norms[1:]*dr_norms[:-1]

            tracks.loc[track.index, 'Turning Angle'] = \
                np.arccos(dot_products[:-1]/norm_products)

            if 'Z' in track.columns:
                tracks.loc[track.index, 'Plane Angle'] = np.nan

                n_vectors = np.cross(dr, dr.shift())
                n_norms = np.linalg.norm(n_vectors, axis=1)
                dot_products = np.sum(n_vectors[1:]*n_vectors[:-1], axis=1)
                norm_products = n_norms[1:]*n_norms[:-1]
                angles = np.arccos(dot_products/norm_products)
                cross_products = np.cross(n_vectors[1:], n_vectors[:-1])
                cross_dot_dr = np.sum(cross_products[2:]*dr.as_matrix()[2:-1],
                    axis=1)
                cross_norms = np.linalg.norm(cross_products[2:], axis=1)
                signs = cross_dot_dr/cross_norms/dr_norms[2:-1]

                tracks.loc[track.index[2:-1], 'Plane Angle'] = signs*angles[2:]


def plot_tracks(tracks, summary=None, n_tracks=25, condition='Condition'):
    """Plot tracks"""
    if type(summary) == pd.core.frame.DataFrame:
        alpha = 0.33
        skip_steps = int(next(word
            for column in summary.columns
            for word in column.split() if word.isdigit()))
    else:
        alpha = 1
        _uniquize_tracks(tracks)
        _split_at_skip(tracks)

    if condition not in tracks.columns:
        tracks[condition] = 'Default'
    n_conditions = len(tracks[condition].unique())

    sns.set_style('white')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1, projection='3d')
    for i, (cond, cond_tracks) in enumerate(tracks.groupby(condition)):
        if summary is not None:
            cond_summary = summary[summary[condition] == cond]
            max_turn_column = next(column for column in summary.columns
                if column.startswith('Max. Turn'))
            choice = cond_summary.sort(max_turn_column, ascending=False)\
                ['Track_ID'][:int(n_tracks/n_conditions)]
            cond_tracks = cond_tracks[cond_tracks['Track_ID'].isin(choice)]
        elif cond_tracks['Track_ID'].unique().__len__() < n_tracks/n_conditions:
            pass
        else:
            choice = np.random.choice(cond_tracks['Track_ID'].unique(),
                n_tracks, replace=False)
            cond_tracks = cond_tracks[cond_tracks['Track_ID'].isin(choice)]

        color = sns.color_palette(n_colors=i+1)[-1]
        for _, track in cond_tracks.groupby(track_identifiers(cond_tracks)):
            ax.plot(track['X'].values, track['Y'].values, track['Z'].values,
                color=color, alpha=alpha)
            if summary is not None:
                track_id = track['Track_ID'].iloc[0]
                turn_time = cond_summary[cond_summary['Track_ID'] == track_id]['Turn Time']
                turn_loc = track.index.get_loc(
                    track[np.isclose(track['Time'], turn_time.values[0])].index.values[0])
                turn_times = track['Time'][turn_loc - 1:turn_loc + skip_steps]
                turn = track[track['Time'].isin(turn_times)]
                ax.plot(turn['X'].values, turn['Y'].values, turn['Z'].values,
                    color=color)

    equalize_axis3d(ax)
    plt.tight_layout()
    plt.show()


def plot(tracks, save=False, palette='deep', max_time=9, condition='Condition',
    plot_each_sample=False):
    """Plot aspects of motility for different conditions"""
    _analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = 'Default'

    sns.set(style="ticks", palette=sns.color_palette(
        palette, len(tracks[condition].unique())))
    if 'Plane Angle' in tracks.columns:
        figure, axes = plt.subplots(ncols=4, figsize=(16,5.5))
    else:
        figure, axes = plt.subplots(ncols=3, figsize=(12,5.5))
    plt.setp(axes, yticks=[])
    plt.setp(axes, xticks=[])

    axes[0].set_ylabel('Median Displacement')
    axes[0].set_xlabel(r'Sqrt. of Time [min$^{1/2}$]')
    axes[0].set_xlim([0, np.sqrt(max_time)])
    axes[0].set_xticks([0, np.sqrt(max_time)])
    axes[0].set_xticklabels(['0', str(int(np.sqrt(max_time)))])

    axes[1].set_xlabel(r'Velocity  [$\mu$m/min]')
    axes[1].set_ylabel('Density')
    axes[1].set_xlim(0, np.percentile(tracks['Velocity'].dropna(), 99.5))
    axes[1].set_xticks([0, 10])
    axes[1].set_xticklabels(['0', '10'])

    axes[2].set_xlabel('Turning Angle')
    axes[2].set_ylabel('Density')
    axes[2].set_xlim([0,np.pi])
    axes[2].set_xticks([0, np.pi/2, np.pi])
    axes[2].set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

    if 'Plane Angle' in tracks.columns:
        axes[3].set_xlabel('Plane Angle')
        axes[3].set_ylabel('Density')
        axes[3].set_xlim([-np.pi, np.pi])
        axes[3].set_xticks([-np.pi, 0, np.pi])
        axes[3].set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    colors = {cond: sns.color_palette(n_colors=i+1)[-1]
        for i, cond in enumerate(tracks[condition].dropna().unique())}
    if plot_each_sample:
        groups = [condition, 'Sample']
    else:
        groups = condition

    for _, cond_tracks in tracks.groupby(groups):
        # Plot displacements, inspired by http://stackoverflow.com/questions/
        # 22795348/plotting-time-series-data-with-seaborn
        label = cond_tracks[condition].iloc[0]
        color = colors[label]
        displacements = cond_tracks[['Track Time', 'Displacement']].groupby(
            'Track Time').describe().unstack()['Displacement']
        if max_time:
            displacements = displacements[displacements.index <= max_time]
        axes[0].plot(np.sqrt(displacements.index), displacements['50%'],
            color=color, label=label)
        if not plot_each_sample:
            axes[0].fill_between(np.sqrt(displacements.index),
                displacements['25%'], displacements['75%'],
                alpha=.2, color=color)
        if not plot_each_sample and len(tracks[condition].unique()) == 1:
            axes[0].fill_between(np.sqrt(displacements.index),
                displacements['min'], displacements['max'],
                alpha=.2, color=color)

        # Plot velocities
        sns.kdeplot(cond_tracks['Velocity'].dropna(), clip=(0,np.inf), color=color,
            shade=not plot_each_sample, ax=axes[1], gridsize=500, label='')

        # Plot turning angles
        turning_angles = cond_tracks['Turning Angle'].dropna().as_matrix()
        if 'Z' in tracks.columns:
            x = np.arange(0, np.pi, 0.1)
            axes[2].plot(x, np.sin(x)/2, '--k')
            sns.kdeplot(turning_angles, clip=(0,np.inf),
                color=color, shade=not plot_each_sample, ax=axes[2])
        else:
            turning_angles = np.concatenate(( # Mirror at boundaries.
                -turning_angles, turning_angles, 2*np.pi-turning_angles))
            axes[2].plot([0, np.pi], [1/(3*np.pi), 1/(3*np.pi)], '--k')
            sns.kdeplot(turning_angles,
                color=color, shade=not plot_each_sample, ax=axes[2])

        # Plot Plane Angles
        if 'Plane Angle' in tracks.columns:
            plane_angles = cond_tracks['Plane Angle'].dropna().as_matrix()
            plane_angles = np.concatenate(( # Mirror at boundaries.
                -2*np.pi+plane_angles, plane_angles, 2*np.pi+plane_angles))
            axes[3].plot([-np.pi, np.pi], [1/(6*np.pi), 1/(6*np.pi)], '--k')
            # sns.distplot(plane_angles, ax=axes[3])
            sns.kdeplot(plane_angles, color=color, shade=not plot_each_sample,
                ax=axes[3])

    handles, labels = axes[0].get_legend_handles_labels()
    unique_entries = OrderedDict(zip(labels, handles))
    axes[0].legend(unique_entries.values(), unique_entries.keys(),
        loc='upper left')

    sns.despine()
    plt.tight_layout()
    if save:
        conditions = [cond.replace('= ', '')
            for cond in tracks[condition].unique()]
        plt.savefig('Motility_' + '-'.join(conditions) +
            '_all-samples'*plot_each_sample + '.png')
    else:
        plt.show()


def plot_dr(tracks, save=False, condition='Condition'):
    """Plot the differences in X, Y (and Z) to show biases"""
    _uniquize_tracks(tracks)
    _split_at_skip(tracks)

    dimensions = [dim for dim in ['X', 'Y', 'Z'] if dim in tracks.columns]

    differences = pd.DataFrame()

    for _, track in tracks.groupby(track_identifiers(tracks)):
        differences = differences.append(track[dimensions].diff().dropna())
        if 'Track_ID' in differences.columns:
            differences = differences.fillna(track['Track_ID'].iloc[0])
        else:
            differences['Track_ID'] = track['Track_ID'].iloc[0]

    sns.set(style="ticks", palette='deep')
    fig, axes = plt.subplots(ncols=3, figsize=(15.5,5.5))
    plt.setp(axes, yticks=[])
    plt.setp(axes, xticks=[])

    axes[0].set_title(r'$\Delta \vec r$')
    axes[0].set_xticks([0])
    axes[0].set_xticklabels([r'$0$'])

    for dimension in dimensions:
        sns.kdeplot(differences[dimension], shade=True, ax=axes[0])

    axes[1].set_title('Joint Distribution')
    axes[1].set_xlabel(r'$\Delta x$')
    axes[1].set_ylabel(r'$\Delta y$')
    axes[1].axis('equal')
    axes[1].set_xlim([differences['X'].quantile(0.1), differences['X'].quantile(0.9)])
    axes[1].set_ylim([differences['Y'].quantile(0.1), differences['Y'].quantile(0.9)])
    sns.kdeplot(differences[['X', 'Y']], shade=False, cmap='Greys', ax=axes[1])

    axes[2].set_title(r'$\Delta \vec r$ Lag Plot')
    axes[2].axis('equal')
    axes[2].set_xlabel(r'$\Delta r_i(t)$')
    axes[2].set_ylabel(r'$\Delta r_i(t+1)$')
    for i, dim in enumerate(dimensions):
        color = sns.color_palette()[i]
        for _, track in differences.groupby('Track_ID'):
            axes[2].scatter(track[dim], track[dim].shift(), facecolors=color)

    sns.despine()
    plt.tight_layout()
    if save:
        conditions = [cond.replace('= ', '')
            for cond in tracks[condition].unique()]
        plt.savefig('dr_' + '-'.join(conditions) + '.png')
    else:
        plt.show()


def joint_plot(tracks, condition='Condition', save=False, palette='deep',
    skip_color=0):
    """Plot the joint distribution of the velocities and turning angles."""
    _analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = 'Default'

    sns.set(style="white", palette=sns.color_palette(
        palette, tracks[condition].unique().__len__() + skip_color))

    y_upper_lim = np.percentile(tracks['Velocity'].dropna(), 99.5)

    for i, (cond, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette()[i + skip_color]
        sns.jointplot(cond_tracks['Turning Angle'], cond_tracks['Velocity'], kind='kde',
            stat_func=None, xlim=[0, np.pi], space=0, color=color,
            ylim=[0, y_upper_lim], joint_kws={'shade': False})
        if save:
            plt.savefig('Joint-Motility_' + cond.replace('= ', '')  + '.png')
        else:
            plt.show()


def plot_tracks_parameter_space(tracks, n_tracks=None, condition='Condition',
    save=False, palette='deep', skip_color=0):
    """Plot tracks in velocities-turning-angles-space"""
    _analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = 'Default'

    sns.set(style="ticks", palette=sns.color_palette(
        palette, tracks[condition].unique().__len__() + skip_color))
    fig, ax = plt.subplots(figsize=(5.5,5.5))
    ax.set_xlabel('Turning Angle')
    ax.set_xlim([0,np.pi])
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    ax.set_ylabel('Velocity')
    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette()[i + skip_color]
        if n_tracks != None:
            cond_tracks = cond_tracks[cond_tracks['Track_ID'].isin(
                np.random.choice(cond_tracks['Track_ID'], n_tracks))]
        for _, track in cond_tracks.groupby('Track_ID'):
            ax.plot(track['Turning Angle'], track['Velocity'],
                color=color, alpha=0.5)

    sns.despine()
    plt.tight_layout()
    if save:
        conditions = [cond.replace('= ', '')
            for cond in tracks[condition].unique()]
        plt.savefig('Motility-TracksInParameterSpace_' + '-'.join(conditions)
            + '.png')
    else:
        plt.show()


def lag_plot(tracks, condition='Condition', save=False, palette='deep',
    skip_color=0, null_model=True):
    """Lag plot for velocities and turning angles"""
    _analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = 'Default'

    sns.set(style="white", palette=sns.color_palette(
        palette, tracks[condition].unique().__len__() + skip_color))
    if 'Plane Angle' in tracks.columns:
        fig, ax = plt.subplots(1,3, figsize=(12,4.25))
    else:
        fig, ax = plt.subplots(1,2, figsize=(8,4.25))
    plt.setp(ax, yticks=[])
    plt.setp(ax, xticks=[])
    ax[0].set_title('Velocity')
    ax[0].set_xlabel('v(t)')
    ax[0].set_ylabel('v(t+1)')
    ax[0].axis('equal')
    ax[1].set_title('Turning Angle')
    ax[1].set_xlabel(r'$\theta$(t)')
    ax[1].set_ylabel(r'$\theta$(t+1)')
    ax[1].axis('equal')
    if 'Plane Angle' in tracks.columns:
        ax[2].set_title('Plane Angle')
        ax[2].set_xlabel(r'$\phi$(t)')
        ax[2].set_ylabel(r'$\phi$(t+1)')
        ax[2].axis('equal')

    if null_model:
        null_model = tracks.ix[np.random.choice(tracks.index, tracks.shape[0])]
        ax[0].scatter(null_model['Velocity'], null_model['Velocity'].shift(),
            facecolors='0.8')
        ax[1].scatter(null_model['Turning Angle'], null_model['Turning Angle'].shift(),
            facecolors='0.8')
        if 'Plane Angle' in tracks.columns:
            ax[2].scatter(null_model['Plane Angle'], null_model['Plane Angle'].shift(),
                facecolors='0.8')

    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette()[i + skip_color]
        for _, track in cond_tracks.groupby('Track_ID'):
            ax[0].scatter(track['Velocity'], track['Velocity'].shift(),
                facecolors=color)
            ax[1].scatter(track['Turning Angle'], track['Turning Angle'].shift(),
                facecolors=color)
            if 'Plane Angle' in tracks.columns:
                ax[2].scatter(track['Plane Angle'], track['Plane Angle'].shift(),
                    facecolors=color)

    sns.despine()
    plt.tight_layout()
    if save:
        conditions = [cond.replace('= ', '')
            for cond in tracks[condition].unique()]
        plt.savefig('Motility-LagPlot_' + '-'.join(conditions) + '.png')
    else:
        plt.show()


def summarize(tracks, skip_steps=4):
    """Summarize track statistics, e.g. mean velocity per track"""
    _analyze(tracks)

    print('\nSummarizing track statistics')

    summary = pd.DataFrame()

    for i, (_, track) in enumerate(tracks.groupby(track_identifiers(tracks))):
        if 'Track_ID' in track.columns:
            summary.loc[i, 'Track_ID'] = track.iloc[0]['Track_ID']
        if 'Condition' in track.columns:
            summary.loc[i, 'Condition'] = track.iloc[0]['Condition']
        else:
            summary.loc[i, 'Condition'] = 'Default'
        if 'Sample' in track.columns:
            summary.loc[i, 'Sample'] = track.iloc[0]['Sample']

        summary.loc[i, 'Mean Velocity'] = track['Velocity'].mean()
        summary.loc[i, 'Mean Turning Angle'] = track['Turning Angle'].mean()
        if 'Plane Angle' in track.columns:
            summary.loc[i, 'Mean Plane Angle'] = track['Plane Angle'].mean()

        summary.loc[i, 'Track Duration'] = \
            track['Time'].iloc[-1] - track['Time'].iloc[0]

        # ratio of v < 2 um/min
        summary.loc[i, 'Arrest Coefficient'] = \
            len(track[track['Velocity'] < 2])/len(track['Velocity'].dropna())

        if 'Z' in track.columns:
            positions = track[['X', 'Y', 'Z']]
            ndim = 3
        else:
            positions = track[['X', 'Y']]
            ndim = 2

        summary.loc[i, 'Motility Coefficient'] = np.pi* \
            track['Displacement'].iloc[-1]/(2*ndim)/track['Track Time'].max()

        dr = positions.diff()
        dr_norms = np.linalg.norm(dr, axis=1)

        summary.loc[i, 'Corr. Confinement Ratio'] = track['Displacement'].iloc[-1] \
            /dr_norms[1:].sum()*np.sqrt(track['Track Time'].max())

        summary.loc[i, 'Mean Sq. Velocity Lag'] = np.mean(
            track['Velocity'].diff()**2)

        summary.loc[i, 'Mean Sq. Turn. Angle Lag'] = np.mean(
            track['Turning Angle'].diff()**2)

        if len(track) > skip_steps + 1:
            dot_products = np.sum(dr.shift(-skip_steps)*dr, axis=1)
            norm_products = dr_norms[skip_steps:]*dr_norms[:-skip_steps]
            turns = np.arccos(dot_products.iloc[1:-skip_steps]/norm_products[1:])

            summary.loc[i, 'Max. Turn Over {} Steps'.format(skip_steps + 1)] = \
                max(turns)

            summary.loc[i, 'Turn Time'] = track.loc[turns.idxmax(), 'Time']

            cross_product = np.cross(dr.shift(-skip_steps).loc[turns.idxmax()],
                dr.loc[turns.idxmax()])
            normal_vec = cross_product/np.linalg.norm(cross_product)

            summary.loc[i, 'Skew Lines Distance'] = abs(np.sum(
                (positions.shift(-skip_steps).loc[turns.idxmax()] - \
                positions.loc[turns.idxmax()])*normal_vec))

    for cond, cond_summary in summary.groupby('Condition'):
        print('  {} tracks in {} with {} timesteps in total.'.format(
            cond_summary.__len__(), cond,
            tracks[tracks['Condition'] == cond].__len__()))

    return summary


def plot_summary(summary, save=False, condition='Condition'):
    """Plot distributions and joint distributions of the track summary"""
    to_drop = [column
        for column in summary.columns
        if (column not in [condition, 'Sample'] and summary[column].var() == 0
            or 'Turn ' in column)]
    to_drop.extend([column for column
        in ['Track_ID', 'Skew Lines Distance',
            'Mean Sq. Turn. Angle Lag', 'Mean Sq. Velocity Lag']
        if column in summary.columns])

    sns.set(style='white')
    sns.pairplot(summary.drop(to_drop, axis=1), hue=condition,
        diag_kind='kde')
    plt.tight_layout()

    if save:
        conditions = [cond.replace('= ', '')
            for cond in summary[condition].unique()]
        plt.savefig('Summary_' + '-'.join(conditions) + '.png')
    else:
        plt.show()


def plot_uturns(summary, critical_rad=2.9, save=False, condition='Condition'):
    """Plot and print steepest turns over more than critical_rad"""
    turn_column = next(col for col in summary.columns
        if col.startswith('Max. Turn Over'))
    columns_of_interest = ['Skew Lines Distance', 'Mean Velocity',
        'Arrest Coefficient', condition, turn_column]

    uturns = summary[summary[turn_column] > critical_rad]

    skip_steps = int(next(word
        for word in turn_column.split() if word.isdigit()))
    print('\nPlotting turns with more than {} rad over {} steps'.format(
        critical_rad, skip_steps))
    for cond, cond_uturns in uturns.groupby(condition):
        n_tracks = len(summary[summary[condition] == cond])
        n_turns = len(cond_uturns)
        print('  {} tracks in {} with {} U-Turns ({:2.2f} %).'.format(
            n_tracks, cond, n_turns, n_turns/n_tracks*100))

    sns.set(style='white')
    sns.pairplot(uturns[columns_of_interest],
        hue=condition, diag_kind='kde')
    plt.tight_layout()

    if save:
        conditions = [cond.replace('= ', '')
            for cond in summary[condition].unique()]
        plt.savefig('U-Turns_' + '-'.join(conditions) +
            '_{:1.1f}over{}steps.png'.format(critical_rad, skip_steps))
    else:
        plt.show()


def all_out(tracks, condition='Condition', return_summary=False):
    """Save all plots and the tracks & summary DataFrame. Return summary."""
    plot(tracks, save=True)
    if 'Sample' in tracks.columns:
        plot(tracks, plot_each_sample=True, save=True)
    plot_dr(tracks, save=True)
    joint_plot(tracks, save=True)
    lag_plot(tracks, save=True, null_model=False)
    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        lag_plot(cond_tracks, save=True, skip_color=i)

    summary = summarize(tracks)
    plot_summary(summary, save=True)
    plot_uturns(summary, save=True)

    conditions = [cond.replace('= ', '')
        for cond in summary[condition].unique()]
    tracks.to_csv('Tracks_' + '-'.join(conditions) + '.csv')
    summary.to_csv('Summary_' + '-'.join(conditions) + '.csv')

    if return_summary:
        return summary


if __name__ == "__main__":
    """Demostrate motility analysis of simulated data."""
    import remix


    """Uniquize & split single track"""
    # to_uniquize = pd.DataFrame({
    #     'Track_ID': 0, 'Time': (0,1,1,0,2), 'X': 0, 'Y': 0, 'Z': 0})
    # to_uniquize = to_uniquize.append(pd.DataFrame({
    #     'Track_ID': 1, 'Time': (0,1,1,0,2), 'X': (0,1,0,1,0), 'Y': 0, 'Z': 0}))
    # track_2 = pd.DataFrame({
    #     'Track_ID': 2, 'Time': (0,1,1,1,2), 'X': 0, 'Y': 0, 'Z': 0})
    # to_uniquize = to_uniquize.append(track_2)
    # _uniquize_tracks(to_uniquize)
    # print(to_uniquize, '\n', track_2, '\n')

    # to_split = pd.DataFrame({'Track_ID': 0, 'Time': np.arange(10)/3}).drop(4)
    # _split_at_skip(to_split)
    # print(to_split)


    """Find steepest turn in single track"""
    # track = pd.DataFrame({
    #     'Velocity':np.ones(7) + np.sort(np.random.rand(7)/100),
    #     'Turning Angle': np.sort(np.random.rand(7))/100,
    #     'Plane Angle': np.random.rand(7)/100})
    # track.loc[2, 'Turning Angle'] = np.pi/2
    # track.loc[3, 'Turning Angle'] = np.pi/2

    # tracks = remix.silly_steps(track)
    # tracks['Track_ID'] = 0
    # tracks['Time'] = np.arange(8)
    # summary = summarize(tracks, skip_steps=2)
    # plot_tracks(tracks, summary)

    """Analyze several tracks"""
    tracks = remix.silly_tracks()
    tracks.loc[:, 'Time'] = tracks['Time']/3
    # plot_dr(tracks)

    # plot(tracks)
    # joint_plot(tracks, skip_color=1)
    # plot_tracks_parameter_space(tracks)
    # lag_plot(tracks, skip_color=1)

    summary = summarize(tracks)
    # plot_summary(summary)
    # plot_uturns(summary)
    plot_tracks(tracks, summary)
