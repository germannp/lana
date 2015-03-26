"""Analyze and plot cell motility from tracks within lymph nodes"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering

from utils import equalize_axis3d


def _track_identifiers(tracks):
    """List criteria that identify a track"""
    return [identifier
        for identifier in ['Condition', 'Sample', 'Track_ID', 'Source']
        if identifier in tracks.dropna(axis=1).columns]


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

    for identifiers, track in tracks.groupby(_track_identifiers(tracks)):
        if sum(track['Time'].duplicated()) != 0:
            n_clusters = track['Time'].value_counts().max()
            index = track.index
            if 'Track_ID' in track.columns:
                tracks.loc[index, 'Orig. Track_ID'] = track['Track_ID']

            clusters = AgglomerativeClustering(n_clusters).fit(
                track[['X', 'Y', 'Z']])
            track['Cluster'] = clusters.labels_

            if sum(track[['Cluster', 'Time']].duplicated()) != 0:
                clusters = AgglomerativeClustering(n_clusters).fit(
                    track[['Orig. Index']])
                track['Cluster'] = clusters.labels_

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
    """Split track if timestep is missing in the original DataFrame"""
    if 'Time' not in tracks.columns:
        return

    if not tracks.index.is_unique:
        tracks.reset_index(drop=True, inplace=True)

    if 'Track_ID' in tracks.columns:
        max_track_id = tracks['Track_ID'].max()
    else:
        max_track_id = 0

    for criterium, track in tracks.groupby(_track_identifiers(tracks)):
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
    """Calculate velocity, turning angle & rolling angle"""
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

    for _, track in tracks.groupby(_track_identifiers(tracks)):
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
                tracks.loc[track.index, 'Rolling Angle'] = np.nan

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

                tracks.loc[track.index[2:-1], 'Rolling Angle'] = signs*angles[2:]


def plot_tracks(tracks, summary=None, condition='Condition'):
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

    sns.set_style('white')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1, projection='3d')
    # TODO: Plot max. number of tracks, steepest turns if summary is provided.
    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette(n_colors=i+1)[-1]
        for _, track in cond_tracks.groupby(_track_identifiers(cond_tracks)):
            ax.plot(track['X'].values, track['Y'].values, track['Z'].values,
                color=color, alpha=alpha)
            if type(summary) == pd.core.frame.DataFrame:
                track_id = track['Track_ID'].iloc[0]
                turn_time = summary[summary['Track_ID'] == track_id]['Turn Time']
                turn_times = np.arange(turn_time, turn_time + skip_steps + 1)
                turn = track[track['Time'].isin(turn_times)]
                ax.plot(turn['X'].values, turn['Y'].values, turn['Z'].values,
                    color=color)

    equalize_axis3d(ax)
    plt.tight_layout()
    plt.show()


def animate_tracks(tracks, palette='deep'):
    """Show an animation of the tracks"""
    if 'Condition' not in tracks.columns:
        tracks['Condition'] = 'Default'

    if 'Time' not in tracks.columns:
        tracks['Time'] = tracks.index

    sns.set(style="white", palette=sns.color_palette(
        palette, tracks['Condition'].unique().__len__()))
    sns.set_context("paper", font_scale=1.5)

    plt.title('Animated Tracks')
    # TODO: Scale axis properly ...
    if 'Z' in tracks.columns:
        ax = plt.gca(projection='3d')
        # equalize_axis3d(ax)
    else:
        plt.axis('equal')

    for t, posis in tracks.groupby('Time'):
        for j, (cond, cond_posis) in enumerate(posis.groupby('Condition')):
            plt.clf()
            if 'Z' in cond_posis.columns:
                ax.scatter(cond_posis['X'], cond_posis['Y'], cond_posis['Z'],
                    c='red')
            else:
                plt.plot(posis['X'], posis['Y'], 'o')
        plt.pause(1)


def plot(tracks, save=False, palette='deep', plot_minmax=False,
    condition='Condition'):
    """Plot aspects of motility for different conditions"""
    _analyze(tracks)

    if condition not in tracks.columns:
        tracks[condition] = 'Default'

    if tracks[condition].unique().__len__() == 1:
        plot_minmax = True

    sns.set(style="white", palette=sns.color_palette(
        palette, tracks[condition].unique().__len__()))
    sns.set_context("paper", font_scale=1.5)
    if 'Rolling Angle' in tracks.columns:
        figure, axes = plt.subplots(ncols=4, figsize=(16,6))
        # axes = [axes[0][0], axes[0][1], axes[1][0], axes[1][1]]
    else:
        figure, axes = plt.subplots(ncols=3, figsize=(12,6))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.setp(axes, yticks=[])
    plt.setp(axes, xticks=[])

    axes[0].set_title('Median Displacements')
    axes[0].set_xlabel('Sqrt. of Time')

    axes[1].set_title('Velocities')
    axes[1].set_xlim(0, np.percentile(tracks['Velocity'].dropna(), 99.5))
    axes[1].set_xticks([0])
    axes[1].set_xticklabels(['0'])

    axes[2].set_title('Turning Angles')
    axes[2].set_xlim([0,np.pi])
    axes[2].set_xticks([0, np.pi/2, np.pi])
    axes[2].set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

    if 'Rolling Angle' in tracks.columns:
        axes[3].set_title('Rolling Angles')
        axes[3].set_xlim([-np.pi, np.pi])
        axes[3].set_xticks([-np.pi, 0, np.pi])
        axes[3].set_xticklabels([r'$-\pi$', r'$0$', r'$\pi$'])

    for i, (cond, cond_tracks) in enumerate(tracks.groupby(condition)):
        # Plot displacements, inspired by http://stackoverflow.com/questions/
        # 22795348/plotting-time-series-data-with-seaborn
        color = sns.color_palette(n_colors=i+1)[-1]
        median = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').median()
        axes[0].plot(np.sqrt(median.index), median)
        low = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').quantile(0.25)
        high = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').quantile(0.75)
        axes[0].fill_between(np.sqrt(median.index),
            low['Displacement'], high['Displacement'],
            alpha=.2, color=color)
        if plot_minmax:
            minima = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').min()
            maxima = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').max()
            axes[0].fill_between(np.sqrt(median.index),
                minima['Displacement'], maxima['Displacement'],
                alpha=.2, color=color)

        # Plot velocities TODO: estimate variation
        sns.kdeplot(cond_tracks['Velocity'].dropna(),
            shade=True, ax=axes[1], gridsize=500, label=cond)

        # Plot turning angles TODO: estimate variation
        turning_angles = cond_tracks['Turning Angle'].dropna().as_matrix()
        if 'Z' in tracks.columns:
            x = np.arange(0, np.pi, 0.1)
            axes[2].plot(x, np.sin(x)/2, '--k')
        else:
            turning_angles = np.concatenate(( # Mirror at boundaries.
                -turning_angles, turning_angles, 2*np.pi-turning_angles))
            axes[2].plot([0, np.pi], [1/(3*np.pi), 1/(3*np.pi)], '--k')
        sns.kdeplot(turning_angles, shade=True, ax=axes[2])

        # Plot Rolling Angles TODO: estimate variation
        if 'Rolling Angle' in tracks.columns:
            rolling_angles = cond_tracks['Rolling Angle'].dropna().as_matrix()
            rolling_angles = np.concatenate(( # Mirror at boundaries.
                -2*np.pi+rolling_angles, rolling_angles, 2*np.pi+rolling_angles))
            axes[3].plot([-np.pi, np.pi], [1/(6*np.pi), 1/(6*np.pi)], '--k')
            # sns.distplot(rolling_angles, ax=axes[3])
            sns.kdeplot(rolling_angles, shade=True, ax=axes[3])

    if save:
        conditions = [cond.replace('= ', '')
            for cond in tracks[condition].unique()]
        plt.savefig('Motility_' + '-'.join(conditions) + '.png')
    else:
        plt.show()


def plot_dr(tracks, save=False, condition='Condition'):
    """Plot the differences in X, Y (and Z) to show biases"""
    _uniquize_tracks(tracks)
    _split_at_skip(tracks)

    dimensions = [dim for dim in ['X', 'Y', 'Z'] if dim in tracks.columns]

    differences = pd.DataFrame()

    for _, track in tracks.groupby(_track_identifiers(tracks)):
        differences = differences.append(track[dimensions].diff().dropna())
        if 'Track_ID' in differences.columns:
            differences = differences.fillna(track['Track_ID'].iloc[0])
        else:
            differences['Track_ID'] = track['Track_ID'].iloc[0]

    sns.set(style="white", palette='deep')
    fig, axes = plt.subplots(ncols=3, figsize=(12,4.25))
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
    sns.kdeplot(differences[['X', 'Y']], shade=True, cmap='Greys', ax=axes[1])

    axes[2].set_title(r'$\Delta \vec r$ Lag Plot')
    axes[2].axis('equal')
    axes[2].set_xlabel(r'$\Delta r_i(t)$')
    axes[2].set_ylabel(r'$\Delta r_i(t+1)$')
    for i, dim in enumerate(dimensions):
        color = sns.color_palette()[i]
        for _, track in differences.groupby('Track_ID'):
            axes[2].scatter(track[dim], track[dim].shift(), facecolors=color)

    plt.tight_layout()
    if save:
        conditions = [cond.replace('= ', '')
            for cond in tracks[condition].unique()]
        plt.savefig('dr_' + '-'.join(conditions) + '.png')
    else:
        plt.show()


def joint_plot(tracks, condition='Condition', save=False,
    palette='deep', skip_color=0):
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
            ylim=[0, y_upper_lim])
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

    sns.set(style="white", palette=sns.color_palette(
        palette, tracks[condition].unique().__len__() + skip_color))
    fig, ax = plt.subplots(figsize=(8,8))
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
    if 'Rolling Angle' in tracks.columns:
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
    if 'Rolling Angle' in tracks.columns:
        ax[2].set_title('Rolling Angle')
        ax[2].set_xlabel(r'$\phi$(t)')
        ax[2].set_ylabel(r'$\phi$(t+1)')
        ax[2].axis('equal')

    if null_model:
        null_model = tracks.ix[np.random.choice(tracks.index, tracks.shape[0])]
        ax[0].scatter(null_model['Velocity'], null_model['Velocity'].shift(),
            facecolors='0.8')
        ax[1].scatter(null_model['Turning Angle'], null_model['Turning Angle'].shift(),
            facecolors='0.8')
        if 'Rolling Angle' in tracks.columns:
            ax[2].scatter(null_model['Rolling Angle'], null_model['Rolling Angle'].shift(),
                facecolors='0.8')

    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette()[i + skip_color]
        for _, track in cond_tracks.groupby('Track_ID'):
            ax[0].scatter(track['Velocity'], track['Velocity'].shift(),
                facecolors=color)
            ax[1].scatter(track['Turning Angle'], track['Turning Angle'].shift(),
                facecolors=color)
            if 'Rolling Angle' in tracks.columns:
                ax[2].scatter(track['Rolling Angle'], track['Rolling Angle'].shift(),
                    facecolors=color)

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

    for i, (_, track) in enumerate(tracks.groupby(_track_identifiers(tracks))):
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
        if 'Rolling Angle' in track.columns:
            summary.loc[i, 'Mean Rolling Angle'] = track['Rolling Angle'].mean()

        summary.loc[i, 'Track Duration'] = \
            track['Time'].iloc[-1] - track['Time'].iloc[0]

        # ratio of v < 2 um/min
        summary.loc[i, 'Arrest Coefficient'] = \
            track[track['Velocity'] < 2].__len__()/track['Velocity'].dropna().__len__()

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

        dot_products = np.sum(dr.shift(-skip_steps)*dr, axis=1).dropna()
        norm_products = dr_norms[skip_steps:]*dr_norms[:-skip_steps]
        u_turns = np.arccos(dot_products/norm_products[1:])

        summary.loc[i, 'Max. Turn Over {} Steps'.format(skip_steps + 1)] = \
            max(u_turns)

        summary.loc[i, 'Turn Time'] = track.loc[u_turns.idxmax(), 'Time'] - 1

    for cond, cond_summary in summary.groupby('Condition'):
        print('  {} tracks in {} with {} timesteps in total.'.format(
            cond_summary.__len__(), cond,
            tracks[tracks['Condition'] == cond].__len__()))

    return summary


def plot_summary(summary, save=False, condition='Condition'):
    """Plot distributions and joint distributions of the track summary"""
    to_drop = [column
        for column in summary.columns
        if column != 'Condition' and summary[column].var() == 0]
    to_drop.extend(['Track_ID', 'Turn Time'])
    sns.set(style='white')
    sns.pairplot(summary.drop(to_drop, axis=1), hue='Condition',
        diag_kind='kde')

    if save:
        conditions = [cond.replace('= ', '')
            for cond in summary[condition].unique()]
        plt.savefig('Summary_' + '-'.join(conditions) + '.png')
    else:
        plt.show()


def analyze_turns(tracks, skip_steps=4):
    """Analyze turns between t and t+skip_steps in cell tracks"""
    _analyze(tracks)

    turns = pd.DataFrame()

    for i, (crits, track) in enumerate(tracks.groupby(_track_identifiers(tracks))):
        if 'Z' in track.columns:
            positions = track[['X', 'Y', 'Z']]
        else:
            positions = track[['X', 'Y']]

        dr = positions.diff()
        dr_norms = np.linalg.norm(dr, axis=1)

        velocities = dr_norms/track['Time'].diff()

        mean_velocities = [velocities[i:i+skip_steps].mean()
            for i in range(velocities.__len__() - 1 - skip_steps)]

        dot_products = np.sum(dr.shift(-skip_steps)*dr, axis=1).dropna()
        norm_products = dr_norms[skip_steps:]*dr_norms[:-skip_steps]

        angles = np.arccos(dot_products/norm_products[1:])

        # TODO: Calculate distance instead of displacement
        displacements = [np.linalg.norm(
            positions.values[i] - positions.values[i+skip_steps+1])
            for i in range(positions.shape[0] - 1 - skip_steps)]

        turns = turns.append(pd.DataFrame({
            'Condition': track['Condition'].iloc[0],
            'Track ID': crits[1],
            'Angle t+{}'.format(skip_steps): angles,
            'Mean Velocity Over Turn': mean_velocities,
            'Displacement Over Turn': displacements,
            'Mean Track Velocity': track['Velocity'].mean()}))

    return turns


if __name__ == "__main__":
    """Demostrate motility analysis of simulated data."""
    import remix


    """Uniquize & split single track"""
    # TODO: Test _uniquize & _split_at_skip


    """Find steepest turn in single track"""
    # track = pd.DataFrame({
    #     'Velocity':np.ones(7),
    #     'Turning Angle': np.sort(np.random.rand(7))/100,
    #     'Rolling Angle': np.random.rand(7)/100
    # })
    # track.loc[3, 'Turning Angle'] = 2.5
    #
    # tracks = remix.silly_steps(track)
    # tracks['Track_ID'] = 0
    # tracks['Time'] = np.arange(8)
    # summary = summarize(tracks, skip_steps=1)
    # plot_tracks(tracks, summary)


    """Analyze several tracks"""
    tracks = remix.silly_tracks()
    # animate_tracks(tracks)
    # plot_dr(tracks)

    # plot(tracks)
    # joint_plot(tracks, skip_color=1)
    # lag_plot(tracks, skip_color=1)

    summary = summarize(tracks)
    # plot_summary(summary)
    plot_tracks(tracks, summary)
