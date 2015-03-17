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


def equalize_axis3d(source_ax, zoom=1, target_ax=None):
    """Equalizes axis for a mpl3d plot; after
    http://stackoverflow.com/questions/8130823/set-matplotlib-3d-plot-aspect-ratio"""
    if target_ax == None:
        target_ax = source_ax
    elif zoom != 1:
        print('Zoom ignored when target axis is provided.')
        zoom = 1
    source_extents = np.array([getattr(source_ax, 'get_{}lim'.format(dim))()
        for dim in 'xyz'])
    target_extents = np.array([getattr(target_ax, 'get_{}lim'.format(dim))()
        for dim in 'xyz'])
    spread = target_extents[:,1] - target_extents[:,0]
    max_spread = max(abs(spread))
    r = max_spread/2
    centers = np.mean(source_extents, axis=1)
    for center, dim in zip(centers, 'xyz'):
        getattr(source_ax, 'set_{}lim'.format(dim))(center-r/zoom, center+r/zoom)
    source_ax.set_aspect('equal')


def plot_tracks(tracks, summary=None, condition='Condition'):
    """Plots tracks"""
    if type(summary) == pd.core.frame.DataFrame:
        alpha = 0.33
        skip_steps = int(next(word
            for column in summary.columns
            for word in column.split() if word.isdigit()))
    else:
        alpha = 1

    if condition not in tracks.columns:
        tracks[condition] = 'Default'

    criteria = [crit for crit in ['Condition', 'Sample', 'Track_ID', 'Source']
        if crit in tracks.dropna(axis=1).columns
        if crit != condition]

    sns.set_style('white')
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1,1,1, projection='3d')
    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette(n_colors=i+1)[-1]
        for _, track in cond_tracks.groupby(criteria):
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
    """Shows an animation of the tracks"""
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


def analyze(tracks, uniform_timesteps=True, min_length=4):
    """Return DataFrame with velocity, turning angle & rolling angle"""


    def uniquize_tracks(tracks, criteria):
        """Tries to guess which tracks belong together, if not unique"""
        if 'Track_ID' in tracks.columns:
            max_track_id = tracks['Track_ID'].max()
        else:
            max_track_id = 0

        for crit, track in tracks.groupby(criteria):
            duplicated_steps = track[track['Time'].duplicated()]
            if duplicated_steps.__len__() != 0:
                n_clusters = track['Time'].value_counts().max()
                # TODO: Use conectivity
                clusters = AgglomerativeClustering(n_clusters=n_clusters).fit(
                    track[['X', 'Y', 'Z']])
                index = track.index
                if 'Track_ID' in track.columns:
                    tracks.loc[index, 'Original Track_ID'] = track['Track_ID']
                tracks.loc[index, 'Track_ID'] = max_track_id+1+clusters.labels_
                max_track_id += n_clusters + 1
                print('Warning: Split non-unique track {} based on index.'
                    .format(crit))

        if sum(tracks[criteria + ['Time']].duplicated()) != 0:
            raise Exception


    def split_at_skip(tracks, criteria):
        """Splits track if timestep is missing in the original DataFrame"""
        if 'Track_ID' in tracks.columns:
            max_track_id = tracks['Track_ID'].max()
        else:
            max_track_id = 0

        for _, track in tracks.groupby(criteria):
            timesteps = track['Time'].diff()
            skips = np.round((timesteps - timesteps.min())/timesteps.min())
            if skips.max() > 0:
                index = track.index
                if 'Track_ID' in track.columns:
                    tracks.loc[index, 'Original Track_ID'] = track['Track_ID']
                skip_sum = skips.fillna(0).cumsum()
                tracks.loc[index, 'Track_ID'] = max_track_id + 1 + skip_sum
                max_track_id += max(skip_sum) + 1


    def analyze_track(track):
        """Calculates velocity and angles for a single track"""
        track['Track Time'] = track['Time'] - track['Time'].iloc[0]
        if track['Track Time'].diff().unique().__len__() > 2:
            print('Warning: Track with non-uniform timesteps.')

        if 'Z' in track.columns:
            positions = track[['X', 'Y', 'Z']]
        else:
            positions = track[['X', 'Y']]

        track['Displacement'] = np.linalg.norm(positions - positions.iloc[0], axis=1)

        dr = positions.diff()
        dr_norms = np.linalg.norm(dr, axis=1)

        track['Velocity'] = dr_norms/track['Time'].diff()

        dot_products = np.sum(dr.shift(-1)*dr, axis=1)
        norm_products = dr_norms[1:]*dr_norms[:-1]

        track['Turning Angle'] = np.arccos(dot_products[:-1]/norm_products)

        if 'Z' in track.columns:
            track['Rolling Angle'] = np.nan

            n_vectors = np.cross(dr, dr.shift())
            n_norms = np.linalg.norm(n_vectors, axis=1)
            dot_products = np.sum(n_vectors[1:]*n_vectors[:-1], axis=1)
            norm_products = n_norms[1:]*n_norms[:-1]
            angles = np.arccos(dot_products/norm_products)
            cross_products = np.cross(n_vectors[1:], n_vectors[:-1])
            cross_dot_dr = np.sum(cross_products[2:]*dr.as_matrix()[2:-1], axis=1)
            cross_norms = np.linalg.norm(cross_products[2:], axis=1)
            signs = cross_dot_dr/cross_norms/dr_norms[2:-1]

            track['Rolling Angle'].iloc[2:-1] = signs*angles[2:]

        return track


    criteria = [crit
        for crit in ['Track_ID', 'Sample', 'Condition']
        if crit in tracks.columns]

    try:
        uniquize_tracks(tracks, criteria)
    except Exception:
        print('Error: Tracks not unique, aborting analysis.')
        return

    tracks[criteria] = tracks[criteria].fillna('Default')

    tracks = tracks.reset_index(drop=True) # split_at_skip() works on index!

    if 'Time' not in tracks.columns:
        print('Warning: no time given, using index!')
        tracks['Time'] = tracks.index
    # Take care of missing time points
    elif uniform_timesteps:
        split_at_skip(tracks, criteria)

    tracks = tracks.groupby(criteria).apply(
        lambda x: x if x.__len__() > min_length else None)
    tracks = tracks.reset_index(drop=True) # Clean up nested index from removal

    return tracks.groupby(criteria).apply(analyze_track)


def plot(tracks, save=False, palette='deep', plot_minmax=False,
    condition='Condition'):
    """Plots aspects of motility for different conditions"""
    if not set(['Velocity', 'Turning Angle']).issubset(tracks.columns):
        print('Error: data not found, tracks must be analyzed first.')
        return

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
    axes[1].set_xticklabels([r'$0$'])

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

        # Plot velocities
        sns.kdeplot(cond_tracks['Velocity'].dropna(),
            shade=True, ax=axes[1], gridsize=500, label=cond)

        # Plot turning angles
        turning_angles = cond_tracks['Turning Angle'].dropna().as_matrix()
        if 'Z' in tracks.columns:
            x = np.arange(0, np.pi, 0.1)
            axes[2].plot(x, np.sin(x)/2, '--k')
        else:
            turning_angles = np.concatenate(( # Mirror at boundaries.
                -turning_angles, turning_angles, 2*np.pi-turning_angles))
            axes[2].plot([0, np.pi], [1/(3*np.pi), 1/(3*np.pi)], '--k')
        sns.kdeplot(turning_angles, shade=True, ax=axes[2])

        # Plot Rolling Angles
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


def plot_dr(tracks):
    """Plots the differences in X, Y (and Z) to show biases"""
    dimensions = [dim for dim in ['X', 'Y', 'Z'] if dim in tracks.columns]

    differences = pd.DataFrame()
    criteria  = [crit
        for crit in ['Track_ID', 'Condition', 'Sample']
        if crit in tracks.columns]
    for _, track in tracks.groupby(criteria):
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
    plt.show()


def joint_plot(tracks, condition='Condition', save=False,
    palette='deep', skip_color=0):
    """Plots the joint distribution of the velocities and turning angles."""
    if not set(['Velocity', 'Turning Angle']).issubset(tracks.columns):
        print('Error: data not found, tracks must be analyzed first.')
        return

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
    """Plots tracks in velocities-turning-angles-space"""
    if not set(['Velocity', 'Turning Angle']).issubset(tracks.columns):
        print('Error: data not found, tracks must be analyzed first.')
        return

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
    """Lag plots for velocities and turning angles"""
    if not set(['Velocity', 'Turning Angle']).issubset(tracks.columns):
        print('Error: data not found, tracks must be analyzed first.')
        return

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
    """Summarize track statistics"""
    if not set(['Velocity', 'Turning Angle']).issubset(tracks.columns):
        print('Error: data not found, tracks must be analyzed first.')
        return

    summary = pd.DataFrame()

    criteria = [crit
        for crit in ['Condition', 'Track_ID', 'Sample']
        if crit in tracks.columns]

    for i, (_, track) in enumerate(tracks.groupby(criteria)):
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

        summary.loc[i, 'Arrest Coefficient'] = \
            track[track['Velocity'] < 2].__len__()/summary.loc[i, 'Track Duration']

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
        print('{} tracks in {} with {} timesteps in total.'.format(
            cond_summary.__len__(), cond, cond_summary['Track Duration'].sum()))

    return summary


def plot_summary(summary):
    """Plot distributions and joint distributions of the track summary"""
    to_drop = [column
        for column in summary.columns
        if column != 'Condition' and summary[column].var() == 0]
    to_drop.extend(['Track_ID', 'Turn Time'])
    sns.set(style='white')
    sns.pairplot(summary.drop(to_drop, axis=1), hue='Condition',
        diag_kind='kde')

    plt.show()


def analyze_turns(tracks, skip_steps=4):
    """Analyze turns between t and t+skip_steps in cell tracks"""
    if not set(['Velocity', 'Turning Angle']).issubset(tracks.columns):
        print('Error: data not found, tracks must be analyzed first.')
        return

    turns = pd.DataFrame()

    criteria = [crit
        for crit in ['Condition', 'Track_ID', 'Sample']
        if crit in tracks.columns]

    for i, (crits, track) in enumerate(tracks.groupby(criteria)):
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
    """Demostrates motility analysis of simulated data."""
    import remix


    # # Single Track
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
    # tracks = analyze(tracks)
    # summary = summarize(tracks, skip_steps=1)
    # plot_tracks(tracks, summary)


    # Several tracks
    tracks = remix.silly_tracks()
    # animate_tracks(tracks)
    # plot_dr(tracks)

    tracks = analyze(tracks)
    # plot(tracks)
    # joint_plot(tracks, skip_color=1)
    # lag_plot(tracks, skip_color=1)

    summary = summarize(tracks)
    print(summary)
    plot_summary(summary)
    plot_tracks(tracks, summary)
