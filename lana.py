"""Tools to analyze and plot cell motility from tracks within lymph nodes"""
import random

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D


def silly_steps(init_position=None, steps=25, step_size=1):
    """Generates a 2D random walk after Nombela-Arrieta et al. 2007"""
    if init_position == None:
        init_position = 10*np.random.rand(1,2)
    track = init_position

    for _ in range(steps):
        if track.shape[0] == 1:
            angle = 2*np.pi*np.random.rand()
        else:
            last_step = track[-1] - track[-2]
            x = last_step[0]/np.linalg.norm(last_step)
            y = last_step[1]/np.linalg.norm(last_step)
            angle = (np.sign(x) - 1)/2*np.pi + np.sign(x)*np.arcsin(y)
        silly_angle = angle + np.random.lognormal(0, 0.5) \
            * (2*np.random.randint(0, 2) - 1)
        silly_displacement = step_size*np.random.lognormal(0, 0.5)
        silly_step = silly_displacement*np.asarray(
            [np.cos(silly_angle),
            np.sin(silly_angle)]).reshape(1,2)
        track = np.concatenate((track, track[-1] + silly_step))
    return track


def silly_tracks(ntracks=25):
    """Generates a DataFrame with tracks"""
    tracks = pd.DataFrame()
    for track_id in range(ntracks):
        track = silly_steps()
        tracks = tracks.append(pd.DataFrame({
            'Track_ID': track_id, 'X': track[:,0], 'Y': track[:,1]}))
    return tracks


def plot_tracks(tracks, save=False, palette='deep'):
    """Plots the x-y-tracks of a (list of) Motility class(es)"""
    if 'Z' in tracks.columns:
        ndim = 3
    else:
        ndim = 2

    if 'Condition' not in tracks.columns:
        tracks['Condition'] = 'Default'

    sns.set(style="white", palette=sns.color_palette(
        palette, tracks['Condition'].unique().__len__()))
    sns.set_context("paper", font_scale=1.5)

    plt.title('Superimposed Tracks')
    for i, dim in enumerate(('Y', 'Z')[:ndim-1]):
        plt.subplot(1, ndim-1, i+1)
        plt.axis('equal')
        plt.ylabel('x-axis')
        plt.xlabel(['y-axis', 'z-axis'][i])
        for j, (cond, cond_tracks) in enumerate(tracks.groupby('Condition')):
            color = sns.color_palette()[j]
            for (track_nr, track) in cond_tracks.groupby('Track_ID'):
                if track_nr == 0:
                    label = cond
                else:
                    label = ''
                plt.plot(track['X']-track['X'][0], track[dim]-track[dim][0],
                    color=color, label=label)
                # track = track - track[1]
                # track = track.reshape(-1, 1, 2)
                # segments = np.concatenate([track[:-1], track[1:]], axis=1)
                # lc = LineCollection(segments, cmap=plt.get_cmap('copper'))
                # plt.gca().add_collection(lc)

    plt.legend()
    plt.tight_layout()
    if save:
        conditions = [cond.replace('= ', '')
            for cond in tracks['Condition'].unique()]
        plt.savefig('Motility_' + '-'.join(conditions) + '.png')
    else:
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


def analyze_track(track):
    """Calculates displacements, velocities and turning angles for a track"""
    track['Track Time'] = track['Time'] - track['Time'].iloc[0]
    if track['Track Time'].diff().unique().__len__() > 2:
        print('Warning: Track with different timesteps.')
        print(track['Source'].iloc[0], track['Sample'].iloc[0], track['Track_ID'].iloc[0])

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

        track['Rolling Angle'].iloc[:-1] = np.arccos(dot_products/norm_products)

    return track


def split_at_skip(track):
    """Splits track if timestep is missing"""
    timesteps = track['Time'].diff()
    n_timesteps = timesteps.unique().__len__() - 1 # Remove NaN for 1st row
    if n_timesteps > 1:
        skips = (timesteps - timesteps.min())/timesteps.min()
        skipsum = skips.fillna(0).cumsum()/(skips.sum() + 1)
        track['Track_ID'] = track['Track_ID'] + skipsum

    return track


def analyze_motility(tracks, sample='Sample', uniform_timesteps=True,
    min_length=4):
    """Prepares tracks for analysis"""
    if sample not in tracks.columns:
        sample = 'Condition'
        if 'Condition' not in tracks.columns:
            tracks['Condition'] = 'Default'

    if 'Time' not in tracks.columns:
        print('Warning: no time given, using index!')
        tracks['Time'] = tracks.index
    # Take care of missing time points
    elif uniform_timesteps:
        tracks = tracks.groupby([sample, 'Track_ID']).apply(split_at_skip)

    if sum(tracks[[sample, 'Track_ID', 'Time']].duplicated()) != 0:
        print('Error: Tracks not unique, aborting analysis.')
        return

    tracks = tracks.groupby([sample, 'Track_ID']).apply(
        lambda x: x if x.__len__() > min_length else None)

    return tracks.groupby([sample, 'Track_ID']).apply(analyze_track)


def plot_motility(tracks, save=False, palette='deep', plot_minmax=False,
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
        axes[3].set_xlim([0,np.pi])
        axes[3].set_xticks([0, np.pi/2, np.pi])
        axes[3].set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

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
                -rolling_angles, rolling_angles, 2*np.pi-rolling_angles))
            axes[3].plot([0, np.pi], [1/(3*np.pi), 1/(3*np.pi)], '--k')
            # sns.distplot(rolling_angles, ax=axes[3])
            sns.kdeplot(rolling_angles, shade=True, ax=axes[3])

    if save:
        conditions = [cond.replace('= ', '')
            for cond in tracks[condition].unique()]
        plt.savefig('Motility_' + '-'.join(conditions) + '.png')
    else:
        plt.show()


def plot_joint_motility(tracks, save=False, palette='deep'):
    """Plots the joint distribution of the velocities and turning angles."""
    if not set(['Velocity', 'Turning Angle']).issubset(tracks.columns):
        print('Error: data not found, tracks must be analyzed first.')
        return

    sns.set(style="white", palette=sns.color_palette(
        palette, tracks['Condition'].unique().__len__()))

    for i, (cond, cond_tracks) in enumerate(tracks.groupby('Condition')):
        color = sns.color_palette()[i]
        sns.jointplot(tracks['Turning Angle'], tracks['Velocity'], kind='kde',
            stat_func=None, xlim=[0, np.pi], space=0,
            ylim=[0, np.percentile(tracks['Velocity'].dropna(), 99.5)])
        if save:
            plt.savefig('Joint-Motility_' + cond.replace('= ', '')  + '.png')
        else:
            plt.show()


def lag_plot(tracks, condition='Condition', save=False, palette='deep'):
    """Lag plots for velocities and turning angles"""
    if not set(['Velocity', 'Turning Angle']).issubset(tracks.columns):
        print('Error: data not found, tracks must be analyzed first.')
        return

    if condition not in tracks.columns:
        tracks[condition] = 'Default'

    sns.set(style="white", palette=sns.color_palette(
        palette, tracks[condition].unique().__len__()))
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

    null_model = tracks.ix[random.sample(list(tracks.index), tracks.shape[0])]
    ax[0].scatter(null_model['Velocity'], null_model['Velocity'].shift(),
        facecolors='0.8')
    ax[1].scatter(null_model['Turning Angle'], null_model['Turning Angle'].shift(),
        facecolors='0.8')
    if 'Rolling Angle' in tracks.columns:
        ax[2].scatter(null_model['Rolling Angle'], null_model['Rolling Angle'].shift(),
            facecolors='0.8')

    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette()[i]
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


def summarize_tracks(tracks):
    """Summarize track statistics"""
    if not set(['Velocity', 'Turning Angle']).issubset(tracks.columns):
        print('Error: data not found, tracks must be analyzed first.')
        return

    summary = pd.DataFrame()
    summary.index.name = 'Track_ID'
    tracks['Track_ID'] = tracks['Track_ID'].astype(str)

    for track_id, track in tracks.groupby('Track_ID'):
        if 'Condition' in track.columns:
            summary.loc[track_id, 'Condition'] = track.iloc[0]['Condition']
        if 'Sample' in track.columns:
            summary.loc[track_id, 'Sample'] = track.iloc[0]['Sample']

        summary.loc[track_id, 'Mean Velocity'] = track['Velocity'].mean()
        summary.loc[track_id, 'Mean Turning Angle'] = track['Turning Angle'].mean()
        if 'Rolling Angle' in track.columns:
            summary.loc[track_id, 'Mean Rolling Angle'] = track['Rolling Angle'].mean()

        summary.loc[track_id, 'Track Duration'] = \
            track['Track Time'].iloc[-1] - track['Track Time'].iloc[0]

    return summary


def plot_summary(summary):
    """Plot distributions and joint distributions of the track summary"""
    sns.set(style='white')
    g = sns.PairGrid(summary)
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)

    plt.show()


if __name__ == "__main__":
    """Demostrates motility analysis of simulated data."""
    tracks = silly_tracks()
    # plot_tracks(tracks)
    # animate_tracks(tracks)

    tracks = analyze_motility(tracks)
    # plot_joint_motility(tracks)
    # plot_motility(tracks)
    # lag_plot(tracks)

    summary = summarize_tracks(tracks)
    plot_summary(summary)
