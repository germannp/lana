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
                ax.scatter(cond_posis['X'], cond_posis['Y'], cond_posis['Z'], c='red')
            else:
                plt.plot(posis['X'], posis['Y'], 'o')
        plt.pause(1)


def analyze_track(track):
    """Calculates displacements, velocities and turning angles for a track"""    
    if 'Z' in track.columns:
        positions = track[['X', 'Y', 'Z']]
    else:
        positions = track[['X', 'Y']]

    track['Track Time'] = track['Time'] - track['Time'].iloc[0]

    track['Displacement'] = np.linalg.norm(positions - positions.iloc[0], axis=1)

    dr = positions.diff()
    dr_norms = np.linalg.norm(dr, axis = 1)

    track['Velocity'] = dr_norms/track['Time'].diff()

    dot_products = np.sum(dr.shift(-1)*dr, axis=1)
    norm_products = dr_norms[1:]*dr_norms[:-1]

    track['Turning Angle'] = np.arccos(dot_products[:-1]/norm_products)

    return track


def analyze_tracks(tracks, condition='Condition'):
    """Prepares tracks for analysis"""
    if condition not in tracks.columns:
        tracks[condition] = 'Default'

    if 'Time' not in tracks.columns:
        print('Warning: no time given, using index!')
        tracks['Time'] = tracks.index

    if sum(tracks[[condition, 'Track_ID', 'Time']].duplicated()) != 0:
        print('Error: Tracks not unique, aborting analysis.')
        return

    return tracks.groupby([condition, 'Track_ID']).apply(analyze_track)


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
    figure, axes = plt.subplots(ncols=3, figsize=(12,6))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.setp(axes, yticks=[])
    plt.setp(axes, xticks=[])

    axes[0].set_title('Velocities')
    axes[0].set_xlim(0, np.percentile(tracks['Velocity'].dropna(), 99.5))
    axes[0].set_xticks([0])
    axes[0].set_xticklabels([r'$0$'])

    axes[1].set_title('Turning Angles')
    axes[1].set_xlim([0,np.pi])
    axes[1].set_xticks([0, np.pi/2, np.pi])
    axes[1].set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

    axes[2].set_title('Mean Displacements')
    axes[2].set_xlabel('Time')

    for i, (cond, cond_tracks) in enumerate(tracks.groupby(condition)):
        # Plot velocities
        sns.kdeplot(cond_tracks['Velocity'].dropna(), 
            shade=True, ax=axes[0], gridsize=500, label=cond)

        # Plot turning angles
        turning_angles = cond_tracks['Turning Angle'].dropna().as_matrix()
        if 'Z' in tracks.columns:
            x = np.arange(0, np.pi, 0.1)
            axes[1].plot(x, np.sin(x)/2, '--k')
        else:
            turning_angles = np.concatenate(( # Mirror at boundaries.
                -turning_angles, turning_angles, 2*np.pi-turning_angles))
            axes[1].plot([0, np.pi], [1/(3*np.pi), 1/(3*np.pi)], '--k')
        sns.kdeplot(turning_angles, shade=True, ax=axes[1])

        # Plot displacements, inspired by http://stackoverflow.com/questions/
        # 22795348/plotting-time-series-data-with-seaborn
        color = sns.color_palette(n_colors=i+1)[-1]
        median = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').median()
        axes[2].plot(np.sqrt(median.index), median)
        low = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').quantile(0.25)
        high = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').quantile(0.75)
        axes[2].fill_between(np.sqrt(median.index), 
            low['Displacement'], high['Displacement'], 
            alpha=.2, color=color)
        if plot_minmax:
            minima = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').min()
            maxima = cond_tracks[['Track Time', 'Displacement']].groupby('Track Time').max()
            axes[2].fill_between(np.sqrt(median.index), 
                minima['Displacement'], maxima['Displacement'], 
                alpha=.2, color=color)

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
    fig, ax = plt.subplots(1,2, figsize=(8, 4.5))
    ax[0].set_title('Velocity')    
    ax[1].set_title('Turning Angle')
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[0].axis('equal')
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[1].axis('equal')

    null_model = tracks.ix[random.sample(tracks.index, tracks.shape[0])]
    pd.tools.plotting.lag_plot(null_model['Velocity'], c='0.8', ax=ax[0])
    pd.tools.plotting.lag_plot(null_model['Turning Angle'], c='0.8', ax=ax[1])

    for i, (_, cond_tracks) in enumerate(tracks.groupby(condition)):
        color = sns.color_palette()[i]
        for _, track in cond_tracks.groupby('Track_ID'):
            pd.tools.plotting.lag_plot(track['Velocity'], ax=ax[0], c=color)
            pd.tools.plotting.lag_plot(track['Turning Angle'], ax=ax[1], c=color)

    plt.tight_layout()
    if save:
        conditions = [cond.replace('= ', '') 
            for cond in tracks[condition].unique()]
        plt.savefig('Motility-LagPlot_' + '-'.join(conditions) + '.png')
    else:
        plt.show()


if __name__ == "__main__":
    """Demostrates motility analysis of simulated data."""
    tracks = silly_tracks()
    # plot_tracks(tracks)
    # animate_tracks(tracks)
    tracks = analyze_tracks(tracks)
    # plot_joint_motility(tracks)
    # plot_motility(tracks)
    lag_plot(tracks)
