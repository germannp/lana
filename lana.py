"""Tools to analyze and plot cell motility from tracks within lymph nodes"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def silly_track(init_position=np.random.rand(1,2), steps=25, step_size=1):
    """Generates a 2D random walk after Nombela-Arrieta et al. 2007"""
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


def plot_tracks(motilities, save_as='', palette='deep'):
    """Plots the tracks of a (list of) Motility class(es)"""
    if not isinstance(motilities, list):
        foo = []
        foo.append(motilities)
        motilities = foo

    sns.set(style="white", palette=sns.color_palette(
        palette, motilities.__len__()))
    sns.set_context("paper", font_scale=1.5)

    plt.axis('equal')

    for i, motility in enumerate(motilities):
        color = sns.color_palette()[i]
        for track in motility.tracks:
            plt.plot(track[:,0]-track[0,0], track[:,1]-track[0,1], color=color)

    if save_as == '':
        plt.show()
    else:
        plt.savefig('{}.png'.format(save_as))


def plot_motility(motilities, save_as='', palette='deep'):
    """Plots aspects of motility for a (list of) Motility class(es)"""
    if not isinstance(motilities, list):
        foo = []
        foo.append(motilities)
        motilities = foo

    sns.set(style="white", palette=sns.color_palette(
        palette, motilities.__len__()))
    sns.set_context("paper", font_scale=1.5)
    figure, axes = plt.subplots(ncols=3, figsize=(12,6))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.setp(axes, yticks=[])
    plt.setp(axes, xticks=[])

    axes[0].set_title('Velocities')
    axes[0].set_xlim(0, np.max([np.percentile(v, 99.5) for v in 
        [motility.velocities() for motility in motilities]]))
    axes[0].set_xticks([0])
    axes[0].set_xticklabels([r'$0$'])

    axes[1].set_title('Turning Angles')
    axes[1].set_xlim([0,np.pi])
    axes[1].set_xticks([0, np.pi/2, np.pi])
    axes[1].set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])

    axes[2].set_title('Mean Displacements')
    axes[2].set_xlabel('sqrt of time')

    for i, motility in enumerate(motilities):
        if motility.label == '':
            sns.kdeplot(motility.velocities(),
                shade=True, ax=axes[0])
        else:
            sns.kdeplot(motility.velocities(), 
                shade=True, ax=axes[0], label=motility.label)
        turning_angles = motility.turning_angles()
        if motility.ndim == 2:
            turning_angles = np.concatenate(( # Mirror at boundaries.
                -turning_angles, turning_angles, 2*np.pi-turning_angles))
            axes[1].plot([0, np.pi], [1/(3*np.pi), 1/(3*np.pi)], '--k')
        if motility.ndim == 3:
            x = np.arange(0, np.pi, 0.1)
            axes[1].plot(x, np.sin(x)/2, '--k')
        sns.kdeplot(turning_angles, shade=True, ax=axes[1])
        color = sns.color_palette(n_colors=i+1)[-1]
        sns.tsplot(motility.displacements(), time='sqrt of time', unit='cell', 
            value='displacement', ax=axes[2], color=color)

    if save_as == '':
        plt.show()
    else:
        plt.savefig('{}.png'.format(save_as))


class Motility:
    """Analyzes motility of attributed list of tracks"""
    def __init__(self, tracks=None, ndim=2, timestep=20, label=''):
        if tracks == None:
            print('Simulating data')
            rand = np.random.rand()
            self.tracks = [silly_track(step_size=rand) for _ in range(100)]
        else:
            self.tracks = tracks
        self.ndim = ndim
        self.ncells = self.tracks.__len__()
        self.timestep = int(timestep)
        self.label = label
        print('Analyzing {} cells, every {} seconds'.format(
            self.ncells, self.timestep))

    def velocities(self):
        velocities = []
        for track in self.tracks:
            differences = (track[1:] - track[:-1])/self.timestep
            velocities.append(np.linalg.norm(differences, axis=1))
        return np.hstack(velocities)

    def mean_velocities(self):
        mean_velocities = []
        for track in self.tracks:
            differences = (track[1:] - track[:-1])/self.timestep
            mean_velocities.append(np.mean(np.linalg.norm(
                differences, axis=1)))
        return np.hstack(mean_velocities)

    def turning_angles(self):
        turning_angles = []
        for track in self.tracks:
            differences = track[1:] - track[:-1]
            dot_products = np.sum(differences[1:]*differences[:-1], axis=1)
            difference_norms = np.linalg.norm(differences, axis=1)
            norm_products = difference_norms[1:]*difference_norms[:-1]
            turning_angles.append(np.arccos(
                np.nan_to_num(np.clip(dot_products/norm_products, -1, 1))))
        return np.hstack(turning_angles)

    def displacements(self):
        displacements = pd.DataFrame()
        for i, track in enumerate(self.tracks):
            time = np.arange(0, track.shape[0]*self.timestep, 
                self.timestep).reshape(-1,1)
            cell = np.ones((track.shape[0], 1))*i
            displacement = np.linalg.norm(track - track[0], 
                axis=1).reshape(-1,1)
            displacements = displacements.append(pd.DataFrame(
                np.hstack((time, np.sqrt(time), cell, displacement)), 
                columns=['time', 'sqrt of time', 'cell', 'displacement']))
        return displacements

    # def motility_coeffs(self):
    #     return self.tracks[-1]**2/(4*self.tracks.shape[0]*self.timestep)

    def plot(self, *args, **kwargs):
        plot_motility(self, *args, **kwargs)


if __name__ == "__main__":
    """Demostrates motility analysis of simulated data."""
    T_cells = Motility()
    T_cells.plot()
    plot_tracks(T_cells)
