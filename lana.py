"""Tools to analyze  cell motility from positions within lymph nodes"""
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def silly_walk(positions=None, steps=1600, ncells=30, ndim=2):
    """Generates a 2D random walk after Nombela-Arrieta et al. 2007"""
    step_size=np.random.rand()
    for _ in range(steps):
        if positions == None:
            positions = np.random.rand(1, ncells*ndim)
        else:
            ncells = positions.shape[1]/ndim
            if positions.shape[0] == 1:
                angles = 2*np.pi*np.random.rand(1, ncells)
            else:
                last_steps = positions[-1] - positions[-2]
                last_displacements = np.sqrt(np.sum(
                    last_steps.reshape(-1,ndim)**2, axis=1)).reshape(-1,ncells)
                x = last_steps[0::ndim]/last_displacements
                y = last_steps[1::ndim]/last_displacements
                angles = (np.sign(x) - 1)/2*np.pi + np.sign(x)*np.arcsin(y)
            silly_angles = angles + np.random.lognormal(0, 0.5, (1, ncells)) \
                * (2*np.random.randint(0, 2, ncells) - 1)
            silly_displacement = np.random.lognormal(
                0, 0.5, (ncells, 1))*step_size
            silly_steps = silly_displacement*np.concatenate(
                (np.cos(silly_angles.T), np.sin(silly_angles.T)), axis=1)
            positions = np.concatenate(
                (positions, positions[-1] + silly_steps.reshape(1,-1)))
    print('Created simulated data')
    return positions


def motility_plot(velocities, turning_angles, displacements, times='',
        legends='', rand_angles_2D=True, rand_angles_3D=False,
        save_as='', palette='deep'):
    """Plots aspects of motility for single arrays or lists of arrays"""
    if isinstance(velocities, list):
        nmea_velo = float(velocities.__len__())
    else:
        nmea_velo = 1
        foo = []
        foo.append(velocities)
        velocities = foo

    if isinstance(turning_angles, list):
        nmea_angles = float(turning_angles.__len__())
    else:
        nmea_angles = 1
        foo = []
        foo.append(turning_angles)
        turning_angles = foo

    if isinstance(displacements, list):
        nmea_displ = float(displacements.__len__())
    else:
        nmea_displ = 1
        foo = []
        foo.append(displacements)
        displacements = foo

    if not times == '' and times.__len__() != nmea_displ:
        foo = []
        foo.append(times)
        times = foo

    sns.set(style="white", palette=sns.color_palette(
        palette, int(max(nmea_velo, nmea_displ, nmea_angles))))
    sns.set_context("paper", font_scale=1.5)
    figure, axes = plt.subplots(ncols=3, figsize=(12,6))
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.setp(axes, yticks=[])
    plt.setp(axes, xticks=[])

    axes[0].set_title('Velocities')
    axes[0].set_xlim(0, np.max([np.percentile(v, 99.5) for v in velocities]))
    axes[0].set_xticks([0])
    axes[0].set_xticklabels([r'$0$'])
    for velocity in velocities:
        sns.kdeplot(velocity, shade=True, ax=axes[0])

    axes[1].set_title('Turning Angles')
    axes[1].set_xlim([0,np.pi])
    axes[1].set_xticks([0, np.pi/2, np.pi])
    axes[1].set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$'])
    for turning_angle in turning_angles:
        turning_angle = np.concatenate((
            -turning_angle, turning_angle, 2*np.pi-turning_angle))
        sns.kdeplot(turning_angle, shade=True, ax=axes[1])
    if rand_angles_2D:
        axes[1].plot([0, np.pi], [1/(3*np.pi), 1/(3*np.pi)], '--k')
    if rand_angles_3D:
        x = np.arange(0, np.pi, 0.1)
        axes[1].plot(x, np.sin(x)/6, '--k')

    axes[2].set_title('Mean Displacements')
    axes[2].set_xlabel('sqrt of time')
    axes[2].set_xlim(0,100)
    for i, displacement in enumerate(displacements):
        if times == '':
            time = range(0, displacement.shape[-2])
        else:
            time = times[i]
        color = sns.color_palette(n_colors=i+1)[-1]
        if legends == '':
            legend = False
            label = ''
        else:
            legend = True
            label = legends[i]
        sns.tsplot(displacement.T, time=np.sqrt(time), ax=axes[2], color=color,
            legend=legend, condition=label)

    if save_as == '':
        plt.show()
    else:
        plt.savefig('{}.png'.format(save_as))


class Motility:
    """Analyzes motility of attributed numpy array of positions"""
    def __init__(self, positions=None, ndim=2, timestep=25):
        if positions == None:
            self.positions = silly_walk(ndim=ndim)
        else:
            self.positions = positions
        self.ndim = ndim
        self.ncells = self.positions.shape[1]/ndim
        self.timestep = int(timestep)
        print('Analyzing {} cells, every {} seconds'.format(
            self.ncells, self.timestep))

    def velocities(self):
        differences = (self.positions[1:] - self.positions[:-1])/self.timestep
        return np.linalg.norm(
            differences.reshape((-1,self.ndim)),axis=1).reshape(-1,self.ncells)

    def turning_angles(self):
        differences = self.positions[1:] - self.positions[:-1]
        dot_products = np.sum(
            (differences[1:]*differences[:-1]).reshape((-1,self.ndim)),
            axis=1).reshape((-1,self.ncells))
        difference_norms = np.sum(
            differences.reshape((-1,self.ndim))**2, 
            axis=1).reshape((-1,self.ncells))
        norm_products = np.sqrt(difference_norms[1:]*difference_norms[:-1])
        return np.arccos(
            np.nan_to_num(np.clip(dot_products/norm_products, -1, 1)))

    def displacements(self):
        return np.linalg.norm(
            (self.positions - self.positions[0]).reshape((-1,self.ndim)), 
            axis=1).reshape((-1,self.ncells))

    def times(self):
        return range(0, self.displacements().shape[0]*self.timestep, 
            self.timestep)

    def motility_coeffs(self):
        return self.positions[-1]**2/(4*self.positions.shape[0]*self.timestep)

    def plot(self, *args, **kwargs):
        motility_plot(
            self.velocities().reshape(-1), self.turning_angles().reshape(-1),
            self.displacements(), self.times(), *args, **kwargs)


if __name__ == "__main__":
    """Demostrates motility analysis of simulated data."""
    T_cells = Motility()
    T_cells.plot()
