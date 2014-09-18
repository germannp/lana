"""Wrapper to configure, run and analyze excalib2 simulations"""
import os
import subprocess
import shutil
import itertools
import timeit
import pickle
import sys

import numpy as np

import lana


class Simulation:
    """Configures and runs excalib2 CPM simulations"""
    def __init__(self, cmd, parfile='', posfile='positions.txt'):
        self.cmd = cmd

        if parfile == '':
            self.parfile = ''.join([cmd, '.par'])
        else:
            self.parfile = parfile
        self.parfile_backup = ''.join([self.parfile, '~'])

        self.posfile = posfile

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        if os.path.isfile(self.parfile_backup):
            shutil.copyfile(self.parfile_backup, self.parfile)
            os.remove(self.parfile_backup)

    def set_parameter(self, parameter, value, verbose=False, dry_run=False):
        if verbose:
            print('Setting {} to {}'.format(parameter, value))

        if dry_run:
            return

        if not os.path.isfile(self.parfile_backup):
            try:
                shutil.copyfile(self.parfile, self.parfile_backup)
            except IOError:
                print('Error: Parameter file "{}" not found!'.format(
                    self.parfile))
                return

        with open(self.parfile, 'r') as parfile:
            lines = parfile.readlines()
        for i, line in enumerate(lines):
            if parameter in line:
                words = line.split()
                words[1] = str(value)
                words.append('\n')
                lines[i] = '\t'.join(words)
                break
        else:
            lines.append('\t'.join([parameter, str(value), '\n']))

        with open(self.parfile, 'w') as parfile:
            parfile.writelines(lines)

    def get_parameter(self, parameter):
        with open(self.parfile, 'r') as parfile:
            lines = parfile.readlines()
        for i, line in enumerate(lines):
            if parameter in line:
                words = line.split()
                return words[1]
        else:
            print('Parameter not found!')

    def run(self):
        try:
            subprocess.call('./{}'.format(self.cmd))
            print('Command "{}" run'.format(self.cmd))
        except OSError:
            print ('Error: Command "{}" not found!'.format(self.cmd))
            return

    def read_positions(self):
        try:
            return np.loadtxt(self.posfile)
        except:
            print ('Error: Cannot read positions file!')
            return


def sweep(Simulation, parameters, all_combinations=True, dry_run=False,
        timesteps='', ndim=2, save_runs=False, wo_legend=False, 
        palette='PuRd'):
    """Simulates all combinations or pairs of parameters from a dict"""
    try:
        names = parameters.keys()
    except AttributeError:
        print('Error: Parameters have to be passed as dict!')
        return

    if all_combinations:
        values = list(itertools.product(*parameters.values()))
    else:
        values = list(zip(*parameters.values()))

    if timesteps == '':
        timesteps = np.ones(values.__len__())

    if ndim == 2:
        rand_angles_2D = True
        rand_angles_3D = False
    if ndim == 3:
        rand_angles_2D = False
        rand_angles_3D = True

    velocities = []; turning_angles = []; displacements = []; times = [] 
    legends = []
    for i, pair in enumerate(values):
        labels = []
        start = timeit.default_timer()
        print('\nSimulation {} of {}:'.format(i+1, values.__len__()))
        for j, name in enumerate(names):
            Simulation.set_parameter(
                name, pair[j], verbose=True, dry_run=dry_run)
            labels.append(' = '.join([name, str(pair[j])]))
        if dry_run:
            CurrentMotility = lana.Motility()
        else:
            Simulation.run()
            CurrentMotility = lana.Motility(Simulation.read_positions(),
                timestep=timesteps[i], ndim=ndim)
            if save_runs:
                CurrentMotility.plot(save_as='_'.join([Simulation.cmd, 
                    '-'.join(labels).replace(' = ', '')]), 
                    rand_angles_2D=rand_angles_2D,
                    rand_angles_3D=rand_angles_3D, palette=palette)
        velocities.append(CurrentMotility.velocities().reshape(-1))
        turning_angles.append(CurrentMotility.turning_angles().reshape(-1))
        displacements.append(CurrentMotility.displacements())
        times.append(CurrentMotility.times())
        legends.append(', '.join(labels))
        end = timeit.default_timer()
        print('Finished in {} seconds'.format(end-start))

    if dry_run:
        save_as = ''
    else:
        save_as = ''.join(
             [Simulation.cmd, '_', '-'.join(parameters.keys()), '-Sweep'])
        with open('{}.py{}kl'.format(save_as, sys.version[0]), 'wb') as pikl:
            pickle.dump([velocities, turning_angles, displacements], pikl)
            # Might not be readable from other python version.

    if wo_legend:
        legends = ''

    lana.motility_plot(velocities, turning_angles, displacements, times, 
        legends=legends, save_as=save_as, palette=palette)


def versus(commands, dry_run=False, save_runs=False):
    """Simulates all commands and parameter files from a dict"""
    velocities = []; turning_angles = []; displacements = []; times = []
    for i, cmd in enumerate(commands.keys()):
        start = timeit.default_timer()
        print(''.join(
            ['\nSimulation ', str(i+1), '/', 
            str(commands.keys().__len__()), ':']))
        if dry_run:
            CurrentMotility = lana.Motility()
        else:
            with Simulation(cmd, parfile=commands[cmd]) as Command:
                Command.run()
                CurrentMotility = lana.Motility(Command.read_positions())
                if save_runs:
                    CurrentMotility.plot(save_as=cmd)
        velocities.append(CurrentMotility.velocities().reshape(-1))
        turning_angles.append(CurrentMotility.turning_angles().reshape(-1))
        displacements.append(CurrentMotility.displacements())
        times.append(CurrentMotility.times())
        end = timeit.default_timer()
        print('Finished in {} seconds'.format(end-start))

    if dry_run:
        save_as = ''
    else:
        save_as = '_versus_'.join(commands.keys())

    lana.motility_plot(velocities, turning_angles, displacements, times, 
        legends=commands.keys(), save_as=save_as)


if __name__ == "__main__":
    """Illustrates dry run parameter sweep"""
    with Simulation('persistence') as Persistence:
        sweep(Persistence, {'descht': [1,2], 'tschegg': [3,4,5]}, dry_run=True, save_runs=True)
        # sweep(Persistence, {'graphics': [1,1337]}, dry_run=True)
        # sweep(Persistence, {'graphics': [1,1337], 'descht': [1,2,3]}, dry_run=True)

    """Illustrates dry run run over several commands"""
    commands = {'cmd1': 'parfile1', 'cmd2': '', 'cmd3': '', 'cmd4': ''}
    versus(commands, dry_run=True)
