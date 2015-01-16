"""Wrapper to configure, run and analyze excalib2 simulations"""
import os
import subprocess
import shutil
import itertools
import timeit
import pickle
import sys
import datetime

import numpy as np
import pandas as pd

import lana


def read_tracks(posfile='positions.txt', ndim=2):
    try:
        positions = np.loadtxt(posfile)
    except:
        print ('Error: Cannot read positions file!')
        return

    tracks = pd.DataFrame()

    for track_id in range(positions.shape[1]/ndim):
        if np.var (positions[:,ndim*track_id]) != 0:
            if ndim == 3:
                tracks = tracks.append(pd.DataFrame({
                    'Track_ID': track_id,
                    'X': positions[:,ndim*track_id],
                    'Y': positions[:,ndim*track_id+1],
                    'Z': positions[:,ndim*track_id+2]}))
            else:
                tracks = tracks.append(pd.DataFrame({
                    'Track_ID': track_id,
                    'X': positions[:,ndim*track_id],
                    'Y': positions[:,ndim*track_id+1]}))

    return tracks


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
            if line.startswith(parameter):
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
            print('Starting "{}" on {}'.format(self.cmd, 
                datetime.datetime.now().strftime("%d.%m. at %H:%M")))
            subprocess.call('./{}'.format(self.cmd))
            print('Command "{}" run'.format(self.cmd))
        except OSError:
            print ('Error: Command "{}" not found!'.format(self.cmd))
            return

    def read_tracks(self, ndim=2):
        tracks = read_tracks(self.posfile, ndim)
        return tracks


def sweep(simulation, parameters, all_combinations=True, dry_run=False,
    timesteps='', ndim=2, save=False, save_runs=False, palette='PuRd'):
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

    tracks = pd.DataFrame()
    for i, pair in enumerate(values):
        start = timeit.default_timer()
        print('\nSimulation {} of {}:'.format(i+1, values.__len__()))
        print('------------------')
        labels = []
        for j, name in enumerate(names):
            simulation.set_parameter(
                name, pair[j], verbose=True, dry_run=dry_run)
            labels.append(' = '.join([name, str(pair[j])]))
        if dry_run:
            run_tracks = lana.silly_tracks()
        else:
            simulation.run()
            run_tracks = simulation.read_tracks(ndim)

        run_tracks['Condition'] = ', '.join(labels)
        if save_runs:
            run_tracks = lana.analyze_tracks(run_tracks)
            lana.plot_motility(run_tracks, save=True, palette=palette)

        tracks = tracks.append(run_tracks)
        end = timeit.default_timer()
        print('Finished in {}'.format(datetime.timedelta(seconds=end-start)))

    if not save_runs:
        tracks = lana.analyze_tracks(tracks)
    lana.plot_motility(tracks, save=save, palette=palette)

    return tracks


def versus(commands, dry_run=False, save=False, save_runs=False, ndim=2):
    """Simulates all commands and parameter files from a dict"""
    tracks = pd.DataFrame()
    for i, cmd in enumerate(commands.keys()):
        start = timeit.default_timer()
        print('\nSimulation {} of {}:'.format(i+1, commands.keys().__len__()))
        print('------------------')
        if dry_run:
            run_tracks = lana.silly_tracks()
        else:
            with Simulation(cmd, parfile=commands[cmd]) as command:
                command.run()
                run_tracks = command.read_tracks(ndim)
        run_tracks['Condition'] = cmd
        if save_runs:
            run_tracks = lana.analyze_tracks(run_tracks)
            lana.plot_motility(run_tracks, save=True)
        tracks = tracks.append(run_tracks)
        end = timeit.default_timer()
        print('Finished in {}'.format(datetime.timedelta(seconds=end-start)))

    if not save_runs:
        tracks = lana.analyze_tracks(tracks)
    lana.plot_motility(tracks, save=save)

    return tracks


if __name__ == "__main__":
    """Illustrates loading and analyzing file"""
    # tracks = read_tracks('Examples/positions.txt')
    # tracks = lana.analyze_tracks(tracks)
    # lana.plot_motility(tracks)

    """Illustrates dry run parameter sweep"""
    with Simulation('persistence') as persistence:
        sweep(persistence, {'descht': [1,2], 'tschegg': [3,4,5]}, dry_run=True)

    """Illustrates dry run run over several commands"""
    commands = {'cmd1': 'parfile1', 'cmd2': '', 'cmd3': '', 'cmd4': ''}
    versus(commands, dry_run=True)
