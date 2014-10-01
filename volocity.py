"""Functions to handle Volocity cell tracks"""
import numpy as np

import lana


def read_tracks(volocity_file, min_track_length=5):
	with open(volocity_file, 'r') as volocity_file:
	    lines = volocity_file.readlines()

	for i, line in enumerate(lines):
		if 'Centroid X' in line:
			words = line.split('\t')
			for j, word in enumerate(words):
				if 'Timepoint' in word:
					index_time = j
				if 'Centroid X' in word:
					data_begin = i + 1
					index_X = j

	tracks = []; track = [];
	for i, line in enumerate(lines[data_begin:]):
		words = line.split('\t')

		try:
			new_time = int(words[index_time])
			if track == []:
				old_time = new_time - 1
		except ValueError:
			new_time = -666

		if new_time == old_time + 1:
			track.append(map(float, words[index_X:index_X+3]))
			old_time = new_time
		else:
			if track.__len__() >= min_track_length:
				tracks.append(np.array(track))
			track = []
			if new_time != -666: # For new tracks without N/A!
				track.append(map(float, words[index_X:index_X+3]))
				old_time = new_time
	else:
		if track.__len__() >= min_track_length:
			tracks.append(np.array(track))

	return tracks


if __name__ == '__main__':
	tracks = read_tracks('Examples/Volocity_example.txt')
	volocity_example = lana.Motility(tracks, ndim=3, timestep=1)
	volocity_example.plot()
	lana.plot_tracks(volocity_example)
