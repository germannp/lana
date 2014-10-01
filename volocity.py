"""Functions to handle Volocity cell tracks"""
import numpy as np

import lana


def read_tracks(volocity_file, min_track_length=5):
	"""Reads a list of numpy arrays from Volocity files"""
	with open(volocity_file, 'r') as volocity_file:
	    lines = volocity_file.readlines()

	for i, line in enumerate(lines):
		if 'Centroid X' in line:
			words = line.split('\t')
			for j, word in enumerate(words):
				if 'Track ID' in word:
					index_track_id = j
				if 'Centroid X' in word:
					data_begin = i + 1
					index_X = j

	tracks = []; track = [];
	for i, line in enumerate(lines[data_begin:]):
		words = line.split('\t')

		if track == []:
			track_id = words[index_track_id]

		if words[index_track_id] == track_id:
			if words[index_X] != 'N/A':
				track.append(map(float, words[index_X:index_X+3]))
		else:
			track_id = words[index_track_id]
			if track.__len__() >= min_track_length:
				tracks.append(np.array(track))
			track = []
			if words[index_X] != 'N/A':
				track.append(map(float, words[index_X:index_X+3]))
	else:
		if track.__len__() >= min_track_length:
			tracks.append(np.array(track))

	return tracks


if __name__ == '__main__':
    """Illustrates the analysis of Volocity data"""
	tracks = read_tracks('Examples/Volocity_example.txt')
	volocity_example = lana.Motility(tracks, ndim=3, timestep=1)
	volocity_example.plot()
	lana.plot_tracks(volocity_example)
