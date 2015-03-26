lana
====

A toolbox to analyze lymphocyte tracks within lymphnodes from microscopy or simulations based on [Pandas](http://pandas.pydata.org/), [numpy](http://www.numpy.org/), [matplotlib](http://matplotlib.org/), [seaborn](http://web.stanford.edu/~mwaskom/software/seaborn/) and [scikit-learn](http://scikit-learn.org/). It produces figures like the following example of a parameter sweep in a celluar Potts model:

![alt text](Examples/sweep.png "Plot of a parameter sweep")


Modules
-------
  * **motility.py**: Tools to analyze cell motility from positions within lymph nodes. Handles data from experiments or simulations and plots the analaysis.
  * **remix.py**: Statistical models to generate tracks based on data.
  * **excalib2.py**: Wrapper to configure, run and analyze excalib2 cellular Potts model simulations. Includes functions to run parameter sweeps or compare different commands.
  * **imaris.py**: Loads cell tracks from excel spreadsheets exported from [Imaris](http://www.bitplane.com/imaris/imaris).
  * **volocity.py**: Handles [Volocity](http://www.perkinelmer.co.uk/volocity) cell tracks.
  * **utils.py**: Snippets used in different modules.
