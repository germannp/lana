lana
====

A toolbox to analyze lymphocyte tracks within lymphnodes from microscopy or simulations based on [numpy](http://www.numpy.org/), [matplotlib](http://matplotlib.org/) and [seaborn](http://web.stanford.edu/~mwaskom/software/seaborn/). It produces figures like the following example of a parameter sweep in a celluar Potts model:

![alt text](Examples/sweep.png "Plot of a parameter sweep")


Modules
-------
  * **lana.py**: Core tools to analyze cell motility from positions within lymph nodes. Handles data from experiments or simulations and plots the analaysis.
  * **excalib2.py**: Wrapper to configure, run and analyze excalib2 cellular Potts model simulations. Includes functions to run parameter sweeps or compare different commands.
  * **volocity.py**: Handles [Volocity](http://www.perkinelmer.co.uk/volocity) cell tracks.
