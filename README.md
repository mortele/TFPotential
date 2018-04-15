[![Project Status: Inactive – The project has reached a stable, usable state but is no longer being actively developed; support/maintenance will be provided as time allows.](http://www.repostatus.org/badges/latest/inactive.svg)](http://www.repostatus.org/#inactive)
# Tensorflow potential

An artificial neural network framework written to allow training on multi dimensional potential energy surface data set, resulting from ab initio calculations at e.g. the Hartree-Fock level of theory. This framework was developed for [the master thesis of Morten Ledum](https://www.duo.uio.no/handle/10852/61196). The code has been tested and validated for energy as a function of one or two molecular degrees of freedom, but can in principle handle any number. The relatively simple training regime used in the present work will probably break down if you try to use much more than two.

The Python API of the Tensorflow package is employed, and it is necessary at run-time for `$PYTHONPATH` to point to the location of the Tensorflow binaries such that `import tensorflow as tf` can execute.

## Building and running

1. Download and install Tensorflow from https://www.tensorflow.org/install/. Make sure to pick the Python2.7 version.
2. Clone the git repository: git@github.com:mortele/TFPotential.git. 
3. Ensure Tensorflow is in your `$PYTHONPATH` and run `python2.7 tfpotential.py` from the newly created local clone. In practice, you *probably* installed Tensorflow in a virtual environment; make sure you activate it by e.g. (if you installed Tensorflow via Anaconda) `source activate <environment name>`.
4. Running `python2.7 tfpotential --help` will show an overview of the many available command line options.
5. For advanced training and fine tuning it is most likely necessary to directly modify the source, as the built in command line arguments can only do so much.

Please note that this has never been tested on Windows machines, but it *might* work. Anyway, if anything breaks you get to keep all the pieces.
