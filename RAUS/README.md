# Ranking Approaches for Unknown Structure (RAUS) Learning

## Summary
This is the Ranking Approaches for Unknown Structures (RAUS) software.

RAUS can readily learn dynamic Bayesian networks (DBNs) and Bayesian networks (BNs) end-to-end as well as be incorporated into existing pipelines as an unknown structure learning engine.

RAUS reduces the complexity of learning unknown structures for greedy search methods from n! to n=3.

RAUS is particularly useful for auto-generating and saving learned feature rankings, network structures, graph visualizations, and model performance metrics, which can significantly reduce project management complexities.

Further, RAUS can be used to learn the intra-structures and inter-structures required for learning a full network (i.e., connecting static and dynamic variables over time).

Note that python files ending in _block1.py, _block2.py, and _block3.py are the DBNs and BNs efficiently learned using feature ranking methods (Cramer's V, Chi-squared, and information gain, respectively). Also, to adjust the hyperparameter settings for these files update the input arguments.

Further, note that python files ending in _track1.py and _track2:3.py run the python files ending in _block1.py, _block2.py, and _block3.py in parallel (i.e., three processes per track). Therefore, depending on the number of CPU cores available on your local/remote machine, python files ending in _track1.py and _track2:3.py can be run simultaneously (i.e., six processes). Also, to adjust the hyperparameter settings for these files update the input arguments.

Further, to run RAUS at multiple prediction windows (or on multiple datasets, multiple outcomes, multiple sites) simultaneously, you can run multiple python files ending in _track1.py and multiple python files ending in _track2:3.py in parallel (again depending on the number of CPU cores available on your local/remote machine).

RAUS can handle both train test (TT) split and train validation test (TVT) split input.

RAUS is built on top of the Bayes Net Toolbox (BNT) by [Murphy et. al.](https://github.com/bayesnet/bnt)

# Cite

Please cite the RAUS software if you use the RAUS software in your work.

The RAUS software is first implemented in the paper: "Dynamic Bayesian Networks for Predicting Acute Kidney Injury Before Onset" by David Gordon et al.

# How to use RAUS

## Installation

To install octave (version 4.2.2) for ubuntu (18.04), run:
# install octave build dependencies for ubuntu (18.04) (~2 minutes)
```shell
$ sudo apt-get install gcc g++ gfortran make libopenblas-dev liblapack-dev libpcre3-dev libarpack2-dev libcurl4-gnutls-dev epstool libfftw3-dev transfig libfltk1.3-dev libfontconfig1-dev libfreetype6-dev libgl2ps-dev libglpk-dev libreadline-dev gnuplot-x11 libgraphicsmagick++1-dev libhdf5-serial-dev openjdk-8-jdk libsndfile1-dev llvm-dev lpr texinfo libgl1-mesa-dev pstoedit portaudio19-dev libqhull-dev libqrupdate-dev libqscintilla2-dev libsuitesparse-dev texlive texlive-generic-recommended libxft-dev zlib1g-dev autoconf automake bison flex gperf gzip icoutils librsvg2-bin libtool perl rsync tar qtbase5-dev qttools5-dev qttools5-dev-tools libqscintilla2-qt5-dev

```
# install octave (version 4.2.2) from source (~10 minutes)
```shell
$ wget https://ftp.gnu.org/gnu/octave/octave-4.2.2.tar.gz
$ tar -xf octave-4.2.2.tar.gz                             
$ cd octave-4.2.2
$ ./configure
$ make -j8
$ sudo make install

```

To install anaconda (version 5.1.0), run:
```shell
$ wget https://repo.anaconda.com/archive/Anaconda3-5.1.0-Linux-x86_64.sh
$ bash Anaconda3-5.1.0-Linux-x86_64.sh

```

To create the conda environment, run:
```shell
$ conda create --name raus python=3.7.3
$ conda activate raus
$ pip install -r requirements.txt

```

## Example Commands

To run the pipeline and return RAUS track 1 and track2:3 for the AKI dataset, run:

```shell
$ screen -S AKI_BOS24_raus_track1 python AKI_BOS24_track1.py & screen -S AKI_BOS48_raus_track1 python AKI_BOS48_track1.py & screen -S AKI_BOS72_raus_track1 python AKI_BOS72_track1.py & screen -S AKI_BOS24_raus_track2:3 python AKI_BOS24_track2:3.py & screen -S AKI_BOS48_raus_track2:3 python AKI_BOS48_track2:3.py & screen -S AKI_BOS72_raus_track2:3 python AKI_BOS72_track2:3.py

```

To run the pipeline and return RAUS track 1 for the AKI dataset, run:

```shell
$ screen -S AKI_BOS24_raus_track1 python AKI_BOS24_track1.py & screen -S AKI_BOS48_raus_track1 python AKI_BOS48_track1.py & screen -S AKI_BOS72_raus_track1 python AKI_BOS72_track1.py

```

To run the pipeline and return RAUS track 2:3 for the AKI dataset, run:

```shell
$ screen -S AKI_BOS24_raus_track2:3 python AKI_BOS24_track2:3.py & screen -S AKI_BOS48_raus_track2:3 python AKI_BOS48_track2:3.py & screen -S AKI_BOS72_raus_track2:3 python AKI_BOS72_track2:3.py

```

To run the pipeline and return RAUS track 1 just for the 48-hour before onset (BOS) prediction window for the AKI dataset, run:

```shell
$ screen -S AKI_BOS48_raus_track1 python AKI_BOS48_track1.py

```

To run the pipeline and return RAUS track 2:3 just for the 48-hour BOS prediction window for the AKI dataset, run:

```shell
$ screen -S AKI_BOS48_raus_track2:3 python AKI_BOS48_track2:3.py

```
