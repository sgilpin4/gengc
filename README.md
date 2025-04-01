# A Generalized Gaspari-Cohn (GenGC) Correlation Function

## Author
## Shay Gilpin

**Please cite the publication and code DOI when you use this code**
- Gilpin, S., Matsuo, T., Cohn, S.E. (2023). _A generalized, compactly-supported correlation function for data assimilation applications._ Q. J. Roy. Meteor. Soc., [https://doi.org/10.1002/qj.4490](https://doi.org/10.1002/qj.4490)
- Gilpin, S. _A Generalized Gaspari-Cohn (GenGC) Correlation Function. [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7859258.svg)](https://doi.org/10.5281/zenodo.7859258)


## Background
The GenGC correlation function is a generalization of the compactly-supported, piecewise rational correlation function introduced by Gaspari and Cohn (1999) and the subsequent extension by Gaspari et al. (2016). The GenGC correlation function allows both parameters a and c to vary, as continous functions, over the domain to generate inhomogeneous and anisoptric correlation functions that remain compactly-supported.

This repository contains the Python code that implements the GenGC correlation function introduced in Gilpin et al. (2023). Two implementations of the GenGC correlation function are included: gengc.py that can be adapted to other languages such as Fortran, and gengc1d.py, an object-oriented version specific to constructing 1D correlations in Python. In addition, this repo contains the scripts to reproduce 1D examples in Sections 3 and 4 of Gilpin et al. (2023).

See requirements.txt for the additional Python packages and their versions needed to run these scripts.


## File Descriptions
_Note: all Python scripts include comments that further detail their use._

**gengc.py**
Standard implementation of the GenGC correlation function. This GenGC Python function as a matrix-vector operator that outputs the GenGC correlation matrix acting on an input vector z, which is a vector of scalars of the normed distance between grid cells. This version is not object-oriented is and can be adapted to other coding languages such as Fortran with minimal modifications.

**gengc.f90**
(Updated April 1, 2025): Fortran 90 version of gengc.py. Please not this is a preliminary version that should be modified based on your own version of Fortran!

**gengc1d.py**
Object-oriented construction of the GenGC correlation function. This script is written specifically for Python to output the GenGC correlation function as a 2D numpy array. To optimize construction, two methods are implemented:
- Continuous form of GenGC, in which a and c are specified as continuous functions over the domain
- Piecewise constant form, in which at least one subdomain is larger than one grid cell and a and c are defined as piecewise constant functions.
This script calls gengc_conv.py to evaluate the GenGC correlation function.

**gengc_conv.py**
Explicit implementation of the GenGC correlation function that is used to construct correlation matrices in gengc1d.py. The coefficient tables in Tables 1-19 in Appendix C of Gilpin et al. (2023) are implemented here in a slightly different form than done in gengc.py.

**run_examples.py**
Generates Figures 2, 4-6 from Gilpin et al. (2023) by running this script. Gives several examples of how to construct the GenGC correlation function using the gengc1d.py script.


## Usage
Please see comments of the specific scripts to see how to use this function.
