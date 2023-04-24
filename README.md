# A Generalized Gaspari-Cohn (GenGC) Correlation Function

## Author
Shay Gilpin

**Please cite the publication and code DOI when you use this code**
- Gilpin, S., Matsuo, T., Cohn, S.E. (2023). _A generalized, compactly-supported correlation function for data assimilation applications._ Q. J. Roy. Meteor. Soc.


## Background
The GenGC correlation function is a generalization of the compactly-supported, piecewise rational correlation function introduced in Gaspari and Cohn (1999) and the subsequent extension in Gaspari et al. (2016). The GenGC correlation function allows both parameters a and c to vary, as continous functions, over the domain to generate inhomogeneous and anisoptric correlation functions that remain compactly-supported.

This repository contains the Python code that implements the GenGC correlation function introduced in Gilpin et al., (2023). Two implementations of the GenGC correlation function are included: gengc.py that can be adapted to other languages such as Fortran, and gengc1d.py, an object-oriented version specific to constructing 1D correlations in Python. In addition, this repo contains the scripts to reproduce 1D examples in Sections 3 and 4 of Gilpin et al., (2023).

See requirements.txt for the additional Python packages and their versions needed to run these scripts.


## File Descriptions
_Note: all Python scripts include comments that further detail their use_

**gengc.py**
Standard implementation of the GenGC correlation function. This GenGC Python function as a matrix-vector operator that outputs the GenGC correlation matrix acting on an input vector z, which is a vector of scalars of the normed distance between grid cells. This version is not object-oriented is and can be adapted to other coding languages such as Fortran with minimal modifications.

**gengc1d.py**
Object-oriented construction of the GenGC correlation function. This script is written specifically for Python to output the GenGC correlation function as a 2D numpy array. To optimize construction, two methods are implemented:
- Continuous form of GenGC, in which a and c are specified as continuous functions over the domain
- Piecewise constant form, in which each subdomain is larger than one grid cell and a and c are defined as piecewise constant functions.
This script calls gengc_conv.py to evaluate the GenGC correlation function.

**gengc_conv.py**
Explicit implementation of the GenGC correlation function that is used to construct correlation matrices in gengc1d.py. The coefficient tables in Tables 1-19 in Appendix C of Gilpin et al (2023) are implemented here.

**run_examples.py**
Generates Figures 2, 4--6 from Gilpin et al. (2023) by running this script. Gives several examples of how to construct the GenGC correlation function using the gengc1d.py script.


## Usage
Please see comments of the specific scripts to see how to use this function.
