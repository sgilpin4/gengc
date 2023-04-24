"""
Generalized Gaspari-Cohn Convolution function
Shay Gilpin
created June 27, 2022
last updated April 24, 2023

** ------------------------------------------------------------------ **
        ** CITE THE CORRESPONDING PUBLICATION AND CODE DOI **
(paper): Gilpin, S., Matsuo, T., and Cohn, S.E. (2023). A generalized,
         compactly-supported correlation function for data assimilation
         applications. Q. J. Roy. Meteor. Soc.
(code): Gilpin, S. A Generalized Gaspari-Cohn Correlation Function
        (see Github repo for DOI)
** ------------------------------------------------------------------ **


Generates the GenGC class object used to construct correlation matrices
in gengc1d.py. The construction of the GenGC correlation matrix is
done in gengc1d.py.

This script is organized as follows:
The Generalized Gaspari-Cohn Function is defined as
B_{kl}(z;ak,al,ck,cl)
where z = ||x1 - x_2|| with x1, x2 two locations in the spatial domain \Omega
with x1 in subregion \Omega_k and x2 in subdomain \Omega_l. The quantities 
ak, ck (al, cl) correspond to the values for the subdomain \Omega_k 
(\Omega_l) B_{kl} is a sixth-order piecewise rational function where 
(assuming ck <= cl, if not the indicies need to be switched)
                        | f1(z;ak,al,ck,cl),    0 < ck <= cl/4
                        | f2(z;ak,al,ck,cl),    cl/4 <= ck <= cl/3
                        | f3(z;ak,al,ck,cl),    cl/3 <= ck <= cl/2
B_{kl}(z;ak,al,ck,cl) = | f4(z;ak,al,ck,cl),    cl/2 <= ck <= 2cl/3
                        | f5(z;ak,al,ck,cl),    2cl/3 <= ck <= 3cl/4
                        | f6(z;ak,al,ck,cl),    3cl/4 <= ck <= cl
                        | f7(z;ak,al,ck,cl),    ck = cl **(from Gaspari et al, (2006)
                        | 0,                    ck > cl
where each function fi, i=1,...,6 is a piecewise function split into 
12 different functions which depend on the relationship between z 
and ck, cl. Each function fij is defined below as its 
own function, the compiled together using np.piecewise below in the 
GenGC class. The coefficients for each function are those listed in the 
Tables of Appendix C of Gilpin et al. (2023).

The GenGC class (defined at the bottom of the script) is the correlation 
object. It can be called as GenGC(a1,a2,c1,c2) to create the object and 
evaluated by calling GenGC(ak,al,ck,cl)(z),where z can be an array of values
with a, c each scalars (not arrays).

**IMPORTANT** For the computation of B_{kl}, it is necessary that 
ck <= cl. If the opposite is true, the indicies must be switched. When 
the GenGC object is initialized, it assigns c1 and c2 to ck and cl 
variables based on this relation so the computations are computed directly.
The user does not have to make this check when calling the GenGC object.

Additional functions include
nfcn(aa): because this is used more than once, it is its own function
normalize(): computes the normalization constant for f1,...,f6. The 
normalizationconstant is different for f7, which is why it is computed 
separately.

The first part of this script implements the individual functions f_{ij}.
The latter part of this script includes to class objects used to construct 
the GenGC correlation function depending on the application:

GenGC Class:
Built for the case that a and c are piecewise constant over \Omega 
(not continuous functions) and thus used in the PWConstantGenGC class 
in gengc1d.py

Continuous GenGC Class:
Built for the case that a and c are continuous functions over \Omega 
and utlizes masks in order to construct the correlation matrix. This is 
used in ContinuousGenGC
"""

import numpy as np
# # functions f1j(z,ak,al,ck,cl) for j=1,...,12
# # need to normalize these functions!!
# # they also assume that ck <= cl by definition of the convolution

def zero(z,ak,al,ck,cl):
    """
    for the case where z>=ck+cl or any other
    time where the output needs to be zero
    """
    return 0.

# # -----------------------f1(z) where 0 < ck <= cl/4 ----------------- # # 
def f11(z,ak,al,ck,cl):
    """
    Case where 0 < ck <= cl/4
    0 <= z <= ck/2
    """
    return z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + z**4*16.*(1.-al)/cl + z**2*40.*ck**2*(al-1.-6.*ak+6.*ak*al)/(3.*cl) + 5.*ck**3*(1.+14.*ak) + 3.*ck**4*(al-1.-30.*ak+30.*ak*al)/cl

def f12(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    ck/2 < z <= ck
    """
    return z**5*32.*(ak*al-ak)/(3.*ck*cl) + \
        z**4*32.*(ak-ak*al)/cl + \
        z**2*320.*ck**2*(ak*al-ak)/(3.*cl) +\
        z*10.*ck**3*(-1.+2.*ak+al-2.*ak*al)/cl +\
        5.*ck**3*(1.+14.*ak)+96.*ck**4*(ak*al-ak)/cl +\
        ck**5*(-1.+2.*ak+al-2.*ak*al)/(3.*z*cl)

def f13(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    ck < z <= cl/2-ck
    """
    return z*10.*ck**3*(-1.-14.*ak+al+14.*ak*al)/cl + \
        5.*ck**3*(1.+14.*ak) + \
        ck**5*(-1.-62.*ak+al+62*ak*al)/(3.*cl*z)

def f14(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    1/2cl < z <= 1/2(cl-ck)
    """
    return z**5*16.*(2.*ak*al-ak)/(3.*ck*cl) + \
        z**4*8*(ak-2.*ak*al)/ck + z**4*16.*(2*ak*al-ak)/cl + \
        z**3*(20.*ak-40.*ak*al) + \
        z**2*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) +\
        z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(-1.-6.*ak+al-2.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
        5.*ck**3*(1.+6.*ak+16.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/(2.*ck) + \
        12.*ck**4*(2.*ak*al - ak)/z + ck**5*(-1.-30.*ak+al-2.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6*ck*z)

def f15(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    1/2(cl-ck) < z <= 1/2cl
    """
    return z**5*16.*(ak-1.+2.*al-2.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
        z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)/(2.*cl) + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)/(2.*ck) + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) +5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

def f16(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    1/2cl < z <= 1/2(cl+ck)
    """ 
    return z**5*16.*(1.-ak-2.*al+2.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+2.*al-2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + \
        z**3.*(10.-20.*al) + \
        z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al+2.*al*ak)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(ak-1.+2.*al-2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(1.-ak-2.*al+2.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(ak-1.+2.*al-2.*ak*al)/(6.*ck*z)

def f17(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    1/2(ck+cl) < z <= 1/2cl + ck
    """
    return z**5*16.*(ak-2.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(2.*ak*al-ak)/ck + z**4*16.*(2.*ak*al-ak)/cl +\
        z**3*(20.*ak-40.*ak*al) + \
        z**2.*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-2.*ak*al)/(3.*ck) + \
        z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(2.*ak*al-al-8.*ak)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(2.*ak*al-ak)/ck + \
        10.*ck**3*(4.*ak+al+6.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(ak-2.*ak*al)*0.5/ck + \
        12.*ck**4*(2.*ak*al-ak)/z + ck**5*(2.*ak*al-al-32.*ak)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(2.*ak*al-ak)/(6.*ck*z)

def f18(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    1/2cl + ck < z <= cl-ck
    """
    return 10.*ck**3*(al+14.*ak*al) - z*10.*ck**3*(al+14.*ak*al)/cl - ck**5*(al+62.*ak*al)/(3.*z*cl)

def f19(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    cl-ck < z <= cl-1/2ck
    """
    return -16.*z**5*ak*al/(3.*ck*cl) + \
        z**4*16.*ak*al/ck - z**4*16.*ak*al/cl + \
        z**3*40.*ak*al + \
        z**2*160.*ck**2*ak*al/(3.*cl) - z**2*160.*cl**2*ak*al/(3.*ck) + \
        z*80.*cl**3*ak*al/ck - z*80.*ck**2*ak*al - z*10.*ck**3*(al+6.*ak*al)/cl - z*80.*cl**2*ak*al + \
        10.*ck**3*(al+6.*ak*al) + 48.*ck**4*ak*al/cl + 80.*cl**3*ak*al - 48.*cl**4*ak*al/ck + \
        80.*ck**2*cl**2*ak*al/(3.*z) - 24.*ck**4*ak*al/z - ck**5*(al+30.*ak*al)/(3.*cl*z) - 24.*cl**4*ak*al/z + 32.*cl**5*ak*al/(3.*ck*z)

def f110(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    cl-1/2ck < z <= cl
    """
    return z**5*16.*(ak*al-al)/(3.*ck*cl) + \
        z**4*16.*(al-ak*al)/ck - z**4*8.*al/cl + \
        z**3*20.*al + \
        z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(ak*al-al)/(3.*ck) + \
        z*80.*cl**3*(al-ak*al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
        5.*ck**3*(14.*ak*al+al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(ak*al-al)/ck + \
        10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(al-ak*al)/(3.*ck*z)

def f111(z,ak,al,ck,cl):
    """
    0 < ck <= cl/4
    cl < z <= cl+1/2ck
    """
    return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
        z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + \
        z**3*20.*al + \
        z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
        z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
        5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(al-ak*al)/ck + \
        10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)


def f112(z,ak,al,ck,cl):
    """
    0 < ck < cl/4
    cl + 1/2ck < z <= cl+ck
    """
    return ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - z**4*16./cl + 40.*z**3 + z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - z*80.*ck**2 - z*80.*ck**3/cl - z*80.*cl**2 - z*80.*cl**3/ck + 80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) + 80.*ck**2*cl**2/(3.*z) - 24.*cl**4/z - 32*cl**5/(3.*ck*z))

# # ------------------------f2(z) where cl/4 < ck <= cl/3 ----------------- # # 
def f21(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    0 <= z <= ck/2
    """
    return z**5*32.*(ak+al-ak*al-1.)/(3.*ck*cl) + z**4*16.*(1.-al)/cl + \
        z**2*40.*ck**2*(al-6.*ak-1.+6.*ak*al)/(3.*cl) + 5.*ck**3*(1.+14.*ak) + \
        3.*ck**4*(al-1.-30.*ak+30.*ak*al)/cl

def f22(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    ck/2 < z <= cl/2-ck
    """
    return z**5*32.*(ak*al-ak)/(3.*ck*cl) + z**4*32.*(ak-ak*al)/cl + \
        z**2*320.*ck**2*(ak*al-ak)/(3.*cl) +  z*10.*ck**3*(2.*ak-1.+al-2.*ak*al)/cl + \
        5.*ck**3*(1.+14.*ak) + 96.*ck**4*(ak*al-ak)/cl + \
        ck**5*(2.*ak-1.+al-2.*ak*al)/(3.*cl*z)

def f23(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    cl/2-ck < z <= ck
    """
    return z**5*16.*(4.*ak*al-3.*ak)/(3.*ck*cl) + \
        z**4*8.*(ak-2.*ak*al)/ck + z**4*16.*ak/cl + z**3*(20.*ak-40.*ak*al) + \
        z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) - z**2*160.*ck**2*ak/(3.*cl) + \
        z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(10.*ak-1.+al-18.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
        5.*ck**3*(1.+6.*ak+16.*ak*al) - 48.*ck**4*ak/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
        12.*ck**4*(2.*ak*al-ak)/z + ck**5*(34.*ak-1.+al-66.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

def f24(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    ck < z <= 1/2(cl-ck)
    """
    return z**5*16*(2.*ak*al-ak)/(3.*ck*cl) + \
        z**4*8.*(ak-2.*ak*al)/ck + z**4*16*(2.*ak*al-ak)/cl + z**3*(20.*ak-40.*ak*al) + \
        z**2*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
        z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(al-1.-6.*ak-2.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
        5.*ck**3*(1.+6.*ak+16.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
        12.*ck**4*(2.*ak*al-ak)/z + ck**5*(al-1.-30.*ak-2.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

def f25(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    1/2(cl-ck) < z <= cl/2
    """
    return z**5*16.*(ak-1.+2.*al-2.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
        z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

def f26(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    cl/2 < z <= 1/2(cl+ck)
    """
    return z**5*16.*(1.-ak-2.*al+2.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+2.*al-2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
        z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al+2.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(ak-1.+2.*al-2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(1.-ak-2.*al+2.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(ak-1.+2.*al-2.*ak*al)/(6.*ck*z)

def f27(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    1/2(ck+cl) < z <= cl-ck
    """
    return z**5*16.*(ak-2.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(2.*ak*al-ak)/ck + z**4*16.*(2.*ak*al-ak)/cl + z**3*(20.*ak-40.*ak*al) + \
        z**2*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-2.*ak*al)/(3.*ck) + \
        z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(2.*ak*al-8.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(2.*ak*al-ak)/ck + \
        10.*ck**3*(4.*ak+al+6.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(ak-2.*ak*al)*0.5/ck + \
        12.*ck**4*(2.*ak*al-ak)/z + ck**5*(2.*ak*al-al-32.*ak)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(2.*ak*al-ak)/(6.*ck*z)

def f28(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    cl-ck < z <= ck+cl/2
    """
    return z**5*16.*(ak-3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(4.*ak*al-ak)/ck + z**4*16.*(ak*al-ak)/cl + z**3*20.*ak + \
        z**2*160.*ck**2*(ak-ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-10.*ak*al)/(3.*ck) + \
        z*10.*ck**3*(10.*ak*al-al-8.*ak)/cl - z*40.*ck**2*ak - z*10.*cl**2*(ak+6.*ak*al) + z*5.*cl**3*(18.*ak*al-ak)/ck + \
        10.*ck**3*(4.*ak+al-2.*ak*al) + 48.*ck**4*(ak-ak*al)/cl + 5.*cl**3*(ak+14.*ak*al) + 3.*cl**4*(ak-34.*ak*al)*0.5/ck + \
        ck**5*(34.*ak*al-al-32.*ak)/(3.*cl*z) -12.*ck**4*ak/z + 10.*ck**2*cl**2*(ak+6.*ak*al)/(3.*z) - 3.*cl**4*(ak+30.*ak*al)/(4.*z) + cl**5*(66.*ak*al-ak)/(6.*ck*z)

def f29(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    ck + cl/2 < z <= cl-ck/2
    """
    return -z**5*16.*ak*al/(3.*ck*cl) + \
        z**4*16.*ak*al/ck - z**4*16.*ak*al/cl + z**3*40.*ak*al + \
        z**2*160.*ck**2*ak*al/(3.*cl) - z**2*160.*cl**2*ak*al/(3.*ck) + \
        z*80.*cl**3*ak*al/ck - z*80.*ck**2*ak*al - z*10.*ck**3*(al+6.*ak*al)/cl - z*80.*cl**2*ak*al + \
        10.*ck**3*(al+6.*ak*al) + 48.*ck**4*ak*al/cl + 80.*cl**3*ak*al - 48.*cl**4*ak*al/ck + \
        80.*ck**2*cl**2*ak*al/(3.*z) - 24.*ck**4*ak*al/z - ck**5*(al+30.*ak*al)/(3.*cl*z) -24.*cl**4*ak*al/z + 32.*cl**5*ak*al/(3.*ck*z)

def f210(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    cl - ck/2 < z <= cl
    """
    return z**5*16.*(ak*al-al)/(3.*ck*cl) + \
        z**4*16.*(al-ak*al)/ck - z**4*8.*al/cl + \
        z**3*20.*al + \
        z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(ak*al-al)/(3.*ck) + \
        z*80.*cl**3*(al-ak*al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
        5.*ck**3*(14.*ak*al+al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(ak*al-al)/ck + \
        10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(al-ak*al)/(3.*ck*z)

def f211(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    cl < z <= cl + ck/2
    """
    return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
        z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + \
        z**3*20.*al + \
        z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
        z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
        5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(al-ak*al)/ck + \
        10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)


def f212(z,ak,al,ck,cl):
    """
    cl/4 < ck <= cl/3
    cl+ck/2 < z <= cl+ck
    """
    return ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - z**4*16./cl + 40.*z**3 + z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - z*80.*ck**2 - z*80.*ck**3/cl - z*80.*cl**2 - z*80.*cl**3/ck + 80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) + 80.*ck**2*cl**2/(3.*z) - 24.*cl**4/z - 32*cl**5/(3.*ck*z))

# # --------------------- f3(z) where cl/3 < ck <= cl/2 ----------------- # #
def f31(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    0 <= z <= cl/2-ck
    """
    return z**5*32.*(ak-1.-ak*al+al)/(3.*ck*cl) + z**4*16.*(1.-al)/cl + \
        z**2*40.*ck**2*(al-1.+6.*ak*al-6.*ak)/(3.*cl) + 5.*ck**3*(1.+14.*ak) + \
        3.*ck**4*(al-1.-30.*ak+30.*ak*al)/cl

def f32(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    cl/2-ck < z <= ck/2
    """
    return z**5*16.*(ak+2.*al-2.)/(3.*ck*cl) + \
        z**4*8.*(ak-2.*ak*al)/ck + z**4*16*(1.-ak-al+2.*ak*al)/cl + z**3*(20.*ak-40.*ak*al) + \
        z**2*40.*ck**2*(al-2.*ak-1.-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
        z*40.*ck**2*(2.*ak*al-ak) + z*80.*ck**3*(ak-2.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
        5.*ck**3*(1.+6.*ak+16.*ak*al) + 3.*ck**4*(al-1.-14.*ak-2.*ak*al)/cl  + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
        12.*ck**4*(2.*ak*al-ak)/z + 32.*ck**5*(ak-2.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

def f33(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    ck/2 < z <= 1/2(cl-ck)
    """
    return z**5*16*(4.*ak*al-3.*ak)/(3.*ck*cl) + \
        z**4*8.*(ak-2.*ak*al)/ck + z**4*16.*ak/cl + z**3*20.*ak*(1.-2.*al) + \
        z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) - z**2*160.*ck**2*ak/(3.*cl) + \
        z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(10.*ak-1+al-18.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
        5.*ck**3*(1.+6.*ak+16.*ak*al) - 48.*ck**4*ak/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
        12.*ck**4*(2.*ak*al-ak)/z + ck**5*(34.*ak-1+al-66.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

def f34(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    1/2(cl-ck) < z <= ck
    """
    return z**5*16.*(2.*al-ak-1.)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(4.*ak-1+2.*al-4.*ak*al)/cl + \
        z**3*10.*(1.-2.*al) + \
        z**2*20.*ck**2*(1.-10.*ak-2.*al+4.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + z*5.*ck**3*(18.*ak-1.-32.*ak*al)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+4.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-128.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

def f35(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    ck < z <= cl/2
    """
    return z**5*16.*(ak-1.+2.*al-2.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
        z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

def f36(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    cl/2 < z <= cl-ck
    """
    return z**5*16.*(1.-ak-2.*al+2.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+2.*al-2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
        z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al+2.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(ak-1.+2.*al-2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(1.-ak-2.*al+2.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(ak-1.+2.*al-2.*ak*al)/(6.*ck*z)

def f37(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    cl-ck < z <= 1/2(cl+ck)
    """
    return z**5*16.*(1.-ak-2.*al+ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(2.*al-1.-2.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(1.+6.*ak-2.*al-4.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(16.*ak*al-1.-14.*ak)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-28.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(64.*ak*al-1.-62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

def f38(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    1/2(ck+cl) < z <= cl-ck/2
    """
    return z**5*16.*(ak-3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(4.*ak*al-ak)/ck + z**4*16.*(ak*al-ak)/cl + z**3*20.*ak + \
        z**2*160.*ck**2*(ak-ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-10.*ak*al)/(3.*ck) + \
        z*10.*ck**3*(10.*ak*al-al-8.*ak)/cl - z*40.*ck**2*ak - z*10.*cl**2*(ak+6.*ak*al) + z*5.*cl**3*(18.*ak*al-ak)/ck + \
        10.*ck**3*(al+4.*ak-2.*ak*al) + 48.*ck**4*(ak-ak*al)/cl + 5.*cl**3*(ak+14.*ak*al) + 3.*cl**4*(ak-34.*ak*al)*0.5/ck + \
        ck**5*(34.*ak*al-32.*ak-al)/(3.*cl*z) - 12.*ck**4*ak/z + 10.*ck**2*cl**2*(ak+6.*ak*al)/(3.*z) - 3.*cl**4*(ak+30.*ak*al)/(4.*z) + cl**5*(66.*ak*al-ak)/(6.*ck*z)

def f39(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    cl - ck/2 < z <= ck + cl/2
    """
    return z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + \
        z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + z**3*20.*(ak+al-2.*ak*al) + \
        z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck)+ \
        z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + \
        5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*ak*al-16.*ak-al)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)

def f310(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    ck+cl/2 < z <= cl
    """
    return z**5*16.*(ak*al-al)/(3.*ck*cl) + \
        z**4*16.*(al-ak*al)/ck - z**4*8.*al/cl + z**3*20.*al + \
        z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(ak*al-al)/(3.*ck) + \
        z*80.*cl**3*(al-ak*al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
        5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(ak*al-al)/ck + \
        10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z)- 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(al-ak*al)/(3.*ck*z)

def f311(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    cl < z <= cl+ck/2
    """
    return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
        z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + \
        z**3*20.*al + \
        z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
        z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
        5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(al-ak*al)/ck + \
        10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)

def f312(z,ak,al,ck,cl):
    """
    cl/3 < ck <= cl/2
    cl+ck/2 < z <= cl+ck
    """
    return ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - z**4*16./cl + 40.*z**3 + z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - z*80.*ck**2 - z*80.*ck**3/cl - z*80.*cl**2 - z*80.*cl**3/ck + 80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) + 80.*ck**2*cl**2/(3.*z) - 24.*cl**4/z - 32*cl**5/(3.*ck*z))

## ---------------- f4, case where cl/2 <= ck <= 2cl/3 ----------------  ##
def f41(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    0 <= z <= ck-cl/2
    """
    return z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + \
        z**4*16.*(ak-2.*ak*al)/ck + z**4*16.*(1.-2.*ak-al+4.*ak*al)/cl + \
        z**2*40.*ck**2*(2.*ak-1.+al-10.*ak*al)/(3.*cl) + z**2*40.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
        5.*ck**3*(1.-2.*ak+32.*ak*al) + 3.*ck**4*(2.*ak-1.+al-34.*ak*al)/cl + 10.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/ck

def f42(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    ck-cl/2 < z <= 1/2(cl-ck)
    """
    return z**5*16.*(ak-2.+2.*al)/(3.*ck*cl) + \
        z**4*8.*(ak-2.*ak*al)/ck + z**4*16.*(1.-ak-al+2.*ak*al)/cl + z**3*(20.*ak-40.*ak*al) + \
        z**2*40.*ck**2*(al-1.-2.*ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
        z*40.*ck**2*(2.*ak*al-ak) + z*80.*ck**3*(ak-2.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
        5.*ck**3*(1.+6.*ak+16.*ak*al) + 3.*ck**4*(al-1.-14.*ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
        12.*ck**4*(2.*ak*al-ak)/z + 32.*ck**5*(ak-2.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

def f43(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    1/2(cl-ck) < z <= ck/2
    """
    return z**5*16.*(3.*ak-3.+4.*al-4.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8./cl + z**3*(10.-20.*al) + \
        z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) - z**2*20.*ck**2*(1.+6.*ak)/(3.*cl) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + z*5.*ck**3*(1.+14.*ak-2.*al-28.*ak*al)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 - 3.*ck**4*(1.+30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + ck**5*(1.+62.*ak-2.*al-124.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

def f44(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    ck/2 < z <= cl-ck
    """
    return z**5*16.*(2.*al-1.-ak)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(4.*ak-1.+2.*al-4.*ak*al)/cl + z**3*(10.-20.*al) + \
        z**2*20.*ck**2*(1.-10.*ak-2.*al+4.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + z*5.*ck**3*(18.*ak-1.-32.*ak*al)/cl + z*5.*cl**2*(2.*al-1) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+4.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-128.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

def f45(z,ak,al,ck,cl):
    """
    cl/2 <= cj <= 2cl/3
    cl-ck < z <= cl/2
    """
    return z**5*16.*(2.*al-1.-ak-ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + \
        z**3*10.*(1.-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

def f46(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    cl/2 < z <= ck
    """
    return z**5*16.*(1.-3.*ak-2.*al+3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + \
        z**3*10.*(1.-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

def f47(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    ck < z <= cl-ck/2
    """
    return z**5*16.*(1.-ak-2.*al+ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(2.*al-1.-2.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(1.+6.*ak-2.*al-4.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(16.*ak*al-1.-14.*ak)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-28.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(64.*ak*al-1.-62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

def f48(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    cl=-ck/2 < z <= 1/2(ck+cl)
    """
    return z**5*16.*(1.-ak-3.*al+3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(al-1.)/cl + z**3*10. + \
        z**2*20.*ck**2*(1.+6.*ak-al-6.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
        z*5.*ck**3*(al-1.-14.*ak+14.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.+30.*ak-al-30.*ak*al)*0.5/cl + 5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
        ck**5*(al-1.-62.*ak+62.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

def f49(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    1/2(ck+cl) < z <= cl
    """
    return z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + \
        z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-al-2.*ak)/cl + z**3*20.*(ak+al-2.*ak*al) + \
        z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck) + \
        z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + \
        5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + \
        3.*ck**4*(-16.*ak-al+2.*ak*al)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)

def f410(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    cl < z <= ck+cl/2
    """
    return z**5*16.*(ak+al-3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(4.*ak*al-ak-2.*al)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + z**3*20.*(ak+al-2.*ak*al) +\
        z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak+8.*al-10.*ak*al)/(3.*ck) + \
        z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(18.*ak*al-ak-16.*al)/ck + \
        5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak+32.*al-34.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*ak*al-al-16.*ak)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-16.*al-ak)/(4.*z) + cl**5*(66.*ak*al-ak-64.*al)/(6.*ck*z)

def f411(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    ck + cl/2 < z <= cl + ck/2
    """
    return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
        z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + z**3*20.*al + \
        z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
        z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
        5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al +48.*cl**4*(al-ak*al)/ck + \
        10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)

def f412(z,ak,al,ck,cl):
    """
    cl/2 < ck <= 2cl/3
    cl + ck/2 < z <= cl+ck
    """
    return ak*al*(z**5*16./(3.*ck*cl) - z**4*16/ck - z**4*16./cl + z**3*40.+ \
                  z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - \
                  z*80.*(ck**2 + ck**3/cl + cl**2 + cl**3/ck) + \
                  80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck + \
                  80.*ck**2*cl**2/(3.*z) - 24.*ck**4/z - 32.*ck**5/(3.*cl*z)\
                  - 24.*cl**4/z - 32.*cl**5/(3.*ck*z))


## ---------------- f5, where 2cl/3 <= ck <= 3cl/4 ---------------- ##
def f51(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    0 <= z <= 1/2(cl-ck)
    """
    return z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + \
        z**4*16.*(ak-2.*ak*al)/ck + z**4*16.*(1.-2.*ak-al+4.*ak*al)/cl + \
        z**2*40.*ck**2*(2.*ak-1.+al-10.*ak*al)/(3.*cl) + z**2*40.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
        5.*ck**3*(1.-2.*ak+32.*ak*al) + 3.*ck**4*(al+2.*ak-1.-34.*ak*al)/cl + 10.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/ck

def f52(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    1/2(cl-ck) < z <= ck=cl/2
    """
    return z**5*16.*(4.*ak-3.+4.*al-6.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-2.*al)/ck + z**4*8.*(1.-2.*ak+4.*ak*al)/cl + z**3*10.*(1.-2.*ak-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(2.*ak-1.-16.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*al-1.)/(3.*ck) + \
        z*5.*(1.-2.*ak)*(1.-2.*al)*(ck**3/cl-ck**2-cl**2+cl**3/ck) + \
        5.*ck**3*(1.-2.*ak+2.*al+60.*ak*al)*0.5 + 3.*ck**4*(2.*ak-1.-64.*ak*al)*0.5/cl + 5.*cl**3*(1.+2.*ak-2.*al-4.*ak*al)*0.5 + 3.*cl**4*(2.*al-1.)*0.5/ck + \
        (1.-2.*ak)*(1.-2.*al)*(ck**5/(6.*cl) - 3.*ck**4/8. + 5.*ck**2*cl**2/12. - 3.*cl**4/8. + cl**5/(6.*ck))/z

def f53(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    ck - cl/2 < z <= cl-ck
    """
    return z**5*16.*(3.*ak-3.+4.*al-4.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8./cl + z**3*10.*(1.-2.*al) + \
        z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) - z**2*20.*ck**2*(1.+6.*ak)/(3.*cl) + \
        z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + z*5.*ck**3*(1.+14.*ak-2.*al-28.*ak*al)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 - 3.*ck**4*(1.+30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + ck**5*(1.+62.*ak-2.*al-124.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

def f54(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    cl-ck < z <= ck/2
    """
    return z**5*16.*(3.*ak-3.+4.*al-5.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(1.-2.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(8.*ak*al-1.-6.*ak)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(1.+14.*ak-2.*al-12.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(32.*ak*al-1.-30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(1.+62.*ak-2.*al-60.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

def f55(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    ck/2 < z <= cl/2
    """
    return z**5*16.*(2.*al-1.-ak-ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

def f56(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    cl/2 < z <= cl-ck/2
    """
    return z**5*16.*(1.-3.*ak-2.*al+3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

def f57(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    cl-ck/2 < z <= ck
    """
    return z**5*16.*(1.-3.*ak-3.*al+5.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(4.*ak-1.+al-4.*ak*al)/cl + z**3*10. + \
        z**2*20.*ck**2*(1.-10.*ak-al+10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
        z*5.*ck**3*(18.*ak-1.+al-18.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.-34.*ak-al+34.*ak*al)*0.5/cl + 5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
        ck**5*(66.*ak-1.+al-66.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

def f58(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    ck < z <= 1/2(cl+ck)
    """
    return z**5*16.*(1.-ak-3.*al+3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(al-1.)/cl + z**3*10. + \
        z**2*20.*ck**2*(1.+6.*ak-al-6.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
        z*5.*ck**3*(al-1.-14.*ak+14.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.+30.*ak-al-30.*ak*al)*0.5/cl + 5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
        ck**5*(al-1.-62.*ak+62.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

def f59(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    1/2(ck+cl) < z <= cl
    """
    return z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + \
        z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-al-2.*ak)/cl + z**3*20.*(ak+al-2.*ak*al) + \
        z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck) + \
        z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + \
        5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + \
        3.*ck**4*(-16.*ak-al+2.*ak*al)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)


def f510(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    cl < z <= ck+cl/2
    """
    return z**5*16.*(ak+al-3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(4.*ak*al-ak-2.*al)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + z**3*20.*(ak+al-2.*ak*al) +\
        z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak+8.*al-10.*ak*al)/(3.*ck) + \
        z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(18.*ak*al-ak-16.*al)/ck + \
        5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak+32.*al-34.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*ak*al-al-16.*ak)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-16.*al-ak)/(4.*z) + cl**5*(66.*ak*al-ak-64.*al)/(6.*ck*z)

def f511(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    ck+cl/2 < z <= cl + ck/2
    """
    return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
        z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + z**3*20.*al + \
        z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
        z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
        5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al +48.*cl**4*(al-ak*al)/ck + \
        10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)

def f512(z,ak,al,ck,cl):
    """
    2cl/3 < ck <= 3cl/4
    cl+ck/2 < z <= cl+ck
    """
    return ak*al*(z**5*16./(3.*ck*cl) - z**4*16/ck - z**4*16./cl + z**3*40.+ \
                  z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - \
                  z*80.*(ck**2 + ck**3/cl + cl**2 + cl**3/ck) + \
                  80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck + \
                  80.*ck**2*cl**2/(3.*z) - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) - \
                  24.*cl**4/z - 32.*cl**5/(3.*ck*z))

## ---------------- f6, where 3cl/4 <= ck <= cl---------------- ##

def f61(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    0 <= z <= 1/2(cl-ck)
    """
    return z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + \
        z**4*16.*(ak-2.*ak*al)/ck + z**4*16.*(1.-2.*ak-al+4.*ak*al)/cl + \
        z**2*40.*ck**2*(2.*ak-1+al-10.*ak*al)/(3.*cl) + z**2*40.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
        5.*ck**3*(1.-2.*ak+32.*ak*al) + 3.*ck**4*(2.*ak-1.+al-34.*ak*al)/cl + 10.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/ck 

def f62(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    1/2(cl-ck) < z <= cl-ck
    """
    return z**5*16.*(4.*ak-3.+4.*al-6.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-2.*al)/ck + z**4*8.*(1.-2.*ak+4.*ak*al)/cl + z**3*10.*(1.-2.*al-2.*ak+4.*ak*al) + \
        z**2*20.*ck**2*(2.*ak-1.-16.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*al-1.)/(3.*ck) + \
        z*5.*(2.*ak-1)*(1.-2.*al)*(ck**2 - ck**3/cl + cl**2 - cl**3/ck) + \
        5.*ck**3*(1.-2.*ak+2.*al+60.*ak*al)*0.5 + 3.*ck**4*(2.*ak-1.-64.*ak*al)*0.5/cl + 5.*cl**3*(1.+2.*ak-2.*al-4.*ak*al)*0.5 + 3.*cl**4*(2.*al-1.)*0.5/ck + \
        (2.*ak-1.)*(1.-2.*al)*(3.*ck**4/8. - ck**5/(6.*cl) - 5.*ck**2*cl**2/12. + 3.*cl**4/8. - cl**5/(6.*ck))/z

def f63(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    cl-ck < z <= ck-cl/2
    """
    return z**5*16.*(4.*ak-3.+4.*al-7.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-2.*al+2.*ak*al)/ck + z**4*8.*(1.-2.*ak+2.*ak*al)/cl + z**3*10.*(1.-2.*ak-2.*al+8.*ak*al) + \
        z**2*20.*ck**2*(2.*ak-1.-8.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*al-1.-8.*ak*al)/(3.*ck) + \
        z*5.*(2.*ak-1.+2.*al-20.*ak*al)*(ck**2 - ck**3/cl + cl**2 - cl**3/ck) + \
        5.*ck**3*(1.-2.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(2.*ak-1.-32.*ak*al)*0.5/cl + 5.*cl**3*(1.+2.*ak-2.*al+28.*ak*al)*0.5 + 3.*cl**4*(2.*al-1.-32.*ak*al)*0.5/ck + \
        (2.*ak-1.+2.*al-68.*ak*al)*(3.*ck**4/8. - ck**5/(6.*cl) - 5.*ck**2*cl**2/12. + 3.*cl**4/8. - cl**5/(6.*ck))/z

def f64(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    ck -cl/2 < z <= ck/2
    """
    return z**5*16.*(3.*ak-3.+4.*al-5.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(1.-2.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(8.*ak*al-1.-6.*ak)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(1.+14.*ak-2.*al-12.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(32.*ak*al-1.-30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(1.+62.*ak-2.*al-60.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

def f65(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    ck/2 < z <= cl/2
    """
    return z**5*16.*(2.*al-1.-ak-ak*al)/(3.*ck*cl) + \
        z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

def f66(z,ak,al,ck,cl):
    """
    3cl/4 <= ck <= cl
    cl/2 < z <= cl-ck/2
    """
    return z**5*16.*(1.-3.*ak-2.*al+3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
        z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

def f67(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    cl-ck/2 < z <= ck
    """
    return z**5*16.*(1.-3.*ak-3.*al+5.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(4.*ak-1.+al-4.*ak*al)/cl + z**3*10. + \
        z**2*20.*ck**2*(1.-10.*ak-al+10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
        z*5.*ck**3*(18.*ak-1.+al-18.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.-34.*ak-al+34.*ak*al)*0.5/cl + 5.*cl**3*(1+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
        ck**5*(66.*ak-1.+al-66.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

def f68(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    ck < z <= 1/2(ck+cl)
    """
    return z**5*16.*(1.-ak-3.*al+3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(al-1.)/cl + z**3*10. + \
        z**2*20.*ck**2*(1.+6.*ak-al-6.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
        z*5.*ck**3*(al-1.-14.*ak+14.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
        5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.+30.*ak-al-30.*ak*al)*0.5/cl + 5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
        ck**5*(al-1.-62.*ak+62.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

def f69(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    1/2(ck+cl) < z <= cl
    = f49(z)
    """
    return z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + \
        z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-al-2.*ak)/cl + z**3*20.*(ak+al-2.*ak*al) + \
        z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck) + \
        z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + \
        5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + \
        3.*ck**4*(-16.*ak-al+2.*ak*al)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)

def f610(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    cl < z <= ck + cl/2
    = f410(z)
    """
    return z**5*16.*(ak+al-3.*ak*al)/(3.*ck*cl) + \
        z**4*8.*(4.*ak*al-ak-2.*al)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + z**3*20.*(ak+al-2.*ak*al) +\
        z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak+8.*al-10.*ak*al)/(3.*ck) + \
        z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(18.*ak*al-ak-16.*al)/ck + \
        5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak+32.*al-34.*ak*al)*0.5/ck + \
        3.*ck**4*(2.*ak*al-al-16.*ak)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-16.*al-ak)/(4.*z) + cl**5*(66.*ak*al-ak-64.*al)/(6.*ck*z)

def f611(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    ck + cl/2 < z <= cl + ck/2
    = f411(z)
    """
    return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
        z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + z**3*20.*al + \
        z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
        z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
        5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al +48.*cl**4*(al-ak*al)/ck + \
        10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)

def f612(z,ak,al,ck,cl):
    """
    3cl/4 < ck <= cl
    cl+ck/2 < z <= cl+ck
    """
    return ak*al*(z**5*16./(3.*ck*cl) - z**4*16/ck - z**4*16./cl + z**3*40.+ \
                  z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - \
                  z*80.*(ck**2 + ck**3/cl + cl**2 + cl**3/ck) + \
                  80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck + \
                  80.*ck**2*cl**2/(3.*z) - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) - \
                  24.*cl**4/z - 32.*cl**5/(3.*ck*z))
    
## ---------------- f7, case where ck = cl---------------- ##

def f71(z,ak,al,ck,cl):
    """
    ck = cl = c
    0 <= z <= 0.5c
    """
    return (z/ck)**5*16.*(4.*ak-3.-7.*ak*al+4.*al)/3. + \
        (z/ck)**4*16.*(1.+2.*ak*al-ak-al) + \
        (z/ck)**3*10.*(1.+8.*ak*al-2.*ak-2.*al) - \
        (z/ck)**2*40.*(1.+8.*ak*al-ak-al)/3. + \
        2.+44.*ak*al+3.*ak+3.*al

def f72(z,ak,al,ck,cl):
    """
    ck = cl
    1/2ck < z <= ck
    """
    return (z/ck)**5*16.*(1.+5.*ak*al-3.*ak-3.*al)/3. + \
        (z/ck)**4*8.*(5.*ak-2.-8.*ak*al+5.*al) + (z/ck)**3*10. + \
        (z/ck)**2*20.*(2.+20.*ak*al-11.*ak-11.*al)/3. + \
        (z/ck)*5.*(13.*ak-4.-36.*ak*al+13.*al) + \
        (16.+204.*ak*al-35.*ak-35.*al)/2. + \
        (29.*ak-8.-84.*ak*al+29.*al)/(12.*(z/ck))

def f73(z,ak,al,ck,cl):
    """
    ck = cl
    ck < z <= 3/2ck
    """
    return (z/ck)**5*16.*(ak-3.*ak*al+al)/3. + \
        (z/ck)**4*8.*(8.*ak*al-3.*ak-3.*al) + \
        (z/ck)**3*20.*(ak-2.*ak*al+al) + \
        (z/ck)**2*20.*(9.*ak-20.*ak*al+9.*al)/3. + \
        (z/ck)*5.*(44.*ak*al-27.*ak-27.*al) + \
        0.5*(189.*ak-244.*ak*al+189.*al) + \
        (460.*ak*al-243.*ak-243.*al)/(12.*(z/ck))

def f74(z,ak,al,ck,cl):
    """
    ck = cl
    3/2ck < z <= 2ck
    """
    return 8.*ak*al*((z/ck)**5*2. - (z/ck)**4*12. + (z/ck)**3*15. + (z/ck)**2*40. - (z/ck)*120. + 96. - 16./(z/ck))/3.


def nfcn(aa):
    """
    computes the function
    n(ak) = 2.+6.*ak+44.*ak^2
    """
    return 2.+6.*aa+44.*aa**2
    
class GenGC:
    def __init__(self,a1,a2,c1,c2):
        """
        Initializes Generalized Gaspari Cohn correlation function.
        Input:
        z - (nd array) distance between two points 
            (after norm is computed), where
            z = ||x1 - x2|| where x1, x2 are two spatial points
            associated with parameters a1, c1 and a2, c2.
            This can be an array or a scalar.
        a1, a2 - (scalar) parameters a associated with points x1, x2
        c1, c2 - (scalar) cut-off parameters associated with points x1, x2
        **NOTE** the GenGC convlution is constructed as
        B_{kl}(z;ak,al,ck,cl)
        where ck<=cl. If ck>=cl, the two sets of a, c need to be swapped.
        This is taken care of here.
        """
        # self.z = z
        if c1 <= c2:
            self.ck = c1
            self.cl = c2
            self.ak = a1
            self.al = a2
        else:                   # here, the indices need to be swapped because ck >=cl
            self.ck = c2
            self.cl = c1
            self.ak = a2
            self.al = a1

    def __call__(self,z):
        """
        Evaluates GenGC based on input z (scalar or array of normed 
        distances between points)
        can redo this using functional programing, partial function 
        for example rather than if statements
        """
        # # for some reason having problem with piecewise function here
        # # have old if statement instead
        if self.ck <=0.25*self.cl:
            return self.f1(z)
        elif (self.ck>0.25*self.cl and self.ck<=self.cl/3):
            return self.f2(z)
        elif (self.ck>self.cl/3 and self.ck <=0.5*self.cl):
            return self.f3(z)
        elif (self.ck>0.5*self.cl and self.ck <= 2.*self.cl/3.):
            return self.f4(z)
        elif (self.ck>2.*self.cl/3. and self.ck <= 0.75*self.cl):
            return self.f5(z)
        elif (self.ck>0.75*self.cl and self.ck < self.cl):
            return self.f6(z)
        elif self.ck==self.cl:
            return self.f7(z)
        else:
            return 0.

    def make_fi_masks(self):
        """
        Returns a list of masks and associated functions based 
        on the input values of c1 and c2, if needed.
        """
        fi_masks = [self.ck <=0.25*self.cl,
                    np.logical_and(self.ck>0.25*self.cl,
                                   self.ck<=self.cl/3),
                    np.logical_and(self.ck>self.cl/3,
                                   self.ck <=0.5*self.cl),
                    np.logical_and(self.ck>0.5*self.cl,
                                   self.ck <= 2.*self.cl/3.),
                    np.logical_and(self.ck>2.*self.cl/3.,
                                   self.ck <= 0.75*self.cl),
                    np.logical_and(self.ck>0.75*self.cl,
                                   self.ck < self.cl),
                    self.ck==self.cl,
                    self.ck > self.cl]
        fcn_list = [self.f1,self.f2,self.f3,self.f4,
                    self.f5,self.f6,self.f4,zero]

        return fi_masks, fcn_list


    def determine_f(self):
        """
        Outputs correct function f1, ..., f7 based on
        input ck and cl
        in the context of the dictonary
        """
        if self.ck <=0.25*self.cl:
            return self.f1
        elif (self.ck>0.25*self.cl and self.ck<=self.cl/3):
            return self.f2
        elif (self.ck>self.cl/3 and self.ck <=0.5*self.cl):
            return self.f3
        elif (self.ck>0.5*self.cl and self.ck <= 2.*self.cl/3.):
            return self.f4
        elif (self.ck>2.*self.cl/3. and self.ck <= 0.75*self.cl):
            return self.f5
        elif (self.ck>0.75*self.cl and self.ck < self.cl):
            return self.f6
        elif self.ck==self.cl:
            return self.f7
        else:
            return zero
    

    def normalize(self):
        """
        computes normalization factor common to functions f1,f2,...,f6
        """
        return np.sqrt(nfcn(self.ak)*nfcn(self.al)*self.ck**3*self.cl**3)
        
    def f1(self,z):
        """
        Case where 0 < ck <= cl/4 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize()
        return np.piecewise(z,
                            [(z >= 0.)*(z <= 0.5*self.ck),
                             (z > 0.5*self.ck)*(z <= self.ck),
                             (z > self.ck)*(z <= 0.5*self.cl - self.ck),
                             (z > 0.5*self.cl - self.ck)*(
                                 z <= 0.5*self.cl - 0.5*self.ck),
                             (z > 0.5*self.cl-0.5*self.ck)*(z <= 0.5*self.cl),
                             (z > 0.5*self.cl)*(z <= 0.5*(self.cl+self.ck)),
                             (z > 0.5*(self.ck+self.cl))*(z <= 0.5*self.cl+self.ck),
                             (z > 0.5*self.cl+self.ck)*(z <= self.cl-self.ck),
                             (z > self.cl-self.ck)*(z <= self.cl -0.5*self.ck),
                             (z > self.cl-0.5*self.ck)*(z <= self.cl),
                             (z > self.cl)*(z <= self.cl + 0.5*self.ck),
                             (z > self.cl+0.5*self.ck)*(z <= self.cl+self.ck),
                             z>=self.cl+self.ck],
                            [f11,f12,f13,f14,f15,f16,f17,f18,f19,f110,f111,
                             f112,zero],self.ak,self.al,self.ck,self.cl)/norm

    def f2(self,z):
        """
        Case where cl/4 < ck <= cl/3 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize()
        return np.piecewise(z,
                            [(z >= 0.)*(z <= 0.5*self.ck),
                             (z > 0.5*self.ck)*(z <= 0.5*self.cl-self.ck),
                             (z > 0.5*self.cl-self.ck)*(z <= self.ck),
                             (z > self.ck)*(z <= 0.5*self.cl - 0.5*self.ck),
                             (z > 0.5*self.cl-0.5*self.ck)*(z <= 0.5*self.cl),
                             (z > 0.5*self.cl)*(z <= 0.5*self.cl+0.5*self.ck),
                             (z > 0.5*self.ck+0.5*self.cl)*(z <= self.cl-self.ck),
                             (z > self.cl-self.ck)*(z <= self.ck+0.5*self.cl),
                             (z > self.ck +0.5*self.cl)*(z <= self.cl-0.5*self.ck),
                             (z > self.cl-0.5*self.ck)*(z <= self.cl),
                             (z > self.cl)*(z <= self.cl + 0.5*self.ck),
                             (z > self.cl+0.5*self.ck)*(z <= self.cl+self.ck),
                             z>=self.cl+self.ck],
                            [f21,f22,f23,f24,f25,f26,f27,f28,f29,f210,f211,
                             f212,zero],self.ak,self.al,self.ck,self.cl)/norm

    def f3(self,z):
        """
        Case where cl/3 < ck <= cl/2 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize()
        return np.piecewise(z,
                            [(z >= 0.)*(z <= 0.5*self.cl-self.ck),
                             (z > 0.5*self.cl-self.ck)*(z <= 0.5*self.ck),
                             (z > 0.5*self.ck)*(z <= 0.5*self.cl-0.5*self.ck),
                             (z > 0.5*self.cl-0.5*self.ck)*(z <= self.ck),
                             (z > self.ck)*(z <= 0.5*self.cl),
                             (z > 0.5*self.cl)*(z <= self.cl-self.ck),
                             (z > self.cl-self.ck)*(z <= 0.5*self.cl+0.5*self.ck),
                             (z > 0.5*self.ck + 0.5*self.cl)*(
                                 z <= self.cl - 0.5*self.ck),
                             (z > self.cl-0.5*self.ck)*(z <= self.ck+0.5*self.cl),
                             (z > self.ck+0.5*self.cl)*(z <= self.cl),
                             (z > self.cl)*(z <= self.cl + 0.5*self.ck),
                             (z > self.cl+0.5*self.ck)*(z <= self.cl+self.ck),
                             z>=self.cl+self.ck],
                            [f31,f32,f33,f34,f35,f36,f37,f38,f39,f310,f311,
                             f312,zero],self.ak,self.al,self.ck,self.cl)/norm

    def f4(self,z):
        """
        Case where cl/2 < ck <= 2cl/3 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize()
        return np.piecewise(z,
                            [(z >= 0)*(z <= self.ck-0.5*self.cl),
                             (z > self.ck -0.5*self.cl)*(z <= 0.5*self.cl-0.5*self.ck),
                             (z > 0.5*self.cl-0.5*self.ck)*(z <= 0.5*self.ck),
                             (z > 0.5*self.ck)*(z <= self.cl-self.ck),
                             (z > self.cl-self.ck)*(z <= 0.5*self.cl),
                             (z > 0.5*self.cl)*(z <= self.ck),
                             (z > self.ck)*(z <= self.cl-0.5*self.ck),
                             (z > self.cl-0.5*self.ck)*(z <= 0.5*self.ck+0.5*self.cl),
                             (z > 0.5*self.ck+0.5*self.cl)*(z <= self.cl),
                             (z > self.cl)*(z <= self.ck+0.5*self.cl),
                             (z > self.ck+0.5*self.cl)*(z <= self.cl+0.5*self.ck),
                             (z >= self.cl+0.5*self.ck)*(z <= self.cl+self.ck),
                             z>=self.cl+self.ck],
                            [f41,f42,f43,f44,f45,f46,f47,f48,f49,f410,f411,
                             f412,zero],self.ak,self.al,self.ck,self.cl)/norm

    def f5(self,z):
        """
        Case where 2cl/3 < ck <= 3cl/4 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize()
        return np.piecewise(z,
                            [(z >= 0.)*(z <= 0.5*(self.cl-self.ck)),
                             (z > 0.5*(self.cl-self.ck))*(z <= self.ck-0.5*self.cl),
                             (z > self.ck-0.5*self.cl)*(z <= self.cl-self.ck),
                             (z > self.cl-self.ck)*(z <= 0.5*self.ck),
                             (z > 0.5*self.ck)*(z <= 0.5*self.cl),
                             (z > 0.5*self.cl)*(z <= self.cl-0.5*self.ck),
                             (z > self.cl-0.5*self.ck)*(z <= self.ck),
                             (z > self.ck)*(z <= 0.5*(self.cl+self.ck)),
                             (z > 0.5*self.ck+0.5*self.cl)*(z <= self.cl),
                             (z > self.cl)*(z <= self.ck+0.5*self.cl),
                             (z > self.ck+0.5*self.cl)*(z <= self.cl+0.5*self.ck),
                             (z >= self.cl+0.5*self.ck)*(z <= self.cl+self.ck),
                             z>=self.cl+self.ck],
                            [f51,f52,f53,f54,f55,f56,f57,f58,f59,f510,f511,
                             f512,zero],self.ak,self.al,self.ck,self.cl)/norm

    def f6(self,z):
        """
        Case where 3cl/4 < ck <= cl
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize()
        return np.piecewise(z,
                            [(z >= 0.)*(z <= 0.5*(self.cl-self.ck)),
                             (z > 0.5*(self.cl-self.ck))*(z <= self.cl-self.ck),
                             (z > self.cl-self.ck)*(z <= self.ck-0.5*self.cl),
                             (z > self.ck-0.5*self.cl)*(z <= 0.5*self.ck),
                             (z > 0.5*self.ck)*(z <= 0.5*self.cl),
                             (z > 0.5*self.cl)*(z <= self.cl-0.5*self.ck),
                             (z > self.cl-0.5*self.ck)*(z <= self.ck),
                             (z > self.ck)*(z <= 0.5*(self.ck+self.cl)),
                             (z > 0.5*self.ck+0.5*self.cl)*(z <= self.cl),
                             (z > self.cl)*(z <= self.ck+0.5*self.cl),
                             (z > self.ck+0.5*self.cl)*(z <= self.cl+0.5*self.ck),
                             (z >= self.cl+0.5*self.ck)*(z <= self.cl+self.ck),
                             z>=self.cl+self.ck],
                            [f61,f62,f63,f64,f65,f66,f67,f68,f69,f610,f611,
                             f612,zero],self.ak,self.al,self.ck,self.cl)/norm
    
    def f7(self,z):
        """
        Case where ck = cl;
        this is not normalized, normalized in evaluation function.
        These functions need to call ck for the numpy piecewise function
        """
        norm = np.sqrt(nfcn(self.ak)*nfcn(self.al))
        return np.piecewise(z, [(z>=0.)*(z <=0.5*self.ck),
                                (z >0.5*self.ck)*(z <= self.ck),
                                (z >self.ck)*(z <= 1.5*self.ck),
                                (z > 1.5*self.ck)*(z<=2.*self.ck),
                                z>2.*self.ck],
                            [f71,f72,f73,f74,zero],
                            self.ak,self.al,self.ck,self.cl)/norm


class GenGCContinuous():
    """
    Continuous version of the GenGC object that can be used for 
    array evaluations. This function requires that ck <= cl has 
    already been checked (see ContinuousGenGC __call__ function in 
    gengc1d.py for how to check and evaluate).
    Goal is to output the masks and evaluate the functions themselves, given
    an input of c, a, and distance arrays.
    """

    def evaluate_fcns(self,conditions,fcns,
                      z,ak,al,ck,cl,norm):
        """
        Given a list of conditions and functions
        evaluates the functions and returns the appropriate list
        """
        ret = np.zeros_like(z)
        for cond, fcn in zip(conditions, fcns):
            if z[cond].size != 0:
                ret[cond] = fcn(z[cond],ak[cond],al[cond],
                                ck[cond],cl[cond])/norm[cond]
        return ret
    
    def make_fi_cmasks(self,ck,cl):
        """
        Returns a list of masks and associated functions based 
        on the input values of ck and cl
        """
        fi_masks = [ck <=0.25*cl,
                    np.logical_and(ck>0.25*cl,
                                   ck<=cl/3),
                    np.logical_and(ck>cl/3,
                                   ck <=0.5*cl),
                    np.logical_and(ck>0.5*cl,
                                   ck <= 2.*cl/3.),
                    np.logical_and(ck>2.*cl/3.,
                                   ck <= 0.75*cl),
                    np.logical_and(ck>0.75*cl,
                                   ck < cl),
                    ck==cl,
                    ck > cl]
        fcn_list = [self.f1,self.f2,self.f3,self.f4,
                    self.f5,self.f6,self.f7,zero]

        return fi_masks, fcn_list

    def normalize(self,ak,al,ck,cl):
        """
        computes normalization factor common to functions f1,f2,...,f6
        """
        return np.sqrt(nfcn(ak)*nfcn(al)*ck**3*cl**3)

    def f1_cmask(self,ck,cl):
        return ck <=0.25*cl
    
    def f1(self,z,ak,al,ck,cl):
        """
        Case where 0 < ck <= cl/4 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize(ak,al,ck,cl)
        conditions = [(z >= 0.)*(z <= 0.5*ck),
                      (z > 0.5*ck)*(z <= ck),
                      (z > ck)*(z <= 0.5*cl - ck),
                      (z > 0.5*cl - ck)*(
                          z <= 0.5*cl - 0.5*ck),
                      (z > 0.5*cl-0.5*ck)*(z <= 0.5*cl),
                      (z > 0.5*cl)*(z <= 0.5*(cl+ck)),
                      (z > 0.5*(ck+cl))*(z <= 0.5*cl+ck),
                      (z > 0.5*cl+ck)*(z <= cl-ck),
                      (z > cl-ck)*(z <= cl -0.5*ck),
                      (z > cl-0.5*ck)*(z <= cl),
                      (z > cl)*(z <= cl + 0.5*ck),
                      (z > cl+0.5*ck)*(z <= cl+ck),
                      z>=cl+ck]
        f1_fcns = [f11,f12,f13,f14,f15,f16,f17,f18,f19,f110,f111,
                   f112,zero]
        return self.evaluate_fcns(conditions,f1_fcns,
                                  z,ak,al,ck,cl,norm)

    def f2(self,z,ak,al,ck,cl):
        """
        Case where cl/4 < ck <= cl/3 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize(ak,al,ck,cl)
        conditions = [np.logical_and(z >= 0.,z <= 0.5*ck),
                      np.logical_and(z > 0.5*ck,z <= 0.5*cl-ck),
                      np.logical_and(z > 0.5*cl-ck,z <= ck),
                      np.logical_and(z > ck,z <= 0.5*cl - 0.5*ck),
                      np.logical_and(z > 0.5*cl-0.5*ck,z <= 0.5*cl),
                      np.logical_and(z > 0.5*cl,z <= 0.5*cl+0.5*ck),
                      np.logical_and(z > 0.5*ck+0.5*cl,z <= cl-ck),
                      np.logical_and(z > cl-ck,z <= ck+0.5*cl),
                      np.logical_and(z > ck +0.5*cl,z <= cl-0.5*ck),
                      np.logical_and(z > cl-0.5*ck,z <= cl),
                      np.logical_and(z > cl,z <= cl + 0.5*ck),
                      np.logical_and(z > cl+0.5*ck,z <= cl+ck),
                      z>=cl+ck]
        f2_fcns = [f21,f22,f23,f24,f25,f26,f27,f28,f29,f210,f211,
                   f212,zero]
        return self.evaluate_fcns(conditions,f2_fcns,
                                  z,ak,al,ck,cl,norm)
        

    def f3(self,z,ak,al,ck,cl):
        """
        Case where cl/3 < ck <= cl/2 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize(ak,al,ck,cl)
        conditions = [np.logical_and(z >= 0.,z <= 0.5*cl-ck),
                      np.logical_and(z > 0.5*cl-ck,z <= 0.5*ck),
                      np.logical_and(z > 0.5*ck,z <= 0.5*cl-0.5*ck),
                      np.logical_and(z > 0.5*cl-0.5*ck,z <= ck),
                      np.logical_and(z > ck,z <= 0.5*cl),
                      np.logical_and(z > 0.5*cl,z <= cl-ck),
                      np.logical_and(z > cl-ck,z <= 0.5*cl+0.5*ck),
                      np.logical_and(z > 0.5*ck + 0.5*cl,
                          z <= cl - 0.5*ck),
                      np.logical_and(z > cl-0.5*ck,z <= ck+0.5*cl),
                      np.logical_and(z > ck+0.5*cl,z <= cl),
                      np.logical_and(z > cl,z <= cl + 0.5*ck),
                      np.logical_and(z > cl+0.5*ck,z <= cl+ck),
                      z>=cl+ck]
        f3_fcns = [f31,f32,f33,f34,f35,f36,f37,f38,f39,f310,f311,
                   f312,zero]
        return self.evaluate_fcns(conditions,f3_fcns,
                                  z,ak,al,ck,cl,norm)
        

    def f4(self,z,ak,al,ck,cl):
        """
        Case where cl/2 < ck <= 2cl/3 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize(ak,al,ck,cl)
        conditions = [np.logical_and(z >= 0,z <= ck-0.5*cl),
                      np.logical_and(z > ck -0.5*cl,z <= 0.5*cl-0.5*ck),
                      np.logical_and(z > 0.5*cl-0.5*ck,z <= 0.5*ck),
                      np.logical_and(z > 0.5*ck,z <= cl-ck),
                      np.logical_and(z > cl-ck,z <= 0.5*cl),
                      np.logical_and(z > 0.5*cl,z <= ck),
                      np.logical_and(z > ck,z <= cl-0.5*ck),
                      np.logical_and(z > cl-0.5*ck,z <= 0.5*ck+0.5*cl),
                      np.logical_and(z > 0.5*ck+0.5*cl,z <= cl),
                      np.logical_and(z > cl,z <= ck+0.5*cl),
                      np.logical_and(z > ck+0.5*cl,z <= cl+0.5*ck),
                      np.logical_and(z >= cl+0.5*ck,z <= cl+ck),
                      z>=cl+ck]
        f4_fcns = [f41,f42,f43,f44,f45,f46,f47,f48,f49,f410,f411,
                   f412,zero]

        return self.evaluate_fcns(conditions,f4_fcns,
                                  z,ak,al,ck,cl,norm)
        

    def f5(self,z,ak,al,ck,cl):
        """
        Case where 2cl/3 < ck <= 3cl/4 
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize(ak,al,ck,cl)
        conditions = [np.logical_and(z >= 0.,z <= 0.5*(cl-ck)),
                      np.logical_and(z > 0.5*(cl-ck),z <= ck-0.5*cl),
                      np.logical_and(z > ck-0.5*cl,z <= cl-ck),
                      np.logical_and(z > cl-ck,z <= 0.5*ck),
                      np.logical_and(z > 0.5*ck,z <= 0.5*cl),
                      np.logical_and(z > 0.5*cl,z <= cl-0.5*ck),
                      np.logical_and(z > cl-0.5*ck,z <= ck),
                      np.logical_and(z > ck,z <= 0.5*(cl+ck)),
                      np.logical_and(z > 0.5*ck+0.5*cl,z <= cl),
                      np.logical_and(z > cl,z <= ck+0.5*cl),
                      np.logical_and(z > ck+0.5*cl,z <= cl+0.5*ck),
                      np.logical_and(z >= cl+0.5*ck,z <= cl+ck),
                      z>=cl+ck]
        f5_fcns = [f51,f52,f53,f54,f55,f56,f57,f58,f59,f510,f511,
                   f512,zero]

        return self.evaluate_fcns(conditions,f5_fcns,
                                  z,ak,al,ck,cl,norm)
        

    def f6(self,z,ak,al,ck,cl):
        """
        Case where 3cl/4 < ck <= cl
        with ck <= cl
        Must normalize output since normalization is different for f7
        """
        norm = self.normalize(ak,al,ck,cl)
        conditions = [np.logical_and(z >= 0.,z <= 0.5*(cl-ck)),
                      np.logical_and(z > 0.5*(cl-ck),z <= cl-ck),
                      np.logical_and(z > cl-ck,z <= ck-0.5*cl),
                      np.logical_and(z > ck-0.5*cl,z <= 0.5*ck),
                      np.logical_and(z > 0.5*ck,z <= 0.5*cl),
                      np.logical_and(z > 0.5*cl,z <= cl-0.5*ck),
                      np.logical_and(z > cl-0.5*ck,z <= ck),
                      np.logical_and(z > ck,z <= 0.5*(ck+cl)),
                      np.logical_and(z > 0.5*ck+0.5*cl,z <= cl),
                      np.logical_and(z > cl,z <= ck+0.5*cl),
                      np.logical_and(z > ck+0.5*cl,z <= cl+0.5*ck),
                      np.logical_and(z >= cl+0.5*ck,z <= cl+ck),
                      z>=cl+ck]
        f6_fcns = [f61,f62,f63,f64,f65,f66,f67,f68,f69,f610,f611,
                   f612,zero]

        return self.evaluate_fcns(conditions,f6_fcns,
                                  z,ak,al,ck,cl,norm)
        
    
    def f7(self,z,ak,al,ck,cl):
        """
        Case where ck = cl;
        this is not normalized, normalized in evaluation function.
        These functions need to call ck for the numpy piecewise function
        """
        norm = np.sqrt(nfcn(ak)*nfcn(al))
        conditions = [np.logical_and(z>=0.,z <=0.5*ck),
                      np.logical_and(z >0.5*ck,z <= ck),
                      np.logical_and(z >ck,z <= 1.5*ck),
                      np.logical_and(z > 1.5*ck,z<=2.*ck),
                      z>2.*ck]
        f7_fcns = [f71,f72,f73,f74,zero]
        
        return self.evaluate_fcns(conditions,f7_fcns,
                                  z,ak,al,ck,cl,norm)
        
