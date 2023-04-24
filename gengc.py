"""
Generalized Gaspari-Cohn (GenGC) Correlation Function
Shay Gilpin
created July 1, 2021
last updated April 24, 2023

** ------------------------------------------------------------------ **
        ** CITE THE CORRESPONDING PUBLICATION AND CODE DOI **
(paper): Gilpin, S., Matsuo, T., and Cohn, S.E. (2023). A generalized,
         compactly-supported correlation function for data assimilation
         applications. Q. J. Roy. Meteor. Soc.
(code): Gilpin, S. A Generalized Gaspari-Cohn Correlation Function
        (see Github repo for DOI)
** ------------------------------------------------------------------ **

Implementation of the GenGC Correlation Function
defined in Gilpin et al. (2023), "A generalied, compactly-supported
correlation function for data assimilation applications" in QJRMS.

This script implements Eq.(36) in Appendix B of Gilpin et al. (2023). 
The function gencpr() constructs GenGC acting on a vector of normed
distances c. The functions f1, f2, ... , f6 define each case of
Eq. (36) and implement the coefficients in Tables 1-19. Normalization
is done in the main gencpr() function for f1, ... , f6. The function
f7 is the case where ck = cl, given in Eq.(33) of Gaspari et al. (2006),
QJRMS.

For this implementation, ak, al, ck, and cl are either scalars or 
1D arrays of length n. These values are computed before evaluating
GenGC using gencpr. For example, consider the case where a and c 
are computed from continuous functions on the unit circle:

Ex:
    grid = np.linspace(0,2.*np.pi,201)
    avalues = 0.25*np.sin(grid)+0.5
    cvalues = 0.25*np.pi - 0.15*np.pi*np.sin(grid)

To obtain one row of the correlation matrix (say the first row),
we can call gencpr() after computing the normed distances (here chordal
distance) with the first grid cell:
Ex:
    dist = 2.*np.sin(np.abs(grid[0]-grid))
    corr_row1 = gencpr(dist,avalues[0],avalues,cvalues[0],cvalues,
                       len(grid))
To recover the full correlation matrix (as a 2D array), loop through
the full set of grid points to construct each row.

This formulation works well to construct 2D correlations with respect
to a fixed grid point (e.g. the examples in Section 4.2 of Gilpin et 
al, 2023).

"""

import numpy as np


def gencpr(z,ak,al,ck,cl,n):
    """
    Constructs the GenGC correlation function. Normalization of
    functions f1, ... , f6 are done here rather than in the 
    specific functions themselves.

    Input:
    z - (1d array) normed distance between fixed grid point xk 
        and xl, where xl is a 1d array of length n
    ak, al - values of a corresponding to grid points xk, xl used in input;
             ak is scalar, al must be the same shape as xl
    ck, cl - values of cut-off lengths corresponding to grid points
             xk, xl; ck is a scalar, cl must be the same shape as xl
    n - length of z, al, cl arrays, and is size of output array

    Output:
    out - (2D square array of length n) GenGC correlation function 
    """

    out = np.zeros(n)
    nk = 2.+44.*ak**2+6.*ak

    for j in np.arange(n):
        nl = 2.+44.*al[j]**2+6.*al[j]
        ckcl = ck**3*cl[j]**3
        if ck == cl[j]:       # here, call Eq. (33) in Gaspari et al., (2006)
            out[j] = f7(z[j],ak,al[j],ck)/np.sqrt(nk*nl)
            
        elif ck < cl[j]:
            if (ck >= 0 and ck <= 0.25*cl[j]):
                out[j] = f1(z[j],ak,al[j],ck,cl[j])/np.sqrt(nk*nl*ckcl)
            elif (ck >= 0.25*cl[j] and ck <= cl[j]/3.):
                out[j] = f2(z[j],ak,al[j],ck,cl[j])/np.sqrt(nk*nl*ckcl)
            elif (ck >= cl[j]/3. and ck <= 0.5*cl[j]):
                out[j] = f3(z[j],ak,al[j],ck,cl[j])/np.sqrt(nk*nl*ckcl)
            elif (ck >= 0.5*cl[j] and ck <= 2.*cl[j]/3.):
                out[j] = f4(z[j],ak,al[j],ck,cl[j])/np.sqrt(nk*nl*ckcl)
            elif (ck >= 2.*cl[j]/3. and ck <= 0.75*cl[j]):
                out[j] = f5(z[j],ak,al[j],ck,cl[j])/np.sqrt(nk*nl*ckcl)
            elif (ck >= 0.75*cl[j] and ck <= cl[j]):
                out[j] = f6(z[j],ak,al[j],ck,cl[j])/np.sqrt(nk*nl*ckcl)
            else:
                out[j] = 0.
        # here, must swap ck and cl indices in function
        elif ck > cl[j]:       
            if (cl[j] >=0 and cl[j] <= 0.25*ck):
                out[j] = f1(z[j],al[j],ak,cl[j],ck)/np.sqrt(nk*nl*ckcl)
            elif (cl[j] >= 0.25*ck and cl[j] <= ck/3.):
                out[j] = f2(z[j],al[j],ak,cl[j],ck)/np.sqrt(nk*nl*ckcl)
            elif (cl[j] >= ck/3. and cl[j] <= 0.5*ck):
                out[j] = f3(z[j],al[j],ak,cl[j],ck)/np.sqrt(nk*nl*ckcl)
            elif (cl[j] >= 0.5*ck and cl[j] <= 2.*ck/3.):
                out[j] = f4(z[j],al[j],ak,cl[j],ck)/np.sqrt(nk*nl*ckcl)
            elif (cl[j] >= 2.*ck/3. and cl[j] <= 0.75*ck):
                out[j] = f5(z[j],al[j],ak,cl[j],ck)/np.sqrt(nk*nl*ckcl)
            elif (cl[j] >= 0.75*ck and cl[j] <= ck):
                out[j] = f6(z[j],al[j],ak,cl[j],ck)/np.sqrt(nk*nl*ckcl)
            else:
                out[j] = 0.

    return out


def f1(z,ak,al,ck,cl):
    """
    Outputs the first function f1 defined for
    0 < ck <= cl/4
    given in Tables 1-3 of Gilpin et al (2023)
    Input:
    z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
        need z >= 0
    ak, al - (scalar) corresponding scalars for xk, xl
    ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    """
    
    if (z >= 0. and z <= 0.5*ck): # f11(z)
        return z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + z**4*16.*(1.-al)/cl + z**2*40.*ck**2*(al-1.-6.*ak+6.*ak*al)/(3.*cl) + 5.*ck**3*(1.+14.*ak) + 3.*ck**4*(al-1.-30.*ak+30.*ak*al)/cl
    
    elif (z > 0.5*ck and z <= ck): # f12(z)
        return z**5*32.*(ak*al-ak)/(3.*ck*cl) + \
            z**4*32.*(ak-ak*al)/cl + \
            z**2*320.*ck**2*(ak*al-ak)/(3.*cl) +\
            z*10.*ck**3*(-1.+2.*ak+al-2.*ak*al)/cl +\
            5.*ck**3*(1.+14.*ak)+96.*ck**4*(ak*al-ak)/cl +\
            ck**5*(-1.+2.*ak+al-2.*ak*al)/(3.*z*cl)
    
    elif (z > ck and z <= 0.5*cl - ck): # f13(z)
        return z*10.*ck**3*(-1.-14.*ak+al+14.*ak*al)/cl + \
            5.*ck**3*(1.+14.*ak) + \
            ck**5*(-1.-62.*ak+al+62*ak*al)/(3.*cl*z)
    
    elif (z > 0.5*cl - ck and z <= 0.5*cl - 0.5*ck): # f14(z)
        return z**5*16.*(2.*ak*al-ak)/(3.*ck*cl) + \
            z**4*8*(ak-2.*ak*al)/ck + z**4*16.*(2*ak*al-ak)/cl + \
            z**3*(20.*ak-40.*ak*al) + \
            z**2*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) +\
            z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(-1.-6.*ak+al-2.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
            5.*ck**3*(1.+6.*ak+16.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/(2.*ck) + \
            12.*ck**4*(2.*ak*al - ak)/z + ck**5*(-1.-30.*ak+al-2.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6*ck*z)
    
    elif (z > 0.5*cl-0.5*ck and z <= 0.5*cl): # f15(z)
        return z**5*16.*(ak-1.+2.*al-2.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)/(2.*cl) + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)/(2.*ck) + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) +5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)
            
    
    elif (z > 0.5*cl and z <= 0.5*(cl+ck)): # f16(z)
        return z**5*16.*(1.-ak-2.*al+2.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+2.*al-2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + \
            z**3.*(10.-20.*al) + \
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al+2.*al*ak)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(ak-1.+2.*al-2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(1.-ak-2.*al+2.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(ak-1.+2.*al-2.*ak*al)/(6.*ck*z)

    elif (z > 0.5*(ck+cl) and z <= 0.5*cl+ck): # f17(z)
        return z**5*16.*(ak-2.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(2.*ak*al-ak)/ck + z**4*16.*(2.*ak*al-ak)/cl +\
            z**3*(20.*ak-40.*ak*al) + \
            z**2.*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-2.*ak*al)/(3.*ck) + \
            z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(2.*ak*al-al-8.*ak)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(2.*ak*al-ak)/ck + \
            10.*ck**3*(4.*ak+al+6.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(ak-2.*ak*al)*0.5/ck + \
            12.*ck**4*(2.*ak*al-ak)/z + ck**5*(2.*ak*al-al-32.*ak)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(2.*ak*al-ak)/(6.*ck*z)

    elif (z > 0.5*cl+ck and z <= cl-ck): # f18(z)
        return 10.*ck**3*(al+14.*ak*al) - z*10.*ck**3*(al+14.*ak*al)/cl - ck**5*(al+62.*ak*al)/(3.*z*cl)

    elif (z > cl-ck and z <= cl -0.5*ck): # f19(z)
        return -16.*z**5*ak*al/(3.*ck*cl) + \
            z**4*16.*ak*al/ck - z**4*16.*ak*al/cl + \
            z**3*40.*ak*al + \
            z**2*160.*ck**2*ak*al/(3.*cl) - z**2*160.*cl**2*ak*al/(3.*ck) + \
            z*80.*cl**3*ak*al/ck - z*80.*ck**2*ak*al - z*10.*ck**3*(al+6.*ak*al)/cl - z*80.*cl**2*ak*al + \
            10.*ck**3*(al+6.*ak*al) + 48.*ck**4*ak*al/cl + 80.*cl**3*ak*al - 48.*cl**4*ak*al/ck + \
            80.*ck**2*cl**2*ak*al/(3.*z) - 24.*ck**4*ak*al/z - ck**5*(al+30.*ak*al)/(3.*cl*z) - 24.*cl**4*ak*al/z + 32.*cl**5*ak*al/(3.*ck*z)

    elif (z > cl-0.5*ck and z <= cl): # f110(z)
        return z**5*16.*(ak*al-al)/(3.*ck*cl) + \
            z**4*16.*(al-ak*al)/ck - z**4*8.*al/cl + \
            z**3*20.*al + \
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(ak*al-al)/(3.*ck) + \
            z*80.*cl**3*(al-ak*al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
            5.*ck**3*(14.*ak*al+al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(ak*al-al)/ck + \
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(al-ak*al)/(3.*ck*z)

    elif (z > cl and z <= cl + 0.5*ck): # f111(z)
        return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + \
            z**3*20.*al + \
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(al-ak*al)/ck + \
                10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)
                                                
    elif (z > cl+0.5*ck and z <= cl+ck): # f112(z)
        return ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - z**4*16./cl + 40.*z**3 + z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - z*80.*ck**2 - z*80.*ck**3/cl - z*80.*cl**2 - z*80.*cl**3/ck + 80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) + 80.*ck**2*cl**2/(3.*z) - 24.*cl**4/z - 32*cl**5/(3.*ck*z))
    
    else:
        return 0.


def f2(z,ak,al,ck,cl):
    """
    Outputs the second function f2 defined for
    cl/4 <= ck <= cl/3
    from Tables 4-6 of Gilpin et al. (2023)
    Input:
    z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
        need z >= 0
    ak, al - (scalar) corresponding scalars for xk, xl
    ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    """

    if (z >= 0 and z <= 0.5*ck): # f21(z)
        return z**5*32.*(ak+al-ak*al-1.)/(3.*ck*cl) + z**4*16.*(1.-al)/cl + \
            z**2*40.*ck**2*(al-6.*ak-1.+6.*ak*al)/(3.*cl) + 5.*ck**3*(1.+14.*ak) + \
            3.*ck**4*(al-1.-30.*ak+30.*ak*al)/cl

    elif (z > 0.5*ck and z <= 0.5*cl-ck): # f22(z)
        return z**5*32.*(ak*al-ak)/(3.*ck*cl) + z**4*32.*(ak-ak*al)/cl + \
            z**2*320.*ck**2*(ak*al-ak)/(3.*cl) +  z*10.*ck**3*(2.*ak-1.+al-2.*ak*al)/cl + \
            5.*ck**3*(1.+14.*ak) + 96.*ck**4*(ak*al-ak)/cl + \
            ck**5*(2.*ak-1.+al-2.*ak*al)/(3.*cl*z)

    elif (z > 0.5*cl-ck and z <= ck): # f23(z)
        return z**5*16.*(4.*ak*al-3.*ak)/(3.*ck*cl) + \
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16.*ak/cl + z**3*(20.*ak-40.*ak*al) + \
            z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) - z**2*160.*ck**2*ak/(3.*cl) + \
            z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(10.*ak-1.+al-18.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
            5.*ck**3*(1.+6.*ak+16.*ak*al) - 48.*ck**4*ak/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
            12.*ck**4*(2.*ak*al-ak)/z + ck**5*(34.*ak-1.+al-66.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

    elif (z > ck and z <= 0.5*cl - 0.5*ck): # f24(z)
        return z**5*16*(2.*ak*al-ak)/(3.*ck*cl) + \
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16*(2.*ak*al-ak)/cl + z**3*(20.*ak-40.*ak*al) + \
            z**2*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
            z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(al-1.-6.*ak-2.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
            5.*ck**3*(1.+6.*ak+16.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
            12.*ck**4*(2.*ak*al-ak)/z + ck**5*(al-1.-30.*ak-2.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

    elif (z > 0.5*cl-0.5*ck and z <= 0.5*cl): # f25(z)
        return z**5*16.*(ak-1.+2.*al-2.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

    elif (z > 0.5*cl and z <= 0.5*cl+0.5*ck): # f26(z)
        return z**5*16.*(1.-ak-2.*al+2.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+2.*al-2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al+2.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(ak-1.+2.*al-2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(1.-ak-2.*al+2.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(ak-1.+2.*al-2.*ak*al)/(6.*ck*z)

    elif (z > 0.5*ck+0.5*cl and z <= cl-ck): # f27(z)
        return z**5*16.*(ak-2.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(2.*ak*al-ak)/ck + z**4*16.*(2.*ak*al-ak)/cl + z**3*(20.*ak-40.*ak*al) + \
            z**2*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-2.*ak*al)/(3.*ck) + \
            z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(2.*ak*al-8.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(2.*ak*al-ak)/ck + \
            10.*ck**3*(4.*ak+al+6.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(ak-2.*ak*al)*0.5/ck + \
            12.*ck**4*(2.*ak*al-ak)/z + ck**5*(2.*ak*al-al-32.*ak)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(2.*ak*al-ak)/(6.*ck*z)

    elif (z > cl-ck and z <= ck+0.5*cl): # f28(z)
        return z**5*16.*(ak-3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(4.*ak*al-ak)/ck + z**4*16.*(ak*al-ak)/cl + z**3*20.*ak + \
            z**2*160.*ck**2*(ak-ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-10.*ak*al)/(3.*ck) + \
            z*10.*ck**3*(10.*ak*al-al-8.*ak)/cl - z*40.*ck**2*ak - z*10.*cl**2*(ak+6.*ak*al) + z*5.*cl**3*(18.*ak*al-ak)/ck + \
            10.*ck**3*(4.*ak+al-2.*ak*al) + 48.*ck**4*(ak-ak*al)/cl + 5.*cl**3*(ak+14.*ak*al) + 3.*cl**4*(ak-34.*ak*al)*0.5/ck + \
            ck**5*(34.*ak*al-al-32.*ak)/(3.*cl*z) -12.*ck**4*ak/z + 10.*ck**2*cl**2*(ak+6.*ak*al)/(3.*z) - 3.*cl**4*(ak+30.*ak*al)/(4.*z) + cl**5*(66.*ak*al-ak)/(6.*ck*z)

    elif (z > ck +0.5*cl and z <= cl-0.5*ck): # f29(z)
        return -z**5*16.*ak*al/(3.*ck*cl) + \
            z**4*16.*ak*al/ck - z**4*16.*ak*al/cl + z**3*40.*ak*al + \
            z**2*160.*ck**2*ak*al/(3.*cl) - z**2*160.*cl**2*ak*al/(3.*ck) + \
            z*80.*cl**3*ak*al/ck - z*80.*ck**2*ak*al - z*10.*ck**3*(al+6.*ak*al)/cl - z*80.*cl**2*ak*al + \
            10.*ck**3*(al+6.*ak*al) + 48.*ck**4*ak*al/cl + 80.*cl**3*ak*al - 48.*cl**4*ak*al/ck + \
            80.*ck**2*cl**2*ak*al/(3.*z) - 24.*ck**4*ak*al/z - ck**5*(al+30.*ak*al)/(3.*cl*z) -24.*cl**4*ak*al/z + 32.*cl**5*ak*al/(3.*ck*z)

    elif (z > cl-0.5*ck and z <= cl): # f210 ( = f110(z))
        return z**5*16.*(ak*al-al)/(3.*ck*cl) + \
            z**4*16.*(al-ak*al)/ck - z**4*8.*al/cl + \
            z**3*20.*al + \
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(ak*al-al)/(3.*ck) + \
            z*80.*cl**3*(al-ak*al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
            5.*ck**3*(14.*ak*al+al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(ak*al-al)/ck + \
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(al-ak*al)/(3.*ck*z)

    elif (z > cl and z <= cl + 0.5*ck): # f211 ( = f111(z))
        return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + \
            z**3*20.*al + \
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(al-ak*al)/ck + \
                10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)
                                                
    elif (z > cl+0.5*ck and z <= cl+ck): # f212 ( = f112(z))
        return ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - z**4*16./cl + 40.*z**3 + z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - z*80.*ck**2 - z*80.*ck**3/cl - z*80.*cl**2 - z*80.*cl**3/ck + 80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) + 80.*ck**2*cl**2/(3.*z) - 24.*cl**4/z - 32*cl**5/(3.*ck*z))
    
    else:
        return 0.




def f3(z,ak,al,ck,cl):
    """
    Outputs the second function f3 defined for
    cl/3 <= ck <= cl/2
    from Tables 7-9 of Gilpin et al. (2023)
    Input:
    z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
        need z >= 0
    ak, al - (scalar) corresponding scalars for xk, xl
    ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    """

    if (z >= 0. and z <= 0.5*cl-ck): # f31(z)
        return z**5*32.*(ak-1.-ak*al+al)/(3.*ck*cl) + z**4*16.*(1.-al)/cl + \
            z**2*40.*ck**2*(al-1.+6.*ak*al-6.*ak)/(3.*cl) + 5.*ck**3*(1.+14.*ak) + \
            3.*ck**4*(al-1.-30.*ak+30.*ak*al)/cl

    elif (z > 0.5*cl-ck and z <= 0.5*ck): # f32(z)
        return z**5*16.*(ak+2.*al-2.)/(3.*ck*cl) + \
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16*(1.-ak-al+2.*ak*al)/cl + z**3*(20.*ak-40.*ak*al) + \
            z**2*40.*ck**2*(al-2.*ak-1.-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
            z*40.*ck**2*(2.*ak*al-ak) + z*80.*ck**3*(ak-2.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
            5.*ck**3*(1.+6.*ak+16.*ak*al) + 3.*ck**4*(al-1.-14.*ak-2.*ak*al)/cl  + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
            12.*ck**4*(2.*ak*al-ak)/z + 32.*ck**5*(ak-2.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

    elif (z > 0.5*ck and z <= 0.5*cl-0.5*ck): # f33(z)
        return z**5*16*(4.*ak*al-3.*ak)/(3.*ck*cl) + \
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16.*ak/cl + z**3*20.*ak*(1.-2.*al) + \
            z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) - z**2*160.*ck**2*ak/(3.*cl) + \
            z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(10.*ak-1+al-18.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
            5.*ck**3*(1.+6.*ak+16.*ak*al) - 48.*ck**4*ak/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
            12.*ck**4*(2.*ak*al-ak)/z + ck**5*(34.*ak-1+al-66.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

    elif (z > 0.5*cl-0.5*ck and z <= ck): # f34(z)
        return z**5*16.*(2.*al-ak-1.)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(4.*ak-1+2.*al-4.*ak*al)/cl + \
            z**3*10.*(1.-2.*al) + \
            z**2*20.*ck**2*(1.-10.*ak-2.*al+4.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + z*5.*ck**3*(18.*ak-1.-32.*ak*al)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+4.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-128.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

    elif (z > ck and z <= 0.5*cl): # f35(z)
        return z**5*16.*(ak-1.+2.*al-2.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

    elif (z > 0.5*cl and z <= cl-ck): # f36(z)
        return z**5*16.*(1.-ak-2.*al+2.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+2.*al-2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + z**3*(10.-20.*al) + \
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al+2.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(ak-1.+2.*al-2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(1.-ak-2.*al+2.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - ck**5*(1.+62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(ak-1.+2.*al-2.*ak*al)/(6.*ck*z)

    elif (z > cl-ck and z <= 0.5*cl+0.5*ck): # f37(z)
        return z**5*16.*(1.-ak-2.*al+ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(2.*al-1.-2.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
            z**2*20.*ck**2*(1.+6.*ak-2.*al-4.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(16.*ak*al-1.-14.*ak)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-28.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(64.*ak*al-1.-62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

    elif (z > 0.5*ck + 0.5*cl and z <= cl - 0.5*ck): # f38(z)
        return z**5*16.*(ak-3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(4.*ak*al-ak)/ck + z**4*16.*(ak*al-ak)/cl + z**3*20.*ak + \
            z**2*160.*ck**2*(ak-ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-10.*ak*al)/(3.*ck) + \
            z*10.*ck**3*(10.*ak*al-al-8.*ak)/cl - z*40.*ck**2*ak - z*10.*cl**2*(ak+6.*ak*al) + z*5.*cl**3*(18.*ak*al-ak)/ck + \
            10.*ck**3*(al+4.*ak-2.*ak*al) + 48.*ck**4*(ak-ak*al)/cl + 5.*cl**3*(ak+14.*ak*al) + 3.*cl**4*(ak-34.*ak*al)*0.5/ck + \
            ck**5*(34.*ak*al-32.*ak-al)/(3.*cl*z) - 12.*ck**4*ak/z + 10.*ck**2*cl**2*(ak+6.*ak*al)/(3.*z) - 3.*cl**4*(ak+30.*ak*al)/(4.*z) + cl**5*(66.*ak*al-ak)/(6.*ck*z)

    elif (z > cl-0.5*ck and z <= ck+0.5*cl): # f39(z)
        return z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + \
            z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + z**3*20.*(ak+al-2.*ak*al) + \
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck)+ \
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + \
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*ak*al-16.*ak-al)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)
    elif (z > ck+0.5*cl and z <= cl): # f310(z)
        return z**5*16.*(ak*al-al)/(3.*ck*cl) + \
            z**4*16.*(al-ak*al)/ck - z**4*8.*al/cl + z**3*20.*al + \
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(ak*al-al)/(3.*ck) + \
            z*80.*cl**3*(al-ak*al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(ak*al-al)/ck + \
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z)- 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(al-ak*al)/(3.*ck*z)


    elif (z > cl and z <= cl + 0.5*ck): # f311 ( = f111(z))
        return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + \
            z**3*20.*al + \
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + 48.*cl**4*(al-ak*al)/ck + \
                10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)
                                                
    elif (z > cl+0.5*ck and z <= cl+ck): # f312 ( = f112(z))
        return ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - z**4*16./cl + 40.*z**3 + z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - z*80.*ck**2 - z*80.*ck**3/cl - z*80.*cl**2 - z*80.*cl**3/ck + 80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) + 80.*ck**2*cl**2/(3.*z) - 24.*cl**4/z - 32*cl**5/(3.*ck*z))
    
    else:
        return 0.


def f4(z,ak,al,ck,cl):
    """
    Outputs the second function f2 defined for
    cl/2 <= ck <= 2cl/3
    from Tables 10-13 from Gilpin et al (2023).
    Input:
    z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
        need z >= 0
    ak, al - (scalar) corresponding scalars for xk, xl
    ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    """

    if (z >= 0 and z <= ck-0.5*cl): # f41(z)
        return z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + \
            z**4*16.*(ak-2.*ak*al)/ck + z**4*16.*(1.-2.*ak-al+4.*ak*al)/cl + \
            z**2*40.*ck**2*(2.*ak-1.+al-10.*ak*al)/(3.*cl) + z**2*40.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
            5.*ck**3*(1.-2.*ak+32.*ak*al) + 3.*ck**4*(2.*ak-1.+al-34.*ak*al)/cl + 10.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/ck

    elif (z > ck -0.5*cl and z <= 0.5*cl-0.5*ck): # f42(z)
        return z**5*16.*(ak-2.+2.*al)/(3.*ck*cl) + \
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16.*(1.-ak-al+2.*ak*al)/cl + z**3*(20.*ak-40.*ak*al) + \
            z**2*40.*ck**2*(al-1.-2.*ak-2.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
            z*40.*ck**2*(2.*ak*al-ak) + z*80.*ck**3*(ak-2.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + \
            5.*ck**3*(1.+6.*ak+16.*ak*al) + 3.*ck**4*(al-1.-14.*ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + \
            12.*ck**4*(2.*ak*al-ak)/z + 32.*ck**5*(ak-2.*ak*al)/(3.*cl*z) + 10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

    elif (z > 0.5*cl-0.5*ck and z <= 0.5*ck): # f43(z)
        return z**5*16.*(3.*ak-3.+4.*al-4.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8./cl + z**3*(10.-20.*al) + \
            z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) - z**2*20.*ck**2*(1.+6.*ak)/(3.*cl) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + z*5.*ck**3*(1.+14.*ak-2.*al-28.*ak*al)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 - 3.*ck**4*(1.+30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + ck**5*(1.+62.*ak-2.*al-124.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

    elif (z > 0.5*ck and z <= cl-ck): # f44(z)
        return z**5*16.*(2.*al-1.-ak)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(4.*ak-1.+2.*al-4.*ak*al)/cl + z**3*(10.-20.*al) + \
            z**2*20.*ck**2*(1.-10.*ak-2.*al+4.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + z*5.*ck**3*(18.*ak-1.-32.*ak*al)/cl + z*5.*cl**2*(2.*al-1) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+4.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-128.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

    elif (z > cl-ck and z <= 0.5*cl): # f45(z)
        return z**5*16.*(2.*al-1.-ak-ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + \
            z**3*10.*(1.-2.*al+4.*ak*al) + \
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

    elif (z > 0.5*cl and z <= ck): # f46(z)
        return z**5*16.*(1.-3.*ak-2.*al+3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + \
            z**3*10.*(1.-2.*al+4.*ak*al) + \
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

    elif (z > ck and z <= cl-0.5*ck): # f47(z)
        return z**5*16.*(1.-ak-2.*al+ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(2.*al-1.-2.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
            z**2*20.*ck**2*(1.+6.*ak-2.*al-4.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(16.*ak*al-1.-14.*ak)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.+30.*ak-2.*al-28.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(64.*ak*al-1.-62.*ak)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

    elif (z > cl-0.5*ck and z <= 0.5*ck+0.5*cl): # f48(z)
        return z**5*16.*(1.-ak-3.*al+3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(al-1.)/cl + z**3*10. + \
            z**2*20.*ck**2*(1.+6.*ak-al-6.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
            z*5.*ck**3*(al-1.-14.*ak+14.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.+30.*ak-al-30.*ak*al)*0.5/cl + 5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
            ck**5*(al-1.-62.*ak+62.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

    elif (z > 0.5*ck+0.5*cl and z <= cl): # f49(z)
        return z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + \
            z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-al-2.*ak)/cl + z**3*20.*(ak+al-2.*ak*al) + \
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck) + \
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + \
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + \
            3.*ck**4*(-16.*ak-al+2.*ak*al)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)

    elif (z > cl and z <= ck+0.5*cl): # f410(z)
        return z**5*16.*(ak+al-3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(4.*ak*al-ak-2.*al)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + z**3*20.*(ak+al-2.*ak*al) +\
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak+8.*al-10.*ak*al)/(3.*ck) + \
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(18.*ak*al-ak-16.*al)/ck + \
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak+32.*al-34.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*ak*al-al-16.*ak)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-16.*al-ak)/(4.*z) + cl**5*(66.*ak*al-ak-64.*al)/(6.*ck*z)

    elif (z > ck+0.5*cl and z <= cl+0.5*ck): # f411(z)
        return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + z**3*20.*al + \
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al +48.*cl**4*(al-ak*al)/ck + \
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)

    elif (z >= cl+0.5*ck and z <= cl+ck): # f412(z)
        return ak*al*(z**5*16./(3.*ck*cl) - z**4*16/ck - z**4*16./cl + z**3*40.+ \
                      z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - \
                      z*80.*(ck**2 + ck**3/cl + cl**2 + cl**3/ck) + \
                      80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck + \
                      80.*ck**2*cl**2/(3.*z) - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) - 24.*cl**4/z - 32.*cl**5/(3.*ck*z))

    else:
        return 0.
    
    
def f5(z,ak,al,ck,cl):
    """
    Outputs the second function f2 defined for
    2cl/3 <= ck <= 3cl/4
    from Tables 14-16 of Gilpin et al (2023).
    Input:
    z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
        need z >= 0
    ak, al - (scalar) corresponding scalars for xk, xl
    ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    """

    if (z >= 0. and z <= 0.5*(cl-ck)): # f51(z)
        return z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + \
            z**4*16.*(ak-2.*ak*al)/ck + z**4*16.*(1.-2.*ak-al+4.*ak*al)/cl + \
            z**2*40.*ck**2*(2.*ak-1.+al-10.*ak*al)/(3.*cl) + z**2*40.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
            5.*ck**3*(1.-2.*ak+32.*ak*al) + 3.*ck**4*(al+2.*ak-1.-34.*ak*al)/cl + 10.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/ck

    elif (z > 0.5*(cl-ck) and z <= ck-0.5*cl): # f52(z)
        return z**5*16.*(4.*ak-3.+4.*al-6.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-2.*al)/ck + z**4*8.*(1.-2.*ak+4.*ak*al)/cl + z**3*10.*(1.-2.*ak-2.*al+4.*ak*al) + \
        z**2*20.*ck**2*(2.*ak-1.-16.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*al-1.)/(3.*ck) + \
        z*5.*(1.-2.*ak)*(1.-2.*al)*(ck**3/cl-ck**2-cl**2+cl**3/ck) + \
        5.*ck**3*(1.-2.*ak+2.*al+60.*ak*al)*0.5 + 3.*ck**4*(2.*ak-1.-64.*ak*al)*0.5/cl + 5.*cl**3*(1.+2.*ak-2.*al-4.*ak*al)*0.5 + 3.*cl**4*(2.*al-1.)*0.5/ck + \
        (1.-2.*ak)*(1.-2.*al)*(ck**5/(6.*cl) - 3.*ck**4/8. + 5.*ck**2*cl**2/12. - 3.*cl**4/8. + cl**5/(6.*ck))/z

    elif (z > ck-0.5*cl and z <= cl-ck): # f53(z)
        return z**5*16.*(3.*ak-3.+4.*al-4.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8./cl + z**3*10.*(1.-2.*al) + \
            z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) - z**2*20.*ck**2*(1.+6.*ak)/(3.*cl) + \
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + z*5.*ck**3*(1.+14.*ak-2.*al-28.*ak*al)/cl + z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 - 3.*ck**4*(1.+30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + ck**5*(1.+62.*ak-2.*al-124.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

    elif (z > cl-ck and z <= 0.5*ck): # f54(z)
        return z**5*16.*(3.*ak-3.+4.*al-5.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(1.-2.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
            z**2*20.*ck**2*(8.*ak*al-1.-6.*ak)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(1.+14.*ak-2.*al-12.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(32.*ak*al-1.-30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(1.+62.*ak-2.*al-60.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

    elif (z > 0.5*ck and z <= 0.5*cl): # f55(z)
        return z**5*16.*(2.*al-1.-ak-ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

    elif (z > 0.5*cl and z <= cl-0.5*ck): # f56(z)
        return z**5*16.*(1.-3.*ak-2.*al+3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1-2.*al+4.*ak*al) + \
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

    elif (z > cl-0.5*ck and z <= ck): # f57(z)
        return z**5*16.*(1.-3.*ak-3.*al+5.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(4.*ak-1.+al-4.*ak*al)/cl + z**3*10. + \
            z**2*20.*ck**2*(1.-10.*ak-al+10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
            z*5.*ck**3*(18.*ak-1.+al-18.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.-34.*ak-al+34.*ak*al)*0.5/cl + 5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
            ck**5*(66.*ak-1.+al-66.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

    elif (z > ck and z <= 0.5*(cl+ck)): # f58(z)
        return z**5*16.*(1.-ak-3.*al+3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(al-1.)/cl + z**3*10. + \
            z**2*20.*ck**2*(1.+6.*ak-al-6.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
            z*5.*ck**3*(al-1.-14.*ak+14.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.+30.*ak-al-30.*ak*al)*0.5/cl + 5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
            ck**5*(al-1.-62.*ak+62.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

    elif (z > 0.5*ck+0.5*cl and z <= cl): # f59(z) (= f49(z))
        return z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + \
            z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-al-2.*ak)/cl + z**3*20.*(ak+al-2.*ak*al) + \
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck) + \
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + \
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + \
            3.*ck**4*(-16.*ak-al+2.*ak*al)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)

    elif (z > cl and z <= ck+0.5*cl): # f510(z) ( = f410(z))
        return z**5*16.*(ak+al-3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(4.*ak*al-ak-2.*al)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + z**3*20.*(ak+al-2.*ak*al) +\
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak+8.*al-10.*ak*al)/(3.*ck) + \
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(18.*ak*al-ak-16.*al)/ck + \
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak+32.*al-34.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*ak*al-al-16.*ak)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-16.*al-ak)/(4.*z) + cl**5*(66.*ak*al-ak-64.*al)/(6.*ck*z)

    elif (z > ck+0.5*cl and z <= cl+0.5*ck): # f511(z) ( = f411(z))
        return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + z**3*20.*al + \
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al +48.*cl**4*(al-ak*al)/ck + \
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)

    elif (z >= cl+0.5*ck and z <= cl+ck): # f512(z) ( = f412(z))
        return ak*al*(z**5*16./(3.*ck*cl) - z**4*16/ck - z**4*16./cl + z**3*40.+ \
                      z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - \
                      z*80.*(ck**2 + ck**3/cl + cl**2 + cl**3/ck) + \
                      80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck + \
                      80.*ck**2*cl**2/(3.*z) - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) - 24.*cl**4/z - 32.*cl**5/(3.*ck*z))

    else:
        return 0.


def f6(z,ak,al,ck,cl):
    """
    Outputs the sixth function f6 defined for
    3cl/4 <= ck <= cl
    from tables 17-19 of Gilpin et al. (2023).
    Input:
    z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
        need z >= 0
    ak, al - (scalar) corresponding scalars for xk, xl
    ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    """
    
    if (z >= 0. and z <= 0.5*(cl-ck)): # f61(z)
        return z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + \
            z**4*16.*(ak-2.*ak*al)/ck + z**4*16.*(1.-2.*ak-al+4.*ak*al)/cl + \
            z**2*40.*ck**2*(2.*ak-1+al-10.*ak*al)/(3.*cl) + z**2*40.*cl**2*(2.*ak*al-ak)/(3.*ck) + \
            5.*ck**3*(1.-2.*ak+32.*ak*al) + 3.*ck**4*(2.*ak-1.+al-34.*ak*al)/cl + 10.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/ck 

    elif (z > 0.5*(cl-ck) and z <= cl-ck): # f62(z)
        return z**5*16.*(4.*ak-3.+4.*al-6.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-2.*al)/ck + z**4*8.*(1.-2.*ak+4.*ak*al)/cl + z**3*10.*(1.-2.*al-2.*ak+4.*ak*al) + \
            z**2*20.*ck**2*(2.*ak-1.-16.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*al-1.)/(3.*ck) + \
            z*5.*(2.*ak-1)*(1.-2.*al)*(ck**2 - ck**3/cl + cl**2 - cl**3/ck) + \
            5.*ck**3*(1.-2.*ak+2.*al+60.*ak*al)*0.5 + 3.*ck**4*(2.*ak-1.-64.*ak*al)*0.5/cl + 5.*cl**3*(1.+2.*ak-2.*al-4.*ak*al)*0.5 + 3.*cl**4*(2.*al-1.)*0.5/ck + \
            (2.*ak-1.)*(1.-2.*al)*(3.*ck**4/8. - ck**5/(6.*cl) - 5.*ck**2*cl**2/12. + 3.*cl**4/8. - cl**5/(6.*ck))/z

    elif (z > cl-ck and z <= ck-0.5*cl): # f63(z)
        return z**5*16.*(4.*ak-3.+4.*al-7.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-2.*al+2.*ak*al)/ck + z**4*8.*(1.-2.*ak+2.*ak*al)/cl + z**3*10.*(1.-2.*ak-2.*al+8.*ak*al) + \
            z**2*20.*ck**2*(2.*ak-1.-8.*ak*al)/(3.*cl) + z**2*20.*cl**2*(2.*al-1.-8.*ak*al)/(3.*ck) + \
        z*5.*(2.*ak-1.+2.*al-20.*ak*al)*(ck**2 - ck**3/cl + cl**2 - cl**3/ck) + \
        5.*ck**3*(1.-2.*ak+2.*al+28.*ak*al)*0.5 + 3.*ck**4*(2.*ak-1.-32.*ak*al)*0.5/cl + 5.*cl**3*(1.+2.*ak-2.*al+28.*ak*al)*0.5 + 3.*cl**4*(2.*al-1.-32.*ak*al)*0.5/ck + \
        (2.*ak-1.+2.*al-68.*ak*al)*(3.*ck**4/8. - ck**5/(6.*cl) - 5.*ck**2*cl**2/12. + 3.*cl**4/8. - cl**5/(6.*ck))/z

    elif (z > ck-0.5*cl and z <= 0.5*ck): # f64(z)
        return z**5*16.*(3.*ak-3.+4.*al-5.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(1.-2.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
           z**2*20.*ck**2*(8.*ak*al-1.-6.*ak)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
           z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(1.+14.*ak-2.*al-12.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
           5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(32.*ak*al-1.-30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
           3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(1.+62.*ak-2.*al-60.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

    elif (z > 0.5*ck and z <= 0.5*cl): # f65(z)
        return z**5*16.*(2.*al-1.-ak-ak*al)/(3.*ck*cl) + \
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)

    elif (z > 0.5*cl and z <= cl-0.5*ck): # f66(z)
        return z**5*16.*(1.-3.*ak-2.*al+3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + \
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + \
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + 3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + 5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + 5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + 3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

    elif (z > cl-0.5*ck and z <= ck): # f67(z)
        return z**5*16.*(1.-3.*ak-3.*al+5.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(4.*ak-1.+al-4.*ak*al)/cl + z**3*10. + \
            z**2*20.*ck**2*(1.-10.*ak-al+10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
            z*5.*ck**3*(18.*ak-1.+al-18.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.-34.*ak-al+34.*ak*al)*0.5/cl + 5.*cl**3*(1+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
            ck**5*(66.*ak-1.+al-66.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

    elif (z > ck and z <= 0.5*(ck+cl)): # f68(z)
        return z**5*16.*(1.-ak-3.*al+3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(al-1.)/cl + z**3*10. + \
            z**2*20.*ck**2*(1.+6.*ak-al-6.*ak*al)/(3.*cl) + z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + \
            z*5.*ck**3*(al-1.-14.*ak+14.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + \
            5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.+30.*ak-al-30.*ak*al)*0.5/cl + 5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + \
            ck**5*(al-1.-62.*ak+62.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + 5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - 3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

    elif (z > 0.5*ck+0.5*cl and z <= cl): # f69(z) (= f49(z))
        return z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + \
            z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-al-2.*ak)/cl + z**3*20.*(ak+al-2.*ak*al) + \
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck) + \
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + \
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + \
            3.*ck**4*(-16.*ak-al+2.*ak*al)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)

    elif (z > cl and z <= ck+0.5*cl): # f610(z) ( = f410(z))
        return z**5*16.*(ak+al-3.*ak*al)/(3.*ck*cl) + \
            z**4*8.*(4.*ak*al-ak-2.*al)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + z**3*20.*(ak+al-2.*ak*al) +\
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + z**2*20.*cl**2*(ak+8.*al-10.*ak*al)/(3.*ck) + \
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(18.*ak*al-ak-16.*al)/ck + \
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + 5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak+32.*al-34.*ak*al)*0.5/ck + \
            3.*ck**4*(2.*ak*al-al-16.*ak)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + 10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + 3.*cl**4*(2.*ak*al-16.*al-ak)/(4.*z) + cl**5*(66.*ak*al-ak-64.*al)/(6.*ck*z)

    elif (z > ck+0.5*cl and z <= cl+0.5*ck): # f611(z) ( = f411(z))
        return z**5*16.*(al-ak*al)/(3.*ck*cl) + \
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + z**3*20.*al + \
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + \
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + \
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al +48.*cl**4*(al-ak*al)/ck + \
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - 3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)

    elif (z >= cl+0.5*ck and z <= cl+ck): # f612(z) ( = f412(z))
        return ak*al*(z**5*16./(3.*ck*cl) - z**4*16/ck - z**4*16./cl + z**3*40.+ \
                      z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - \
                      z*80.*(ck**2 + ck**3/cl + cl**2 + cl**3/ck) + \
                      80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck + \
                      80.*ck**2*cl**2/(3.*z) - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) - 24.*cl**4/z - 32.*cl**5/(3.*ck*z))

    else:
        return 0.



def f7(z,ak,al,cc):
    """
    For the case where ck = cl, calls Eq. (33)
    of Gaspari et al., (2006). See Appendix C Equations (C.1) and (C.2)
    for coefficients

    Input: 
    z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
        need z >= 0
    ak, al - (scalar) corresponding scalars for xk, xl
    cc - (scalar) cut-off length parameter, ck = cl, only one 
         input necessary
    """
    rr = z/cc
    
    if (rr >= 0. and rr <= 0.5): # f1(z) in (C.1)
        return rr**5*16.*(4.*ak-3.-7.*ak*al+4.*al)/3. + \
            rr**4*16.*(1.+2.*ak*al-ak-al) + \
            rr**3*10.*(1.+8.*ak*al-2.*ak-2.*al) - \
            rr**2*40.*(1.+8.*ak*al-ak-al)/3. + \
            2.+44.*ak*al+3.*ak+3.*al

    elif (rr > 0.5 and rr <= 1.): #  f2(z) in (C.1)
        return rr**5*16.*(1.+5.*ak*al-3.*ak-3.*al)/3. + \
            rr**4*8.*(5.*ak-2.-8.*ak*al+5.*al) + rr**3*10. + \
            rr**2*20.*(2.+20.*ak*al-11.*ak-11.*al)/3. + \
            rr*5.*(13.*ak-4.-36.*ak*al+13.*al) + \
            (16.+204.*ak*al-35.*ak-35.*al)/2. + \
            (29.*ak-8.-84.*ak*al+29.*al)/(12.*rr)

    elif (rr > 1. and rr <= 1.5): # f3(z) in (C.1)
        return rr**5*16.*(ak-3.*ak*al+al)/3. + \
            rr**4*8.*(8.*ak*al-3.*ak-3.*al) + \
            rr**3*20.*(ak-2.*ak*al+al) + \
            rr**2*20.*(9.*ak-20.*ak*al+9.*al)/3. + \
            rr*5.*(44.*ak*al-27.*ak-27.*al) + \
            0.5*(189.*ak-244.*ak*al+189.*al) + \
            (460.*ak*al-243.*ak-243.*al)/(12.*rr)

    elif (rr > 1.5 and rr <= 2.): # f4(z) in (C.1)
        return 8.*ak*al*(rr**5*2. - rr**4*12. + rr**3*15. + rr**2*40. - rr*120. + 96. - 16./rr)/3.

    else:
        return 0.
