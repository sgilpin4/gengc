"""
gengc1d.py

Shay Gilpin 
created January 24, 2023
last updated on April 24, 2023

** ------------------------------------------------------------------ **
        ** CITE THE CORRESPONDING PUBLICATION AND CODE DOI **
(paper): Gilpin, S., Matsuo, T., and Cohn, S.E. (2023). A generalized,
         compactly-supported correlation function for data assimilation
         applications. Q. J. Roy. Meteor. Soc.
(code): Gilpin, S. A Generalized Gaspari-Cohn Correlation Function
        (see Github repo for DOI)
** ------------------------------------------------------------------ **


Object-oriented construction of the Generalized Gaspari-Cohn
correlation function from Gilpin et al. (2023) on one-dimensional domains, 
optimized for construction of correlation matrices in Python.

GenGC can be constructed in two different ways:
PWConstantGenGC - to construct GenGC where subdomains are large
                  (contain multiple grid cells) and values of a
                  and c are constant over eac subdomain. Continuity
                  of these correlations are not guarenteed.
ContinuousGenGC - to construct GenGC where a and c are continuous functions
                  over the full domain. Value of a and c for each grid cells 
                  are average values computed using numerical integration 
                  methods, defined by Eq.(25) of Gilpin et al. (2023).

Two additional classes are used in either one or both GenGC constructions:
Norm - computation of norms in 1D (Euclidean distance and chordal distance,
       for instance)
AverageValue - computation of the average values of functions over 
               a specified interval. Can implement different integration 
               techniques according to Eq.(25) of Gilpin et al. (2023).


Evaluation and Exmaples

PWConstantGenGC:
This assumes that the number of subdomains are less than the number 
of grid cells and a and c are constant over each subdomain.
Construction of GenGC is done by first defining the partitions using
GenGCPartition from gengc_conv.py. Each partition takes in a value of a, c,
and a conditional function that defines the subdomain with respect 
to the input grid x:

Ex: grid = np.linspace(0,1,201) # includes boundaries of grid cells
    midpoint_grid = 0.5*(grid[:-1]+grid[1:]) # midpoint of the grid cells
    p1 = GenGCPartition(-0.25,0.6,lambda x: x<0.5)
    p2 = GenGCPartition(0.25,0.2,lambda x: x>=0.5)

Then, the GenGC object is initialized. To create the object, one must include
the list of partitions, the Norm object used to compute normed distances 
between grid points, and the spatial grid that is being evaluated:

Ex: fcn = PWConstantGenGC([p1,p2],Norm(midpoint_grid,midpoint_grid,len(grid)).dist_outer_1d,midpoint_grid)

To output the full correlation matrix as 2D numpy array, it simply needs to be called:

Ex: mat = fcn()

-------------------------------------------
ContinuousGenGC:

Given a discretized grid that defines the boundaries of the subregions,
continuous functions a(x), c(x), and a norm, ContinuousGenGC can construct
the corresponding correlation matrix as a 2D numpy array.

The ContinuousGenGC object requires several inputs that can be computed 
either inline our before this object is initialized. First, we compute 
the average values of the functions a(x) and c(x) (see AverageValue 
below for description)

Ex: grid = np.linspace(0,2.*np.pi,201) # boundaries of grid cells
    midpoint_grid = 0.5*(grid[:-1]+grid[1:]) # midpoint of the grid cells
    a_average_value = AverageValue(lambda x: 0.25*np.sin(x)+0.5,
                                   grid[:-1],grid[1:],4).midpoint_rule
    c_average_value = AverageValue(lambda x: 0.25*np.pi - 0.15*np.pi*np.sin(x),
                                   grid[:-1],grid[1:],4).midpoint_rule

The ContinuousGenGC object can be constructed as follows (see ContinuousGenGC
below for more details)

Ex: obj = ContinuousGenGC(Norm(midpoint_grid,midpoint_grid,2*np.pi).dist_chord_1d, a_average_value, c_average_value)

To extract the matrix itself requires calling the object,

Ex: mat = obj()
-------------------------------------------
"""

from itertools import product
import numpy as np
from typing import Callable, List
from gengc_conv import GenGC, GenGCContinuous

class Norm:
    """
    Norm class for computing norms between two sets of points
    Inputs:
    points1, points2 - (1d array) values in domain in which norm
                       will be evaluated
    domain_length - (scalar) for norms that require domain length 
                    (e.g. chordal distance). Default is None if not
                    specified.
    """
    def __init__(self,points1,points2,domain_length=None):
        self.points1 = points1
        self.points2 = points2
        self.domain_length = domain_length

    def dist_outer_1d(self):
        """
        Euclidean distance function in 1dimension
        """
        return np.abs(np.subtract.outer(self.points1,
                                        self.points2))

    def dist_chord_1d(self):
        """
        Computes chordal distance between points1 and
        points 2. Domain_length (scalar) is the length of the
        domain_length = x_right - x_left
        2sin(2*pi*|points1-points2|/domain_length
        """
        return 2.*np.sin(np.pi*
                         np.abs(np.subtract.outer(self.points1,self.points2))
                         /self.domain_length)


class AverageValue:
    """
    Average Value computation class that computes the average value 
    of the function f(x) over the 1D interval [xl,xr]:

             \int_{xl}^xr f(x)dx 
     f_avg =  -------------------        (1)
             \int_{xl}^{xr} 1 dx
    Initialize by specifying a function fcn, left and right endpoints of 1D
    domain, then can choose different numerical integration schemes for the
    computation. Note that the numerator and denominator in Eq. (1) 
    are evaluated using the same numerical integration scheme.
    """

    def __init__(self,fcn,left_endpoint,right_endpoint,num_subintervals=1):
        """
        Input:
        fcn - (function) function that will be integrated
        left_endpoint - (1d array) left endpoints of intervals for 
                        integration
        right_endpoint - (1d array) right endpoints of intervals
                         for integration
        num_subintervals - (integer) number of subintervals used to 
                          evaluate the numerical integration scheme.
                          If not specified, default is 1.
        """
        self.fcn = fcn
        self.left_endpoint = left_endpoint
        self.right_endpoint = right_endpoint
        self.interval_length = right_endpoint - left_endpoint
        self.num_subintervals = num_subintervals

    def midpoint_rule(self):
        """
        Approximates the numerator and denominator of Eq. (1) with the 
        midpoint rule.
        num_subintervals (integer) assigns the number of subintervals
        when approximating Eq. (1). The default of 1 implies is equivalent to
        evaluating the function at the midpoint of the interval. Assumes
        each subinterval is of equal length. The computation here 
        is simplified after Eq.(1) is evaluated using the midpoint rule.
        Output - 1D array of average values over each interval defined by the
                 left and right endpoint inputs
        """
        
        value_array = np.tile(self.left_endpoint,(self.num_subintervals,1)).transpose() +np.tile(self.interval_length,(self.num_subintervals,1)).transpose()*0.5*(1.+2.*np.tile(np.arange(self.num_subintervals),(len(self.left_endpoint),1)))/self.num_subintervals

        return np.sum(self.fcn(value_array),axis=1)/self.num_subintervals
        
        
        
class ContinuousGenGC:
    """
    For constructing GenGC correlation matrix from continuous functions for 
    a and c on the one-dimensional domain defined by grid.
    """

    def __init__(self,pairwise_dist,a_average_values,c_average_values):
        """
        Input:
        pairwise_dist - (object) object that computes the distance of the 
                        values desired when called
        a_average_values, c_average_vales - (object) ak and ck 
                       values for each subregion (denoted by index k) computed in
                       AverageValue class, when called below
        """
        self.pairwise_dist = pairwise_dist
        self.a_average_values = a_average_values
        self.c_average_values = c_average_values

    def __call__(self):
        """
        Output:
        Correlation matrix (2d array) constructed using the GenGC correlation
        function.
        """
        zz = self.pairwise_dist()
        a1, a2 = np.meshgrid(self.a_average_values(),self.a_average_values())
        c1, c2 = np.meshgrid(self.c_average_values(),self.c_average_values())
        ret = np.zeros_like(zz)
        order_mask1 = c1 < c2
        order_mask2 = c1 >= c2
        
        ckcl_masks1, gengc_fcns1 = GenGCContinuous().make_fi_cmasks(c1,c2)
        for c_mask, fcn in zip(ckcl_masks1, gengc_fcns1):
            new_mask = np.logical_and(order_mask1,c_mask)
            if zz[new_mask].size != 0: # does not evaluate if empty
                ret[new_mask] = fcn(zz[new_mask],a1[new_mask],a2[new_mask],
                                    c1[new_mask],c2[new_mask])
        # # order must be switched here to take care of
        # # when c1 >= c2
        ckcl_masks2, gengc_fcns2 = GenGCContinuous().make_fi_cmasks(c2,c1)
        # # now loops through to construct the function
        for c_mask, fcn in zip(ckcl_masks2, gengc_fcns2):
            new_mask = np.logical_and(order_mask2,c_mask)
            if zz[new_mask].size != 0: # does not evaluate if empty
                ret[new_mask] = fcn(zz[new_mask],a2[new_mask],a1[new_mask],
                                    c2[new_mask],c1[new_mask])

        return ret
        

class Partition:
    """
    Creates partition object where the partition is done on
    the spatial domain (indicated by indices k,l in correlation
    function). Users specify predicate function which defines
    functionally the partition on the domain with respect
    to the spatial variables. The partition object
    is then used as the primary function in evaluating the correlation
    function along with specified input parameters a and c (which
    are tied to the partition). With partition defined using predicate
    function, we can then determine which spatial variables
    are in which partition here, rather than the user doing so.
    """

    def __init__(self, predicate: Callable[[np.ndarray], np.ndarray]):
        """
        Initializes the predicate function, which is callable, takes
        in a numpy array and spits out a numpy array
        would be better to subclass the partition to have gaspari cohn partition
        and foar partition
        """
        self.predicate = predicate

    def contains(self, points: np.ndarray) -> np.ndarray:
        """
        Contains operation that returns a bool if the points
        are in the partition.
        """
        return self.predicate(points)

class GenGCPartition(Partition):
    def __init__(self, a, c, predicate: Callable[[np.ndarray], np.ndarray]):
        """
        Subclass of partition used to construct the GenGC correlation 
        function when the subregions contain a signficant number of 
        points.
        """
        self.predicate = predicate
        self.a = a
        self.c = c


def gengc_part_dict(partitions:Partition):
    """
    Builds partition dictionary needed to evaluate GenGC convolution.
    Each partition is represented by an index k, therefore the convolution
    B_{kl} is evaluated with respect to a pair of partitions k and l.
    Since B_{kl} is symmetric, so will the dictionary. 
    The dictionary is built so that per pair of partitions,
    which each have their own a and c value, will determine
    which function in B_{kl} should be called based on the ck and
    cl values.
    Input:
    partitions - (list of length >=1) list of partition objects
                 ready to be paired
    can use the partial function here instead from functools
    """
    keys = [(p1,p2) for p1 in partitions for p2 in partitions] # builds pairs of partition
    # # determines correct B_{kl} function for each pair of partition
    fcns = [GenGC(pp[0].a,pp[1].a,pp[0].c,pp[1].c).determine_f() for pp in keys]
    # # could put this all into one line use dictionary comprehension (like list comprehension above)
    return dict(zip(keys,fcns)) # creates dictionary

class PWConstantGenGC:
    """
    Insert discussion here
    """
    def __init__(self, partition_list, pairwise_dist,points):
        """
        part_dict is the partition dictionary which deteremines
        which functions should be evaluated based on the pair of
        partitions. Recall each partition as a corresponding
        a and c value, where for each pair of partions (k,l)
        the relationship between ck,cl determine which part of the
        correlation function should be evaluated.
        Input:
        partition_list - (list) list of partition objects that define
                         the subregions (by functions) and the values
                         of a and c on those subregions
        pariwise_dist - (Norm object) Norm object that holds the 
                        information to compute distances between 
                        points
        points - (1d array) grid values used to compute pairwise_dist
                 that define the indices of the output correlation 
                 matrix. 
        """
        self.part_dict = gengc_part_dict(partition_list)
        self.pairwise_dist = pairwise_dist
        self.points = points

    def __call__(self):
        """
        For each pair of partitions, determines which points 
        are in each partition, then creates an array of bools
        over full difference array for values that are in the pair
        of partitions.
        domain length is an optional argument needed for chordal distance
        norm
        """
        ret = np.zeros((len(self.points), len(self.points)))
        dist = self.pairwise_dist()

        for (p1, p2), func in self.part_dict.items():
            p1_mask = p1.contains(self.points)
            p2_mask = p2.contains(self.points)
            mask = np.logical_and.outer(p1_mask, p2_mask)
            ret[mask] = func(dist[mask])

        return ret
