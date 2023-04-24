"""
run_experiments.py

created by Shay Gilpin, Jan 25, 2023
last updated on April 24, 2023

** ------------------------------------------------------------------ **
        ** CITE THE CORRESPONDING PUBLICATION AND CODE DOI **
(paper): Gilpin, S., Matsuo, T., and Cohn, S.E. (2023). A generalized,
         compactly-supported correlation function for data assimilation
         applications. Q. J. Roy. Meteor. Soc.
(code): Gilpin, S. A Generalized Gaspari-Cohn Correlation Function
        (see Github repo for DOI)
** ------------------------------------------------------------------ **

Constructs GenGC correlation matrices for a 1D domain, 
following the examples from Gilpin et al., (2023), Figures 2, 4--6.

Run the script as 

python run_experiments.py 

and will generate Figures 2, 4-6 in sequence according to main() below. 
"""

import numpy as np
import gengc1d as gc
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import colors

color_trio=["grey","coral","gold"]

def main():
    """
    """
    make_figure_2()
    make_figure_4()
    make_figure_5()
    make_figure_6()

def make_figure_2():
    """
    creates Figure 2
    """
    midpoint_grid1, d_mat, d_afcn, d_cfcn, d_plot_dict = discontinuous_case_hyptan()
    midpoint_grid2, c_mat, c_afcn, c_cfcn, c_plot_dict = continuous_case_hyptan()
    make_cts_disc_plot([midpoint_grid1,midpoint_grid2],[d_mat,c_mat],
                       [d_afcn,c_afcn],[d_cfcn,c_cfcn],[50,100,170],
                       [d_plot_dict,c_plot_dict])
    return

def make_figure_4():
    """
    creates Figure 4
    """

    midpoint_grid, mat, afcn, cfcn, plot_dict = continuous_case_sine1()
    make_single_plot(midpoint_grid,mat,afcn,cfcn,[50,125,150],plot_dict)

    return

def make_figure_5():
    """
    creates Figure 5
    """
    midpoint_grid, mat, afcn, cfcn, plot_dict = continuous_case_sine2()
    make_single_plot(midpoint_grid,mat,afcn,cfcn,[50,125,150],plot_dict)

    return

def make_figure_6():
    """
    creates Figure 6
    """
    midpoint_grid, mat, afcn, cfcn, plot_dict = continuous_case_sine2()
    foarL = {"value":0.25*np.pi,
             "string": "$\\pi/4$",
             "fn_string": "pi4"}
    foar_mat = np.exp(-1.*gc.Norm(midpoint_grid,midpoint_grid,
                                  2.*np.pi).dist_chord_1d()/foarL["value"])
    hadamarad_product_plot(midpoint_grid,mat,afcn,cfcn,
                           [50,125,150],plot_dict,foar_mat,foarL)
    
    return

def continuous_case_hyptan(nx=201):
    """
    Example used to generate the bottom row of
    Fig. 2 of Gilpin et al (2023).
    """
    grid = np.linspace(0.,1.,nx)
    grid_length = grid[-1]-grid[0]
    midpoint_grid = 0.5*(grid[:-1]+grid[1:])
    domain = "$(0,1)$"

    def cfcn(x):
        return 0.5-0.25*np.tanh(20.*(x-0.5))
    cstring = "$0.5 - 0.25\\tanh(20(r-0.5))$"

    def afcn(x):
        return 0.5*np.tanh(25.*(x-0.5))
    astring = "$0.5\\tanh(25(r-0.5))$"

    gengc_fcn = gc.ContinuousGenGC(gc.Norm(midpoint_grid,
                                           midpoint_grid,
                                           grid_length).dist_outer_1d,
                                   gc.AverageValue(afcn,grid[:-1],
                                                   grid[1:],1).midpoint_rule,
                                   gc.AverageValue(cfcn,grid[:-1],
                                                   grid[1:],1).midpoint_rule)
    plot_dict = {"astring":astring,
                 "cstring":cstring,
                 "type":"Continuous",
                 "domain":domain,
                 "norm":"Euclidean",
                 "function version": "Generalized Gaspari-Cohn",
                 "white_index": 0,
        }

    return midpoint_grid, gengc_fcn(), afcn, cfcn, plot_dict

def discontinuous_case_hyptan(nx=201):
    """
    Example that accompanies continuous_case_hyptan, show in
    the top row of Fig. 2 of Gilpin et al (2023).
    """
    grid = np.linspace(0.,1.,nx)
    grid_length = grid[-1]-grid[0]
    midpoint_grid = 0.5*(grid[:-1]+grid[1:])
    domain = "$(0,1)$"
    a1 = -0.25
    c1 = 0.6
    a2 = 0.25
    c2 = 0.2

    def afcn(x):
        return np.piecewise(x, [np.logical_and(x >= 0., x <0.5),x>=0.5],
                            [a1,a2])
    def cfcn(x):
        return np.piecewise(x, [np.logical_and(x >= 0., x <0.5),x>=0.5],
                            [c1,c2])

    p1 = gc.GenGCPartition(a1,c1,lambda x: x<0.5)
    p2 = gc.GenGCPartition(a2,c2,lambda x: x>=0.5)

    gengc_fcn = gc.PWConstantGenGC([p1,p2],gc.Norm(midpoint_grid,
                                                   midpoint_grid,
                                                   grid_length).dist_outer_1d,
                                   midpoint_grid)

    plot_dict = {"astring": f"{a1}, {a2}",
                 "cstring": f"{c1}, {c2}",
                 "type": "Discontinuous",
                 "jump_loc": [0.5],
                 "domain":"$(0,1)$",
                 "norm": "Euclidean",
                 "white_index": 0,
                 "function version": "Generalized Gaspari-Cohn",
        }

    return midpoint_grid, gengc_fcn(), afcn, cfcn, plot_dict

def continuous_case_sine1(nx=201):
    """
    To create Fig. 4 in Gilpin et al. (2023)
    """
    grid = np.linspace(0.,2.*np.pi,nx)
    grid_length = 2.*np.pi
    midpoint_grid = 0.5*(grid[:-1]+grid[1:])
    domain = "$S_1^1$"

    def cfcn(x):
        return 0.25*np.pi - 0.15*np.pi*np.sin(x)
    cstring = "$0.25\\pi - 0.15\\pi\\sin(r)$"

    def afcn(x):
        return 0.25*np.sin(x)+0.5
    astring = "$0.25\\sin(r)+0.5$"

    white_index = 0
    inner_mesh = 1

    gengc_fcn = gc.ContinuousGenGC(gc.Norm(midpoint_grid,
                                           midpoint_grid,
                                           grid_length).dist_chord_1d,
                                   gc.AverageValue(afcn,grid[:-1],
                                                   grid[1:],
                                                   inner_mesh).midpoint_rule,
                                   gc.AverageValue(cfcn,grid[:-1],
                                                   grid[1:],
                                                   inner_mesh).midpoint_rule)
    plot_dict = {"astring":astring,
                 "cstring":cstring,
                 "type":"Continuous",
                 "domain":domain,
                 "norm":"Chordal Distance",
                 "function version": "Generalized Gaspari-Cohn",
                 "white_index": white_index,
        }

    return midpoint_grid, gengc_fcn(), afcn, cfcn, plot_dict


def continuous_case_sine2(nx=201):
    """
    Creates information for Fig. 5 of Gilpin et al. (2023).
    """
    grid = np.linspace(0.,2.*np.pi,nx)
    grid_length = 2.*np.pi
    midpoint_grid = 0.5*(grid[:-1]+grid[1:])
    domain = "$S_1^1$"

    def cfcn(x):
        return 0.25*np.pi - 0.15*np.pi*np.cos(x-0.2*np.pi)
    cstring = "$0.25\\pi - 0.15\\pi\\cos(r-0.2\\pi)$"

    def afcn(x):
        return 0.5*np.sin(3*x)+0.25
    astring = "$0.5\\sin(3r)+0.25$"

    white_index = 61
    inner_mesh = 1

    gengc_fcn = gc.ContinuousGenGC(gc.Norm(midpoint_grid,
                                           midpoint_grid,
                                           grid_length).dist_chord_1d,
                                   gc.AverageValue(afcn,grid[:-1],
                                                   grid[1:],
                                                   inner_mesh).midpoint_rule,
                                   gc.AverageValue(cfcn,grid[:-1],
                                                   grid[1:],
                                                   inner_mesh).midpoint_rule)
    plot_dict = {"astring":astring,
                 "cstring":cstring,
                 "type":"Continuous",
                 "domain":domain,
                 "norm":"Chordal Distance",
                 "function version": "Generalized Gaspari-Cohn",
                 "white_index": white_index,
        }

    return midpoint_grid, gengc_fcn(), afcn, cfcn, plot_dict

def make_single_plot(points,mat,afcn,cfcn,idxs,plot_dict):
    """
    Generates the plots for a single correlation matrix case
    (as defined above).
    Input: points (1d array) -  grid in which correlation is constructed
           mat (2d array) - correlation matrix
           afcn, cfcn (function) - functions for a and c
           plot_dict (dictionary) - dictionary needed to specify aspects of
                                    the ouput figure
    """
    fig = plt.figure(figsize=(12,4.4))
    spec = gridspec.GridSpec(ncols=3,nrows=1,
                             width_ratios=[0.8,1.1,1],figure=fig)
    ac_fcn_ax = fig.add_subplot(spec[0])
    mat_ax = fig.add_subplot(spec[1])
    row_ax = fig.add_subplot(spec[2])

    ac_fcn_ax.set_title("a(r) = {0} \n c(r) = {1}".format(plot_dict["astring"],
                                                          plot_dict["cstring"]))
    ac_fcn_ax.plot(points,np.zeros_like(points),"dimgrey",lw=1)
    if plot_dict["type"] == "Discontinuous":
        plot_discont_fcn(ac_fcn_ax,afcn,points,plot_dict["jump_loc"],
                         "k","solid",2.5,label="a(x)")
        plot_discont_fcn(ac_fcn_ax,cfcn,points,plot_dict["jump_loc"],
                         "dimgrey","dashed",2.5,label="c(x)")
    else:
        ac_fcn_ax.plot(points,afcn(points),"k",lw=2.5,label="a(r)")
        ac_fcn_ax.plot(points,cfcn(points),"dimgrey",ls="dashed",
                       lw=2.5,label="c(r)")
    ac_fcn_ax.set_xlim(points[0],points[-1])
    ac_fcn_ax.legend(fontsize=12,loc="upper left")
    ac_fcn_ax.set_xlabel("$r$",fontsize=12)

    cmap = "jet"
    jetcmap = plt.get_cmap("jet",256)
    newcolors = jetcmap(np.linspace(0.,1.,256))
    white = np.array([255/256,255/256,255/256,1])
    newcolors[plot_dict["white_index"],:]=white
    newcmap = colors.ListedColormap(newcolors)
    
    im = mat_ax.imshow(mat,cmap=newcmap,aspect="auto")
    mat_ax.set_title("Correlation Matrix")
    fig.colorbar(im,ax=mat_ax)

    xx,yy = np.meshgrid(np.arange(len(points)),np.arange(len(points)))
    levels = [-0.004,-0.003,-0.002,-0.001,0.001,0.002,0.003,0.004]
    levels_ls = ["-"]*len(levels)
    levels_colors = ["white"]*len(levels)
    mat_ax.contour(xx,yy,mat,levels=levels,linestyles=levels_ls,
                   colors=levels_colors,linewidths=1.5)

    mat_ax.hlines(idxs,0,len(points)-1,colors=color_trio,
                  lw=1.75)

    for idx,color in zip(idxs,color_trio):
        if plot_dict["type"] == "Discontinuous":
            plot_discont_array(row_ax,mat[idx],points,plot_dict["jump_loc"],
                                   "indexes",color,
                                   "solid",2.5,label=f"Row {idx}")
        else:
            row_ax.plot(mat[idx],lw=2.5,color=color,label=f"Row {idx}")
    row_ax.plot([0,len(points)],[0.,0.],"k",ls="dotted",lw=1.5)
    row_ax.set_title("Select 1D Correlations (rows)")
    row_ax.set_xlim(0,len(points))
    row_ax.set_ylim([-0.32,1.05])

    plt.suptitle("{4} on {0} for {3} a(r), c(r) ({1} grid points, {2} Norm)".format(plot_dict["domain"],len(points),plot_dict["norm"],plot_dict["type"],plot_dict["function version"]),fontsize=15)
    fig.subplots_adjust(left=0.048,right=0.974,bottom=0.12,top=0.8,
                       wspace=0.2,hspace=0.2)
    plt.show()
    return

def make_cts_disc_plot(points,mats,afcns,cfcns,idxs,plot_dicts):
    """
    Modification of make_single_plot to generate Fig. 2 of Gilpin et
    al (2023) which includes a two cases for a and c.
    Input: points (list of 1d arrays) -  grid in which correlation is constructed
           mats (list of 2d array) - correlation matrix
           afcns, cfcns (list of function) - functions for a and c
           idxs (list of integers) - list of row indices to plot
           plot_dicts (list of dictionary) - dictionary needed to specify
                                             aspects of the ouput figure
    
    """
    fig = plt.figure(figsize=(11.5,8))
    spec = gridspec.GridSpec(ncols=3,nrows=2,
                             width_ratios=[0.8,1.1,1],
                             height_ratios=[1,1],figure=fig)
    
    for row, points,mat, afcn, cfcn, plot_dict in zip([1,0],points,mats,afcns,
                                                      cfcns,plot_dicts):
        
        ac_fcn_ax = fig.add_subplot(spec[row,0])
        mat_ax = fig.add_subplot(spec[row,1])
        row_ax = fig.add_subplot(spec[row,2])

        ac_fcn_ax.set_title("a(r) = {0} \n c(r) = {1}".format(plot_dict["astring"],
                                                              plot_dict["cstring"]))
        ac_fcn_ax.plot(points,np.zeros_like(points),"dimgrey",lw=1)
        if plot_dict["type"] == "Discontinuous":
            plot_discont_fcn(ac_fcn_ax,afcn,points,plot_dict["jump_loc"],
                             "k","solid",2.5,label="a(r)")
            plot_discont_fcn(ac_fcn_ax,cfcn,points,plot_dict["jump_loc"],
            "dimgrey","dashed",2.5,label="c(r)")
        else:
            ac_fcn_ax.plot(points,afcn(points),"k",lw=2.5,label="a(r)")
            ac_fcn_ax.plot(points,cfcn(points),"dimgrey",ls="dashed",
                           lw=2.5,label="c(r)")
        ac_fcn_ax.set_xlim(points[0],points[-1])
        ac_fcn_ax.legend(fontsize=12)
        im = mat_ax.imshow(mat,cmap="jet",aspect="auto")
        
        
        mat_ax.set_title("Correlation Matrix")
        mat_ax.hlines(idxs,0,len(points)-1,colors=color_trio,lw=1.5)
        fig.colorbar(im,ax=mat_ax)

        for idx,color in zip(idxs,color_trio):
            if plot_dict["type"] == "Discontinuous":
                plot_discont_array(row_ax,mat[idx],points,plot_dict["jump_loc"],
                                   "indexes",color,
                                   "solid",2.5,label=f"Row {idx}")
            else:
                row_ax.plot(mat[idx],color=color,lw=2.5,label=f"Row {idx}")
        row_ax.plot([0,len(points)],[0.,0.],"K",lw=1.5,ls="dotted")
        row_ax.set_title("Select 1D Correlations (rows)")
        row_ax.set_xlim(0,len(points))
        
    plt.suptitle("Generalized Gaspari-Cohn on {0} ({1} grid points, {2} Norm)".format(plot_dict["domain"],len(points),plot_dict["norm"]),fontsize=15)
    fig.subplots_adjust(left=0.058,right=0.97,bottom=0.057,top=0.87,
                        wspace=0.275,hspace=0.31)
    plt.show()
    return

def hadamarad_product_plot(points,mat,afcn,cfcn,
                           idxs,plot_dict,foar_mat,foarL):
    """
    Generates Fig. 6 of Gilpin et al (2023) which constructs the Hadamard
    product of Fig. 5 with the First Order Autoregressive correlation function
    Input: points (1d array) -  grid in which correlation is constructed
           mat (2d array) - correlation GenGC matrix
           afcn, cfcn (function) - functions for a and c
           idxs (list of integers) - row indices to plot
           plot_dict (dictionary) - dictionary needed to specify aspects of
                                    the ouput figure
           foar_mat (2d array) FOAR correlation matrix
           foarL (dictionary) - dictionary that includes scalar value
                                for lengthscale parameter, string version and 
                                filename information (if needed)
    """
    fig = plt.figure(figsize=(13,4))
    spec = gridspec.GridSpec(ncols=4,nrows=1,
                             width_ratios=[1,1,1.25,1],figure=fig)
    foar_mat_ax = fig.add_subplot(spec[0])
    gengc_mat_ax = fig.add_subplot(spec[1])
    schur_mat_ax = fig.add_subplot(spec[2])
    row_ax = fig.add_subplot(spec[3])

    # # builds the hadamarad product
    schur_mat = foar_mat*mat
    jetcmap = plt.get_cmap("jet",256)
    newcolors = jetcmap(np.linspace(0.,1.,256))
    white = np.array([255/256,255/256,255/256,1])
    newcolors[plot_dict["white_index"],:]=white
    newcmap = colors.ListedColormap(newcolors)

    im1 = foar_mat_ax.imshow(foar_mat,cmap=newcmap,aspect="auto",
                             vmax = np.max(mat), vmin=np.min(mat))
    foar_mat_ax.set_title("FOAR Correlation Matrix")
    im2 = gengc_mat_ax.imshow(mat,cmap=newcmap,aspect="auto",
                              vmax = np.max(mat), vmin=np.min(mat))
    xx,yy = np.meshgrid(np.arange(len(points)),np.arange(len(points)))
    levels = [-0.003,-0.0025,-0.002,-0.0015,-0.001,0.001,0.002,0.003]
    levels_ls = ["-"]*len(levels)
    levels_colors = ["white"]*len(levels)
    gengc_mat_ax.contour(xx,yy,mat,levels=levels,linestyles=levels_ls,
                         colors=levels_colors,linewidths=1.5)
    gengc_mat_ax.set_title("GenGC Correlation Matrix")
    im3 = schur_mat_ax.imshow(schur_mat,cmap=newcmap,aspect="auto",
                              vmax = np.max(mat), vmin=np.min(mat))
    schur_mat_ax.contour(xx,yy,schur_mat,levels=levels,linestyles=levels_ls,
                         colors=levels_colors,linewidths=1.5)
    fig.colorbar(im3,ax=schur_mat_ax)
    
    schur_mat_ax.set_title(f"Hadamard Product\n Correlation Matrix")
    
    for idx,color in zip(idxs,color_trio):
        if plot_dict["type"] == "Discontinuous":
            plot_discont_array(row_ax,schur_mat[idx],points,plot_dict["jump_loc"],
                                   "indexes",color,
                                   "solid",2.5,label=f"Row {idx}")
        else:
            row_ax.plot(foar_mat[idx],color="k",lw=1.5,ls="dashed")
            row_ax.plot(schur_mat[idx],lw=2.5,label=f"Row {idx}",color=color)
    row_ax.plot([0,len(points)],[0.,0.],"k",lw=1.5,ls="dotted")
    schur_mat_ax.hlines(idxs,0,len(points)-1,colors=color_trio,lw=1.5)
    foar_mat_ax.hlines(idxs,0,len(points)-1,colors="k",
                       lw=1.5,linestyles="dashed")
    row_ax.set_title("Select 1D Correlations (rows)")
    row_ax.set_xlim(0,len(points))

    plt.suptitle("Hadamard Product of FOAR ($L_0=${4}) and Generalized Gaspari-Cohn on {0} for {3} a(r) = {5},\n c(r) = {6} ({1} grid points, {2} Norm)".format(plot_dict["domain"],len(points),plot_dict["norm"],plot_dict["type"],foarL["string"],plot_dict["astring"],plot_dict["cstring"]),fontsize=15)
    fig.subplots_adjust(left=0.048,right=0.984,bottom=0.117,top=0.725,
                       wspace=0.2,hspace=0.2)
    plt.show()
    return

def plot_discont_fcn(plot_ax,fcn,points,jump_loc,
                     color,ls,lw,label):
    """
    Plots the jump discontinuity without connecting the line between the two.
    Can resolve this by inserting a np.nan wherever it needs to be.
    Input fcn must be a function that can be evaluated
    """
    jump_idx_list = [np.where(points <= jump)[0][-1] for jump in jump_loc]
    new_points = np.insert(points,jump_idx_list,[np.nan]*len(jump_idx_list))
    new_fcn_vals = np.insert(fcn(points),jump_idx_list,[np.nan]*len(jump_idx_list))
    plot_ax.plot(new_points,new_fcn_vals,color=color,ls=ls,lw=lw,label=label)
    return

def plot_discont_array(plot_ax,array, points, jump_loc, plot_grid,
                       color,ls,lw,label):
    """
    splits the array at the jump discontinuity and preps for plotting
    array - np.array to split
    points - np.array that matches jump_loc
    """
    jump_idx_list = [np.where(points <= jump)[0][-1] for jump in jump_loc]
    if plot_grid == "points":
        new_points = np.insert(points,jump_idx_list,[np.nan]*len(jump_idx_list))
    else:
        new_points = np.insert(np.arange(len(points),dtype=np.float),jump_idx_list,[np.nan]*len(jump_idx_list))
        
    new_fcn_vals = np.insert(array,jump_idx_list,[np.nan]*len(jump_idx_list))
    plot_ax.plot(new_points,new_fcn_vals,color=color,ls=ls,lw=lw,label=label)

    return

main()
    
