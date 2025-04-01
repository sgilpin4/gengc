! Generalize Gaspari-Cohn (GenGC) Correlation Function
! Fortran Version
! Shay Gilpin
! created July 29th, 2024
! last updated August 6th, 2024
!
! ** ------------------------------------------------------------------ **
 !        ** CITE THE CORRESPONDING PUBLICATION AND CODE DOI **
! (paper): Gilpin, S., Matsuo, T., and Cohn, S.E. (2023). A generalized,
!          compactly-supported correlation function for data assimilation
!          applications. Q. J. Roy. Meteor. Soc.
!          https://doi.org/10.1002/qj.4490
! (code): Gilpin, S. A Generalized Gaspari-Cohn Correlation Function
!         https://doi.org/10.5281/zenodo.7859258
!         (which includes the link to the Github Repo
! ** ------------------------------------------------------------------ **
! Note: this is modified from the original python version gengc.py
! Implementation of the GenGC Correlation Function
! defined in Gilpin et al. (2023), "A generalied, compactly-supported
! correlation function for data assimilation applications" in QJRMS.

! This script implements Eq.(36) in Appendix B of Gilpin et al. (2023). 
! The function gengc() constructs GenGC acting on a vector of normed
! distances c. The functions f1, f2, ... , f6 define each case of
! Eq. (36) and implement the coefficients in Tables 1-19. Normalization
! is done in the main gengc() subroutine for f1, ... , f6. The function
! f7 is the case where ck = cl, given in Eq.(33) of Gaspari et al. (2006),
! QJRMS.

! For this implementation, ak, al, ck, and cl are either scalars or 
! 1D arrays of length n. These values are computed before evaluating
! GenGC using gengc. For example, consider the case where a and c 
! are computed from continuous functions on the unit circle:
! Ex:
!     grid = [(i*2.*Pi/201., i=0,200)]
!     avalues = 0.25*Sin(grid)+0.5
!     cvalues = 0.25*Pi - 0.15*Pi*Sin(grid)

! To obtain one row of the correlation matrix (say the first row),
! we can call gengc() after computing the normed distances (here chordal
! distance) with the first grid cell:
! Ex:
!     z = 2.*Sin(Abs(grid(1)-grid))
!     corr_row1 = gengc(z,avalues(1),avalues,cvalues(1),cvalues,
!                        201)
! To recover the full correlation matrix (as a 2D array), loop through
! the full set of grid points to construct each row.

! This formulation works well to construct 2D correlations with respect
! to a fixed grid point (e.g. the examples in Section 4.2 of Gilpin et 
! al, 2023).

module gengc_function
  implicit none
contains
  subroutine gengc(z,ak,al,ck,cl,n,out)
    ! Constructs the GenGC correlation function. Normalization of
    ! functions f1, ... , f6 are done here rather than in the 
    ! specific functions themselves.

    ! Input:
    ! z - (1d array) normed distance between fixed grid point xk 
    !     and xl, where xl is a 1d array of length n;
    !     z = ||xk - xl||
    ! ak, al - values of a corresponding to grid points xk, xl used in input;
    !          ak is scalar, al must be the same shape as xl
    ! ck, cl - values of cut-off lengths corresponding to grid points
    !          xk, xl; ck is a scalar, cl must be the same shape as xl
    ! n - length of z, al, cl arrays, and is size of output array

    ! Output:
    ! out - (1D array of length n) GenGC correlation function
    
    integer, intent(in) :: n
    real*8, intent(in) :: ak, ck
    real*8, intent(in) :: z(n), al(n), cl(n)
    ! declaring other variables
    real*8 :: nk, nl, ckcl
    integer :: j
    ! output array
    real*8, intent(out) :: out(n)

    nk = 2.+44.*ak**2+6.*ak
    do j = 1,n
       nl = 2.+44.*al(j)**2+6.*al(j)
       ckcl = ck**3*cl(j)**3
       if (ck == cl(j)) then ! calls Eq.(33) in Gaspari et al., (2006)
          call f7(z(j),ak,al(j),ck,out(j)) ! /sqrt(nk*nl)
          out(j) = out(j)/sqrt(nk*nl)
       else if (ck < cl(j)) then
          if ((ck >= 0.) .and. (ck <= 0.25*cl(j))) then
             call f1(z(j),ak,al(j),ck,cl(j),out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((ck >= 0.25*cl(j)) .and. (ck <= cl(j)/3.)) then 
             call f2(z(j),ak,al(j),ck,cl(j),out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((ck >= cl(j)/3.) .and. (ck <= 0.5*cl(j))) then
             call f3(z(j),ak,al(j),ck,cl(j),out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((ck >= 0.5*cl(j)) .and. (ck <= 2.*cl(j)/3.)) then 
             call f4(z(j),ak,al(j),ck,cl(j),out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((ck >= 2.*cl(j)/3.) .and. (ck <= 0.75*cl(j))) then
             call f5(z(j),ak,al(j),ck,cl(j),out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((ck >= 0.75*cl(j)) .and. (ck <= cl(j))) then
             call f6(z(j),ak,al(j),ck,cl(j),out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else 
             out(j) = 0.
          endif
          ! here, must swap ck and cl indicies in the function
       else if (ck > cl(j)) then
          if ((cl(j) >=0) .and. (cl(j) <= 0.25*ck)) then
             call f1(z(j),al(j),ak,cl(j),ck,out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((cl(j) >= 0.25*ck) .and. (cl(j) <= ck/3.)) then
             call f2(z(j),al(j),ak,cl(j),ck,out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((cl(j) >= ck/3.) .and. (cl(j) <= 0.5*ck)) then
             call f3(z(j),al(j),ak,cl(j),ck,out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((cl(j) >= 0.5*ck) .and. (cl(j) <= 2.*ck/3.)) then
             call f4(z(j),al(j),ak,cl(j),ck,out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((cl(j) >= 2.*ck/3.) .and. (cl(j) <= 0.75*ck)) then
             call f5(z(j),al(j),ak,cl(j),ck,out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else if ((cl(j) >= 0.75*ck) .and. (cl(j) <= ck)) then 
             call f6(z(j),al(j),ak,cl(j),ck,out(j))
             out(j) = out(j)/sqrt(nk*nl*ckcl)
          else 
             out(j) = 0.
          endif
       endif
    enddo
    
    
  end subroutine gengc
  subroutine f1(z,ak,al,ck,cl,out)
    implicit none
    ! Outputs the first function f1 defined for
    ! 0 < ck <= cl/4
    ! given in Tables 1-3 of Gilpin et al (2023)
    ! Input:
    ! z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
    !     need z >= 0
    ! ak, al - (scalar) corresponding scalars for zk, zl
    ! ck, cl - (scalar) corresponding cut-off length parameters for zk, zl
    real*8 :: z, ak, al, ck, cl
    real*8, intent(out) :: out

    if ((z >= 0.) .and. (z <= 0.5*ck)) then ! f11(z)
       out = z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(1.-al)/cl + &
            z**2*40.*ck**2*(al-1.-6.*ak+6.*ak*al)/(3.*cl) + &
            5.*ck**3*(1.+14.*ak) + 3.*ck**4*(al-1.-30.*ak+30.*ak*al)/cl
       
    else if ((z > 0.5*ck) .and. (z <= ck)) then  !  f12(z)
       out = z**5*32.*(ak*al-ak)/(3.*ck*cl) + &
            z**4*32.*(ak-ak*al)/cl + &
            z**2*320.*ck**2*(ak*al-ak)/(3.*cl) + &
            z*10.*ck**3*(-1.+2.*ak+al-2.*ak*al)/cl + & 
            5.*ck**3*(1.+14.*ak)+96.*ck**4*(ak*al-ak)/cl + &
            ck**5*(-1.+2.*ak+al-2.*ak*al)/(3.*z*cl)
       
    else if ((z > ck) .and. (z <= 0.5*cl - ck)) then ! f13(z)
       out = z*10.*ck**3*(-1.-14.*ak+al+14.*ak*al)/cl + &
            5.*ck**3*(1.+14.*ak) + &
            ck**5*(-1.-62.*ak+al+62*ak*al)/(3.*cl*z)
       
    else if ((z > 0.5*cl - ck) .and. (z <= 0.5*cl - 0.5*ck)) then ! f14(z)
       out = z**5*16.*(2.*ak*al-ak)/(3.*ck*cl) + &
            z**4*8*(ak-2.*ak*al)/ck + z**4*16.*(2*ak*al-ak)/cl + &
            z**3*(20.*ak-40.*ak*al) + &
            z**2*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) +& 
            z*40.*ck**2*(2.*ak*al-ak) + &
            z*10.*ck**3*(-1.-6.*ak+al-2.*ak*al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + &
            5.*ck**3*(1.+6.*ak+16.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + &
            5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/(2.*ck) + &
            12.*ck**4*(2.*ak*al - ak)/z + &
            ck**5*(-1.-30.*ak+al-2.*ak*al)/(3.*cl*z) + &
            10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6*ck*z)
        
    else if ((z > 0.5*cl-0.5*ck) .and. (z <= 0.5*cl)) then ! f15(z)
       out = z**5*16.*(ak-1.+2.*al-2.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + &
            z**3*(10.-20.*al) + &
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - &
            z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + &
            z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + &
            3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)/(2.*cl) + &
            5.*cl**3*(1.-2.*al)*0.5 + &
            3.*cl**4*(ak-1.+2.*al-2.*ak*al)/(2.*ck) + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - &
            ck**5*(1.+62.*ak)/(6.*cl*z) +&
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*cl) .and. (z <= 0.5*(cl+ck))) then  ! f16(z)
       out = z**5*16.*(1.-ak-2.*al+2.*ak*al)/(3.*ck*cl) + & 
            z**4*8.*(ak-1.+2.*al-2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + &
            z**3.*(10.-20.*al) + &
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-2.*al+2.*al*ak)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - &
            z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + &
            z*5.*cl**3*(ak-1.+2.*al-2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + &
            3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(1.-ak-2.*al+2.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - &
            ck**5*(1.+62.*ak)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(ak-1.+2.*al-2.*ak*al)/(6.*ck*z)

    else if ((z > 0.5*(ck+cl)) .and. (z <= 0.5*cl+ck)) then ! f17(z)
       out = z**5*16.*(ak-2.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(2.*ak*al-ak)/ck + z**4*16.*(2.*ak*al-ak)/cl +& 
            z**3*(20.*ak-40.*ak*al) + & 
            z**2.*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-2.*ak*al)/(3.*ck) + &
            z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(2.*ak*al-al-8.*ak)/cl + &
            z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(2.*ak*al-ak)/ck + &
            10.*ck**3*(4.*ak+al+6.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + &
            5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(ak-2.*ak*al)*0.5/ck + &
            12.*ck**4*(2.*ak*al-ak)/z + ck**5*(2.*ak*al-al-32.*ak)/(3.*cl*z) + &
            10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(2.*ak*al-ak)/(6.*ck*z)
       
    else if ((z > 0.5*cl+ck) .and. (z <= cl-ck)) then ! f18(z)
       out = 10.*ck**3*(al+14.*ak*al) - z*10.*ck**3*(al+14.*ak*al)/cl - &
            ck**5*(al+62.*ak*al)/(3.*z*cl)
       
    else if ((z > cl-ck) .and. (z <= cl -0.5*ck)) then ! f19(z)
       out = -16.*z**5*ak*al/(3.*ck*cl) + & 
            z**4*16.*ak*al/ck - z**4*16.*ak*al/cl + & 
            z**3*40.*ak*al + & 
            z**2*160.*ck**2*ak*al/(3.*cl) - z**2*160.*cl**2*ak*al/(3.*ck) + & 
            z*80.*cl**3*ak*al/ck - z*80.*ck**2*ak*al - &
            z*10.*ck**3*(al+6.*ak*al)/cl - z*80.*cl**2*ak*al + & 
            10.*ck**3*(al+6.*ak*al) + 48.*ck**4*ak*al/cl + 80.*cl**3*ak*al - &
            48.*cl**4*ak*al/ck + &
            80.*ck**2*cl**2*ak*al/(3.*z) - 24.*ck**4*ak*al/z - &
            ck**5*(al+30.*ak*al)/(3.*cl*z) - 24.*cl**4*ak*al/z + &
            32.*cl**5*ak*al/(3.*ck*z)

    else if ((z > cl-0.5*ck) .and. (z <= cl)) then ! f110(z)
       out = z**5*16.*(ak*al-al)/(3.*ck*cl) + & 
            z**4*16.*(al-ak*al)/ck - z**4*8.*al/cl + & 
            z**3*20.*al + & 
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + &
            z**2*160.*cl**2*(ak*al-al)/(3.*ck) + & 
            z*80.*cl**3*(al-ak*al)/ck - z*10.*ck**2*(al+6.*ak*al) - &
            z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + & 
            5.*ck**3*(14.*ak*al+al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + &
            40.*cl**3*al + 48.*cl**4*(ak*al-al)/ck + &
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - &
            3.*ck**4*(al+30.*ak*al)/(4.*z) - &
            ck**5*(al+62.*ak*al)/(6.*cl*z) - &
            12.*cl**4*al/z + 32.*cl**5*(al-ak*al)/(3.*ck*z)
       
    else if ((z > cl) .and. (z <= cl + 0.5*ck)) then ! f111(z)
       out = z**5*16.*(al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + & 
            z**3*20.*al + & 
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + &
            z**2*160.*cl**2*(al-ak*al)/(3.*ck) + &
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - &
            z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + &
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + &
            40.*cl**3*al + 48.*cl**4*(al-ak*al)/ck + & 
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - &
            3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - &
            12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)
       
    else if ((z > cl+0.5*ck) .and. (z <= cl+ck)) then ! f112(z)
       out = ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - z**4*16./cl + &
            40.*z**3 + z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - &
            z*80.*ck**2 - z*80.*ck**3/cl - z*80.*cl**2 - z*80.*cl**3/ck + &
            80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck - &
            24.*ck**4/z - 32.*ck**5/(3.*cl*z) + 80.*ck**2*cl**2/(3.*z) - &
            24.*cl**4/z - 32*cl**5/(3.*ck*z))
    else 
       out = 0.
    endif
  end subroutine f1

  subroutine f2(z,ak,al,ck,cl,out)
    implicit none
    ! Outputs the second function f2 defined for
    ! cl/4 <= ck <= cl/3
    ! from Tables 4-6 of Gilpin et al. (2023)
    ! Input:
    ! z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
    !     need z >= 0
    ! ak, al - (scalar) corresponding scalars for xk, xl
    ! ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    real*8 :: z, ak, al, ck, cl
    real*8, intent(out) :: out

    if ((z >= 0) .and. (z <= 0.5*ck)) then ! f21(z)
       out = z**5*32.*(ak+al-ak*al-1.)/(3.*ck*cl) + &
            z**4*16.*(1.-al)/cl + &
            z**2*40.*ck**2*(al-6.*ak-1.+6.*ak*al)/(3.*cl) + &
            5.*ck**3*(1.+14.*ak) + &
            3.*ck**4*(al-1.-30.*ak+30.*ak*al)/cl
       
    else if ((z > 0.5*ck) .and. (z <= 0.5*cl-ck)) then ! f22(z)
       out = z**5*32.*(ak*al-ak)/(3.*ck*cl) + z**4*32.*(ak-ak*al)/cl + &
            z**2*320.*ck**2*(ak*al-ak)/(3.*cl) + &
            z*10.*ck**3*(2.*ak-1.+al-2.*ak*al)/cl + &
            5.*ck**3*(1.+14.*ak) + 96.*ck**4*(ak*al-ak)/cl + &
            ck**5*(2.*ak-1.+al-2.*ak*al)/(3.*cl*z)
       
    else if ((z > 0.5*cl-ck) .and. (z <= ck)) then ! f23(z)
       out = z**5*16.*(4.*ak*al-3.*ak)/(3.*ck*cl) + &
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16.*ak/cl &
            + z**3*(20.*ak-40.*ak*al) + &
            z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) &
            - z**2*160.*ck**2*ak/(3.*cl) + &
            z*40.*ck**2*(2.*ak*al-ak) + &
            z*10.*ck**3*(10.*ak-1.+al-18.*ak*al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + &
            5.*ck**3*(1.+6.*ak+16.*ak*al) - 48.*ck**4*ak/cl + &
            5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + &
            12.*ck**4*(2.*ak*al-ak)/z + &
            ck**5*(34.*ak-1.+al-66.*ak*al)/(3.*cl*z) + &
            10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)
       
    else if ((z > ck) .and. (z <= 0.5*cl - 0.5*ck)) then ! f24(z)
       out = z**5*16*(2.*ak*al-ak)/(3.*ck*cl) + &
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16*(2.*ak*al-ak)/cl + &
            z**3*(20.*ak-40.*ak*al) + &
            z**2*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) + &
            z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(al-1.-6.*ak-2.*ak*al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + &
            5.*ck**3*(1.+6.*ak+16.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + &
            5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + &
            12.*ck**4*(2.*ak*al-ak)/z + &
            ck**5*(al-1.-30.*ak-2.*ak*al)/(3.*cl*z) + &
            10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*cl-0.5*ck) .and. (z <= 0.5*cl)) then ! f25(z)
       out = z**5*16.*(ak-1.+2.*al-2.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + &
            z**3*(10.-20.*al) + &
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - &
            z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + &
            z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + &
            3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - &
            ck**5*(1.+62.*ak)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*cl) .and. (z <= 0.5*cl+0.5*ck)) then ! f26(z)
       out = z**5*16.*(1.-ak-2.*al+2.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+2.*al-2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + &
            z**3*(10.-20.*al) + &
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-2.*al+2.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - &
            z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + &
            z*5.*cl**3*(ak-1.+2.*al-2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + &
            3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(1.-ak-2.*al+2.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - &
            ck**5*(1.+62.*ak)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(ak-1.+2.*al-2.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*ck+0.5*cl) .and. (z <= cl-ck)) then ! f27(z)
       out = z**5*16.*(ak-2.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(2.*ak*al-ak)/ck + z**4*16.*(2.*ak*al-ak)/cl + &
            z**3*(20.*ak-40.*ak*al) + &
            z**2*160.*ck**2*(ak-2.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-2.*ak*al)/(3.*ck) + &
            z*40.*ck**2*(2.*ak*al-ak) + z*10.*ck**3*(2.*ak*al-8.*ak-al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(2.*ak*al-ak)/ck + &
            10.*ck**3*(4.*ak+al+6.*ak*al) + 48.*ck**4*(ak-2.*ak*al)/cl + &
            5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(ak-2.*ak*al)*0.5/ck + &
            12.*ck**4*(2.*ak*al-ak)/z + ck**5*(2.*ak*al-al-32.*ak)/(3.*cl*z) + &
            10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(2.*ak*al-ak)/(6.*ck*z)
       
    else if ((z > cl-ck) .and. (z <= ck+0.5*cl)) then ! f28(z)
       out = z**5*16.*(ak-3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(4.*ak*al-ak)/ck + z**4*16.*(ak*al-ak)/cl + z**3*20.*ak + &
            z**2*160.*ck**2*(ak-ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-10.*ak*al)/(3.*ck) + &
            z*10.*ck**3*(10.*ak*al-al-8.*ak)/cl - z*40.*ck**2*ak -&
            z*10.*cl**2*(ak+6.*ak*al) + z*5.*cl**3*(18.*ak*al-ak)/ck + &
            10.*ck**3*(4.*ak+al-2.*ak*al) + 48.*ck**4*(ak-ak*al)/cl + &
            5.*cl**3*(ak+14.*ak*al) + 3.*cl**4*(ak-34.*ak*al)*0.5/ck + &
            ck**5*(34.*ak*al-al-32.*ak)/(3.*cl*z) -12.*ck**4*ak/z + &
            10.*ck**2*cl**2*(ak+6.*ak*al)/(3.*z) - &
            3.*cl**4*(ak+30.*ak*al)/(4.*z) + cl**5*(66.*ak*al-ak)/(6.*ck*z)
       
    else if ((z > ck +0.5*cl) .and. (z <= cl-0.5*ck)) then ! f29(z)
       out = -z**5*16.*ak*al/(3.*ck*cl) + &
            z**4*16.*ak*al/ck - z**4*16.*ak*al/cl + z**3*40.*ak*al + &
            z**2*160.*ck**2*ak*al/(3.*cl) - z**2*160.*cl**2*ak*al/(3.*ck) + &
            z*80.*cl**3*ak*al/ck - z*80.*ck**2*ak*al - &
            z*10.*ck**3*(al+6.*ak*al)/cl - z*80.*cl**2*ak*al + &
            10.*ck**3*(al+6.*ak*al) + 48.*ck**4*ak*al/cl +&
            80.*cl**3*ak*al - 48.*cl**4*ak*al/ck + &
            80.*ck**2*cl**2*ak*al/(3.*z) - 24.*ck**4*ak*al/z - &
            ck**5*(al+30.*ak*al)/(3.*cl*z) -24.*cl**4*ak*al/z + &
            32.*cl**5*ak*al/(3.*ck*z)
       
    else if ((z > cl-0.5*ck) .and. (z <= cl)) then ! f210 ( = f110(z))
       out = z**5*16.*(ak*al-al)/(3.*ck*cl) + &
            z**4*16.*(al-ak*al)/ck - z**4*8.*al/cl + &
            z**3*20.*al + z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + &
            z**2*160.*cl**2*(ak*al-al)/(3.*ck) + &
            z*80.*cl**3*(al-ak*al)/ck - z*10.*ck**2*(al+6.*ak*al) - &
            z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + &
            5.*ck**3*(14.*ak*al+al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + &
            40.*cl**3*al + 48.*cl**4*(ak*al-al)/ck + &
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - &
            3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - &
            12.*cl**4*al/z + 32.*cl**5*(al-ak*al)/(3.*ck*z)
       
    else if ((z > cl) .and. (z <= cl + 0.5*ck)) then ! f211 ( = f111(z))
       out = z**5*16.*(al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + &
            z**3*20.*al + z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + &
            z**2*160.*cl**2*(al-ak*al)/(3.*ck) + &
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - &
            z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + &
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + &
            40.*cl**3*al + 48.*cl**4*(al-ak*al)/ck + &
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - &
            3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - &
            12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)
       
    else if ((z > cl+0.5*ck) .and. (z <= cl+ck)) then ! f212 ( = f112(z))
       out = ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - z**4*16./cl + &
            40.*z**3 + z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - &
            z*80.*ck**2 - z*80.*ck**3/cl - z*80.*cl**2 - &
            z*80.*cl**3/ck + 80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + &
            48.*cl**4/ck - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) + &
            80.*ck**2*cl**2/(3.*z) - 24.*cl**4/z - 32*cl**5/(3.*ck*z))
       
    else 
       out = 0.
    endif
  end subroutine f2

  subroutine f3(z,ak,al,ck,cl,out)
    implicit none
    real*8 :: z, ak, al, ck, cl
    real*8, intent(out) :: out
    ! Outputs the second function f3 defined for
    ! cl/3 <= ck <= cl/2
    ! from Tables 7-9 of Gilpin et al. (2023)
    ! Input:
    ! z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
    !     need z >= 0
    ! ak, al - (scalar) corresponding scalars for xk, xl
    ! ck, cl - (scalar) corresponding cut-off length parameters for xk, xl

    if ((z >= 0.) .and. (z <= 0.5*cl-ck)) then ! f31(z)
       out = z**5*32.*(ak-1.-ak*al+al)/(3.*ck*cl) + z**4*16.*(1.-al)/cl + & 
            z**2*40.*ck**2*(al-1.+6.*ak*al-6.*ak)/(3.*cl) + 5.*ck**3*(1.+14.*ak) + & 
            3.*ck**4*(al-1.-30.*ak+30.*ak*al)/cl
       
    else if ((z > 0.5*cl-ck) .and. (z <= 0.5*ck)) then ! f32(z)
       out = z**5*16.*(ak+2.*al-2.)/(3.*ck*cl) + &
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16*(1.-ak-al+2.*ak*al)/cl + &
            z**3*(20.*ak-40.*ak*al) + &
            z**2*40.*ck**2*(al-2.*ak-1.-2.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) + &
            z*40.*ck**2*(2.*ak*al-ak) + z*80.*ck**3*(ak-2.*ak*al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + &
            5.*ck**3*(1.+6.*ak+16.*ak*al) + &
            3.*ck**4*(al-1.-14.*ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + &
            3.*cl**4*(2.*ak*al-ak)*0.5/ck + &
            12.*ck**4*(2.*ak*al-ak)/z + 32.*ck**5*(ak-2.*ak*al)/(3.*cl*z) + &
            10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*ck) .and. (z <= 0.5*cl-0.5*ck)) then ! f33(z)
       out = z**5*16*(4.*ak*al-3.*ak)/(3.*ck*cl) + &
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16.*ak/cl + z**3*20.*ak*(1.-2.*al) + &
            z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) - z**2*160.*ck**2*ak/(3.*cl) + &
            z*40.*ck**2*(2.*ak*al-ak) + &
            z*10.*ck**3*(10.*ak-1+al-18.*ak*al)/cl + z*10.*cl**2*(2.*ak*al-ak) + &
            z*5.*cl**3*(ak-2.*ak*al)/ck + &
            5.*ck**3*(1.+6.*ak+16.*ak*al) - 48.*ck**4*ak/cl + &
            5.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)*0.5/ck + &
            12.*ck**4*(2.*ak*al-ak)/z + &
            ck**5*(34.*ak-1+al-66.*ak*al)/(3.*cl*z) + &
            10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*cl-0.5*ck) .and. (z <= ck)) then ! f34(z)
       out = z**5*16.*(2.*al-ak-1.)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + &
            z**4*8.*(4.*ak-1+2.*al-4.*ak*al)/cl + &
            z**3*10.*(1.-2.*al) + &
            z**2*20.*ck**2*(1.-10.*ak-2.*al+4.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + &
            z*5.*ck**3*(18.*ak-1.-32.*ak*al)/cl + z*5.*cl**2*(2.*al-1.) + &
            z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + &
            3.*ck**4*(1.-34.*ak-2.*al+4.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + &
            ck**5*(66.*ak-1.-128.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)
       
    else if ((z > ck) .and. (z <= 0.5*cl)) then ! f35(z)
       out = z**5*16.*(ak-1.+2.*al-2.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + &
            z**3*(10.-20.*al) + &
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - &
            z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + &
            z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + &
            3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - &
            ck**5*(1.+62.*ak)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

    else if ((z > 0.5*cl) .and. (z <= cl-ck)) then ! f36(z)
       out = z**5*16.*(1.-ak-2.*al+2.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+2.*al-2.*ak*al)/ck + z**4*8.*(2.*al-1.)/cl + &
            z**3*(10.-20.*al) + &
            z**2*20.*ck**2*(1.+6.*ak-2.*al-12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-2.*al+2.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) - &
            z*5.*ck**3*(1.+14.*ak)/cl + z*5.*cl**2*(2.*al-1.) + &
            z*5.*cl**3*(ak-1.+2.*al-2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + &
            3.*ck**4*(1.+30.*ak-2.*al-60.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(1.-ak-2.*al+2.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) - &
            ck**5*(1.+62.*ak)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(ak-1.+2.*al-2.*ak*al)/(6.*ck*z)

    else if ((z > cl-ck) .and. (z <= 0.5*cl+0.5*ck)) then ! f37(z)
       out = z**5*16.*(1.-ak-2.*al+ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(2.*al-1.-2.*ak*al)/cl + &
            z**3*10.*(1.-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(1.+6.*ak-2.*al-4.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(16.*ak*al-1.-14.*ak)/cl + &
            z*5.*cl**2*(2.*al-1.-16.*ak*al) + &
            z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(1.+30.*ak-2.*al-28.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + &
            3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(64.*ak*al-1.-62.*ak)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

    else if ((z > 0.5*ck + 0.5*cl) .and. (z <= cl - 0.5*ck)) then ! f38(z)
       out = z**5*16.*(ak-3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(4.*ak*al-ak)/ck + z**4*16.*(ak*al-ak)/cl + &
            z**3*20.*ak + z**2*160.*ck**2*(ak-ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-10.*ak*al)/(3.*ck) + &
            z*10.*ck**3*(10.*ak*al-al-8.*ak)/cl - z*40.*ck**2*ak - &
            z*10.*cl**2*(ak+6.*ak*al) + z*5.*cl**3*(18.*ak*al-ak)/ck + &
            10.*ck**3*(al+4.*ak-2.*ak*al) + 48.*ck**4*(ak-ak*al)/cl + &
            5.*cl**3*(ak+14.*ak*al) + 3.*cl**4*(ak-34.*ak*al)*0.5/ck + &
            ck**5*(34.*ak*al-32.*ak-al)/(3.*cl*z) - 12.*ck**4*ak/z + &
            10.*ck**2*cl**2*(ak+6.*ak*al)/(3.*z) - &
            3.*cl**4*(ak+30.*ak*al)/(4.*z) + cl**5*(66.*ak*al-ak)/(6.*ck*z)

    else if ((z > cl-0.5*ck) .and. (z <= ck+0.5*cl)) then ! f39(z)
       out = z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + &
            z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + &
            z**3*20.*(ak+al-2.*ak*al) + &
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck)+ &
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + &
            z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + &
            5.*ck**3*(8.*ak+al-2.*ak*al) + &
            3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + &
            5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*ak*al-16.*ak-al)/(4.*z) + &
            ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + &
            10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)
       
    else if ((z > ck+0.5*cl) .and. (z <= cl)) then ! f310(z)
       out = z**5*16.*(ak*al-al)/(3.*ck*cl) + &
            z**4*16.*(al-ak*al)/ck - z**4*8.*al/cl + z**3*20.*al + &
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(ak*al-al)/(3.*ck) + &
            z*80.*cl**3*(al-ak*al)/ck - z*10.*ck**2*(al+6.*ak*al) - &
            z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + &
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl +&
            40.*cl**3*al + 48.*cl**4*(ak*al-al)/ck + &
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z)- &
            3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - &
            12.*cl**4*al/z + 32.*cl**5*(al-ak*al)/(3.*ck*z)

    else if ((z > cl) .and. (z <= cl + 0.5*ck)) then ! f311 ( = f111(z))
       out = z**5*16.*(al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + &
            z**3*20.*al + &
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + &
            z**2*160.*cl**2*(al-ak*al)/(3.*ck) + &
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - &
            z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + &
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + &
            40.*cl**3*al + 48.*cl**4*(al-ak*al)/ck + &
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - &
            3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - &
            12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)
                                                
    else if ((z > cl+0.5*ck) .and. (z <= cl+ck)) then ! f312 ( = f112(z))
       out = ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - &
            z**4*16./cl + 40.*z**3 + z**2*160.*ck**2/(3.*cl) + &
            z**2*160.*cl**2/(3.*ck) - z*80.*ck**2 - z*80.*ck**3/cl - &
            z*80.*cl**2 - z*80.*cl**3/ck + 80.*ck**3 + 48.*ck**4/cl +&
            80.*cl**3 + 48.*cl**4/ck - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) + &
            80.*ck**2*cl**2/(3.*z) - 24.*cl**4/z - 32*cl**5/(3.*ck*z))
    
    else
       out = 0.
    endif 
  end subroutine f3

  subroutine f4(z,ak,al,ck,cl,out)
    implicit none
    real*8 :: z, ak, al, ck, cl
    real*8, intent(out) :: out
    ! Outputs the second function f2 defined for
    ! cl/2 <= ck <= 2cl/3
    ! from Tables 10-13 from Gilpin et al (2023).
    ! Input:
    ! z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
    !     need z >= 0
    ! ak, al - (scalar) corresponding scalars for xk, xl
    ! ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    
    if ((z >= 0.) .and. (z <= ck-0.5*cl)) then ! f41(z)
       out = z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(ak-2.*ak*al)/ck + z**4*16.*(1.-2.*ak-al+4.*ak*al)/cl + &
            z**2*40.*ck**2*(2.*ak-1.+al-10.*ak*al)/(3.*cl) + &
            z**2*40.*cl**2*(2.*ak*al-ak)/(3.*ck) + &
            5.*ck**3*(1.-2.*ak+32.*ak*al) + &
            3.*ck**4*(2.*ak-1.+al-34.*ak*al)/cl + 10.*cl**3*(ak-2.*ak*al) + &
            3.*cl**4*(2.*ak*al-ak)/ck

    else if ((z > ck -0.5*cl) .and. (z <= 0.5*cl-0.5*ck)) then ! f42(z)
       out = z**5*16.*(ak-2.+2.*al)/(3.*ck*cl) + &
            z**4*8.*(ak-2.*ak*al)/ck + z**4*16.*(1.-ak-al+2.*ak*al)/cl + &
            z**3*(20.*ak-40.*ak*al) + &
            z**2*40.*ck**2*(al-1.-2.*ak-2.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(2.*ak*al-ak)/(3.*ck) + &
            z*40.*ck**2*(2.*ak*al-ak) + z*80.*ck**3*(ak-2.*ak*al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak) + z*5.*cl**3*(ak-2.*ak*al)/ck + &
            5.*ck**3*(1.+6.*ak+16.*ak*al) + &
            3.*ck**4*(al-1.-14.*ak-2.*ak*al)/cl + 5.*cl**3*(ak-2.*ak*al) + &
            3.*cl**4*(2.*ak*al-ak)*0.5/ck + &
            12.*ck**4*(2.*ak*al-ak)/z + 32.*ck**5*(ak-2.*ak*al)/(3.*cl*z) + &
            10.*ck**2*cl**2*(ak-2.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak)/(4.*z) + cl**5*(ak-2.*ak*al)/(6.*ck*z)

    else if ((z > 0.5*cl-0.5*ck) .and. (z <= 0.5*ck)) then ! f43(z)
       out = z**5*16.*(3.*ak-3.+4.*al-4.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8./cl + &
            z**3*(10.-20.*al) + &
            z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) - &
            z**2*20.*ck**2*(1.+6.*ak)/(3.*cl) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + &
            z*5.*ck**3*(1.+14.*ak-2.*al-28.*ak*al)/cl + &
            z*5.*cl**2*(2.*al-1.) + z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 - &
            3.*ck**4*(1.+30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + &
            3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + &
            ck**5*(1.+62.*ak-2.*al-124.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

    else if ((z > 0.5*ck) .and. (z <= cl-ck)) then ! f44(z)
       out = z**5*16.*(2.*al-1.-ak)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + &
            z**4*8.*(4.*ak-1.+2.*al-4.*ak*al)/cl + z**3*(10.-20.*al) + &
            z**2*20.*ck**2*(1.-10.*ak-2.*al+4.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + &
            z*5.*ck**3*(18.*ak-1.-32.*ak*al)/cl + z*5.*cl**2*(2.*al-1) + &
            z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 + &
            3.*ck**4*(1.-34.*ak-2.*al+4.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + &
            ck**5*(66.*ak-1.-128.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)

    else if ((z > cl-ck) .and. (z <= 0.5*cl)) then ! f45(z)
       out = z**5*16.*(2.*al-1.-ak-ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + &
            z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + &
            z**3*10.*(1.-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + &
            z*5.*cl**2*(2.*al-1.-16.*ak*al) + &
            z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + &
            3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + &
            cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*cl) .and. (z <= ck)) then ! f46(z)
       out = z**5*16.*(1.-3.*ak-2.*al+3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + &
            z**3*10.*(1.-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + &
            z*5.*cl**2*(2.*al-1.-16.*ak*al) + &
            z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + &
            3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + &
            cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

    else if ((z > ck) .and. (z <= cl-0.5*ck)) then ! f47(z)
       out = z**5*16.*(1.-ak-2.*al+ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(2.*al-1.-2.*ak*al)/cl + &
            z**3*10.*(1.-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(1.+6.*ak-2.*al-4.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(16.*ak*al-1.-14.*ak)/cl + &
            z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(1.+30.*ak-2.*al-28.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + &
            3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(64.*ak*al-1.-62.*ak)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + &
            cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)

    else if ((z > cl-0.5*ck) .and. (z <= 0.5*ck+0.5*cl)) then ! f48(z)
       out = z**5*16.*(1.-ak-3.*al+3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + &
            z**4*8.*(al-1.)/cl + z**3*10. + &
            z**2*20.*ck**2*(1.+6.*ak-al-6.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + &
            z*5.*ck**3*(al-1.-14.*ak+14.*ak*al)/cl - &
            z*5.*ck**2*(1.+6.*ak) - z*5*cl**2*(1.+6.*al) + &
            z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak)*0.5 + &
            3.*ck**4*(1.+30.*ak-al-30.*ak*al)*0.5/cl + &
            5.*cl**3*(1.+14.*al)*0.5 + &
            3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + &
            ck**5*(al-1.-62.*ak+62.*ak*al)/(6.*cl*z) - &
            3.*ck**4*(1.+30.*ak)/(8.*z) + &
            5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - &
            3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)

    else if ((z > 0.5*ck+0.5*cl) .and. (z <= cl)) then ! f49(z)
       out = z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + &
            z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-al-2.*ak)/cl + &
            z**3*20.*(ak+al-2.*ak*al) + &
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck) + &
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + &
            z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak-4.*al) + &
            z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + &
            5.*ck**3*(8.*ak+al-2.*ak*al) + &
            3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + &
            5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + &
            3.*ck**4*(-16.*ak-al+2.*ak*al)/(4.*z) + &
            ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + &
            10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + &
            cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)

    else if ((z > cl) .and. (z <= ck+0.5*cl)) then ! f410(z)
       out = z**5*16.*(ak+al-3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(4.*ak*al-ak-2.*al)/ck + &
            z**4*8.*(4.*ak*al-2.*ak-al)/cl + z**3*20.*(ak+al-2.*ak*al) +&
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak+8.*al-10.*ak*al)/(3.*ck) + &
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + &
            z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(18.*ak*al-ak-16.*al)/ck + &
            5.*ck**3*(8.*ak+al-2.*ak*al) + &
            3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + &
            5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak+32.*al-34.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*ak*al-al-16.*ak)/(4.*z) + &
            ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + &
            10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-16.*al-ak)/(4.*z) + &
            cl**5*(66.*ak*al-ak-64.*al)/(6.*ck*z)

    else if ((z > ck+0.5*cl) .and. (z <= cl+0.5*ck)) then ! f411(z)
       out = z**5*16.*(al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + z**3*20.*al + &
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + &
            z**2*160.*cl**2*(al-ak*al)/(3.*ck) + &
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - &
            z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + &
            5.*ck**3*(al+14.*ak*al) + &
            3.*ck**4*(al+30.*ak*al)*0.5/cl + 40.*cl**3*al + &
            48.*cl**4*(al-ak*al)/ck + &
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - &
            3.*ck**4*(al+30.*ak*al)/(4.*z) - &
            ck**5*(al+62.*ak*al)/(6.*cl*z) - 12.*cl**4*al/z + &
            32.*cl**5*(ak*al-al)/(3.*ck*z)

    else if ((z >= cl+0.5*ck) .and. (z <= cl+ck)) then ! f412(z)
       out = ak*al*(z**5*16./(3.*ck*cl) - z**4*16/ck - z**4*16./cl + z**3*40.+ &
            z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - &
            z*80.*(ck**2 + ck**3/cl + cl**2 + cl**3/ck) + &
            80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck + &
            80.*ck**2*cl**2/(3.*z) - 24.*ck**4/z - &
            32.*ck**5/(3.*cl*z) - 24.*cl**4/z - 32.*cl**5/(3.*ck*z))

    else 
       out = 0.
    endif

  end subroutine f4

  subroutine f5(z,ak,al,ck,cl,out)
    implicit none
    ! Outputs the second function f2 defined for
    ! 2cl/3 <= ck <= 3cl/4
    ! from Tables 14-16 of Gilpin et al (2023).
    ! Input:
    ! z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
    !     need z >= 0
    ! ak, al - (scalar) corresponding scalars for xk, xl
    ! ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    real*8 :: z, ak, al, ck, cl
    real*8, intent(out) :: out

    if ((z >= 0.) .and. (z <= 0.5*(cl-ck))) then ! f51(z)
       out = z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(ak-2.*ak*al)/ck + z**4*16.*(1.-2.*ak-al+4.*ak*al)/cl + &
            z**2*40.*ck**2*(2.*ak-1.+al-10.*ak*al)/(3.*cl) + &
            z**2*40.*cl**2*(2.*ak*al-ak)/(3.*ck) + &
            5.*ck**3*(1.-2.*ak+32.*ak*al) + &
            3.*ck**4*(al+2.*ak-1.-34.*ak*al)/cl + 10.*cl**3*(ak-2.*ak*al) + &
            3.*cl**4*(2.*ak*al-ak)/ck
       
    else if ((z > 0.5*(cl-ck)) .and. (z <= ck-0.5*cl)) then ! f52(z)
       out = z**5*16.*(4.*ak-3.+4.*al-6.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-2.*al)/ck + z**4*8.*(1.-2.*ak+4.*ak*al)/cl + &
            z**3*10.*(1.-2.*ak-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(2.*ak-1.-16.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(2.*al-1.)/(3.*ck) + &
            z*5.*(1.-2.*ak)*(1.-2.*al)*(ck**3/cl-ck**2-cl**2+cl**3/ck) + &
            5.*ck**3*(1.-2.*ak+2.*al+60.*ak*al)*0.5 + &
            3.*ck**4*(2.*ak-1.-64.*ak*al)*0.5/cl + &
            5.*cl**3*(1.+2.*ak-2.*al-4.*ak*al)*0.5 + 3.*cl**4*(2.*al-1.)*0.5/ck + &
            (1.-2.*ak)*(1.-2.*al)*(ck**5/(6.*cl) - 3.*ck**4/8. + &
            5.*ck**2*cl**2/12. - 3.*cl**4/8. + cl**5/(6.*ck))/z
       
    else if ((z > ck-0.5*cl) .and. (z <= cl-ck)) then ! f53(z)
       out = z**5*16.*(3.*ak-3.+4.*al-4.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+2.*ak*al)/ck + z**4*8./cl + z**3*10.*(1.-2.*al) + &
            z**2*20.*cl**2*(ak-1.+2.*al-2.*ak*al)/(3.*ck) - &
            z**2*20.*ck**2*(1.+6.*ak)/(3.*cl) + &
            z*5.*ck**2*(2.*al-1.-6.*ak+12.*ak*al) + &
            z*5.*ck**3*(1.+14.*ak-2.*al-28.*ak*al)/cl + z*5.*cl**2*(2.*al-1.) + &
            z*5.*cl**3*(1.-ak-2.*al+2.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al+28.*ak*al)*0.5 - &
            3.*ck**4*(1.+30.*ak)*0.5/cl + 5.*cl**3*(1.-2.*al)*0.5 + &
            3.*cl**4*(ak-1.+2.*al-2.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak+60.*ak*al)/(8.*z) + &
            ck**5*(1.+62.*ak-2.*al-124.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al-12.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.)/(8.*z) + cl**5*(1.-ak-2.*al+2.*ak*al)/(6.*ck*z)
       
    else if ((z > cl-ck) .and. (z <= 0.5*ck)) then ! f54(z)
       out = z**5*16.*(3.*ak-3.+4.*al-5.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(1.-2.*ak*al)/cl + &
            z**3*10.*(1.-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(8.*ak*al-1.-6.*ak)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(1.+14.*ak-2.*al-12.*ak*al)/cl + &
            z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(32.*ak*al-1.-30.*ak)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(1.+62.*ak-2.*al-60.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*ck) .and. (z <= 0.5*cl)) then ! f55(z)
       out = z**5*16.*(2.*al-1.-ak-ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + &
            z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + &
            z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*cl) .and. (z <= cl-0.5*ck)) then ! f56(z)
       out = z**5*16.*(1.-3.*ak-2.*al+3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + &
            z**3*10.*(1-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + &
            z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)
       
    else if ((z > cl-0.5*ck) .and. (z <= ck)) then ! f57(z)
       out = z**5*16.*(1.-3.*ak-3.*al+5.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + &
            z**4*8.*(4.*ak-1.+al-4.*ak*al)/cl + z**3*10. + &
            z**2*20.*ck**2*(1.-10.*ak-al+10.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + &
            z*5.*ck**3*(18.*ak-1.+al-18.*ak*al)/cl - &
            z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + &
            z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.-34.*ak-al+34.*ak*al)*0.5/cl + &
            5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + &
            ck**5*(66.*ak-1.+al-66.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + &
            5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - &
            3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)
       
    else if ((z > ck) .and. (z <= 0.5*(cl+ck))) then ! f58(z)
       out = z**5*16.*(1.-ak-3.*al+3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(al-1.)/cl &
            + z**3*10. + &
            z**2*20.*ck**2*(1.+6.*ak-al-6.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + &
            z*5.*ck**3*(al-1.-14.*ak+14.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - &
            z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.+30.*ak-al-30.*ak*al)*0.5/cl + &
            5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + &
            ck**5*(al-1.-62.*ak+62.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + &
            5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - &
            3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*ck+0.5*cl) .and. (z <= cl)) then ! f59(z) (= f49(z))
       out = z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + &
            z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-al-2.*ak)/cl + &
            z**3*20.*(ak+al-2.*ak*al) + &
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck) + &
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl +&
            z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + &
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl +&
            5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + &
            3.*ck**4*(-16.*ak-al+2.*ak*al)/(4.*z) + &
            ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + &
            10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)
       
    else if ((z > cl) .and. (z <= ck+0.5*cl)) then ! f510(z) ( = f410(z))
       out = z**5*16.*(ak+al-3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(4.*ak*al-ak-2.*al)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + &
            z**3*20.*(ak+al-2.*ak*al) +&
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak+8.*al-10.*ak*al)/(3.*ck) + &
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + &
            z*10.*cl**2*(2.*ak*al-ak-4.*al) + z*5.*cl**3*(18.*ak*al-ak-16.*al)/ck + &
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + &
            5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak+32.*al-34.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*ak*al-al-16.*ak)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + &
            10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-16.*al-ak)/(4.*z) + cl**5*(66.*ak*al-ak-64.*al)/(6.*ck*z)
       
    else if ((z > ck+0.5*cl) .and. (z <= cl+0.5*ck)) then ! f511(z) ( = f411(z))
       out = z**5*16.*(al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + z**3*20.*al + &
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + &
            z**2*160.*cl**2*(al-ak*al)/(3.*ck) + &
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - &
            z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + &
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + &
            40.*cl**3*al +48.*cl**4*(al-ak*al)/ck + &
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - &
            3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - &
            12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)
       
    else if ((z >= cl+0.5*ck) .and. (z <= cl+ck)) then ! f512(z) ( = f412(z))
       out = ak*al*(z**5*16./(3.*ck*cl) - z**4*16/ck - z**4*16./cl + z**3*40.+ &
            z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - &
            z*80.*(ck**2 + ck**3/cl + cl**2 + cl**3/ck) + &
            80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck + &
            80.*ck**2*cl**2/(3.*z) - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) &
            - 24.*cl**4/z - 32.*cl**5/(3.*ck*z))
       
    else 
       out =  0.
    endif
    
  end subroutine f5

  subroutine f6(z,ak,al,ck,cl,out)
    implicit none
    ! Outputs the sixth function f6 defined for
    ! 3cl/4 <= ck <= cl
    ! from tables 17-19 of Gilpin et al. (2023).
    ! Input:
    ! z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
    !     need z >= 0
    ! ak, al - (scalar) corresponding scalars for xk, xl
    ! ck, cl - (scalar) corresponding cut-off length parameters for xk, xl
    real*8 :: z, ak, al, ck, cl
    real*8, intent(out) :: out
    
    if ((z >= 0.) .and. (z <= 0.5*(cl-ck))) then ! f61(z)
       out = z**5*32.*(ak-1.+al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(ak-2.*ak*al)/ck + z**4*16.*(1.-2.*ak-al+4.*ak*al)/cl + &
            z**2*40.*ck**2*(2.*ak-1+al-10.*ak*al)/(3.*cl) + &
            z**2*40.*cl**2*(2.*ak*al-ak)/(3.*ck) + &
            5.*ck**3*(1.-2.*ak+32.*ak*al) + 3.*ck**4*(2.*ak-1.+al-34.*ak*al)/cl + &
            10.*cl**3*(ak-2.*ak*al) + 3.*cl**4*(2.*ak*al-ak)/ck 
       
    else if ((z > 0.5*(cl-ck)) .and. (z <= cl-ck)) then ! f62(z)
       out = z**5*16.*(4.*ak-3.+4.*al-6.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-2.*al)/ck + z**4*8.*(1.-2.*ak+4.*ak*al)/cl + &
            z**3*10.*(1.-2.*al-2.*ak+4.*ak*al) + &
            z**2*20.*ck**2*(2.*ak-1.-16.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(2.*al-1.)/(3.*ck) + &
            z*5.*(2.*ak-1)*(1.-2.*al)*(ck**2 - ck**3/cl + cl**2 - cl**3/ck) + &
            5.*ck**3*(1.-2.*ak+2.*al+60.*ak*al)*0.5 + &
            3.*ck**4*(2.*ak-1.-64.*ak*al)*0.5/cl + &
            5.*cl**3*(1.+2.*ak-2.*al-4.*ak*al)*0.5 + 3.*cl**4*(2.*al-1.)*0.5/ck + &
            (2.*ak-1.)*(1.-2.*al)*(3.*ck**4/8. - ck**5/(6.*cl) - &
            5.*ck**2*cl**2/12. + 3.*cl**4/8. - cl**5/(6.*ck))/z
       
    else if ((z > cl-ck) .and. (z <= ck-0.5*cl)) then ! f63(z)
       out = z**5*16.*(4.*ak-3.+4.*al-7.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-2.*al+2.*ak*al)/ck + z**4*8.*(1.-2.*ak+2.*ak*al)/cl + &
            z**3*10.*(1.-2.*ak-2.*al+8.*ak*al) + &
            z**2*20.*ck**2*(2.*ak-1.-8.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(2.*al-1.-8.*ak*al)/(3.*ck) + &
            z*5.*(2.*ak-1.+2.*al-20.*ak*al)*(ck**2 - ck**3/cl + cl**2 - cl**3/ck) + &
            5.*ck**3*(1.-2.*ak+2.*al+28.*ak*al)*0.5 + &
            3.*ck**4*(2.*ak-1.-32.*ak*al)*0.5/cl + &
            5.*cl**3*(1.+2.*ak-2.*al+28.*ak*al)*0.5 + 3.*cl**4*(2.*al-1.-32.*ak*al)*0.5/ck + &
            (2.*ak-1.+2.*al-68.*ak*al)*(3.*ck**4/8. - ck**5/(6.*cl) - &
            5.*ck**2*cl**2/12. + 3.*cl**4/8. - cl**5/(6.*ck))/z
       
    else if ((z > ck-0.5*cl) .and. (z <= 0.5*ck)) then ! f64(z)
       out = z**5*16.*(3.*ak-3.+4.*al-5.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + z**4*8.*(1.-2.*ak*al)/cl + &
            z**3*10.*(1.-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(8.*ak*al-1.-6.*ak)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(1.+14.*ak-2.*al-12.*ak*al)/cl + &
            z*5.*cl**2*(2.*al-1.-16.*ak*al) + z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(32.*ak*al-1.-30.*ak)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(1.+62.*ak-2.*al-60.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + &
            cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*ck) .and. (z <= 0.5*cl)) then ! f65(z)
       out = z**5*16.*(2.*al-1.-ak-ak*al)/(3.*ck*cl) + &
            z**4*8.*(1.-ak-2.*al+4.*ak*al)/ck + &
            z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + z**3*10.*(1.-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-1.+2.*al-10.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + &
            z*5.*cl**3*(1.-ak-2.*al+18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(ak-1.+2.*al-34.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(1.-ak-2.*al+66.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*cl) .and. (z <= cl-0.5*ck)) then ! f66(z)
       out = z**5*16.*(1.-3.*ak-2.*al+3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+2.*al)/ck + z**4*8.*(4.*ak-1.+2.*al-6.*ak*al)/cl + &
            z**3*10.*(1.-2.*al+4.*ak*al) + &
            z**2*20.*ck**2*(1.-10.*ak-2.*al+12.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-2.*al-6.*ak*al)/(3.*ck) + &
            z*5.*ck**2*(2.*al-1.-6.*ak-4.*ak*al) + &
            z*5.*ck**3*(18.*ak-1.-16.*ak*al)/cl + z*5.*cl**2*(2.*al-1.-16.*ak*al) + &
            z*5.*cl**3*(ak-1.+2.*al+14.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak+2.*al-4.*ak*al)*0.5 + &
            3.*ck**4*(1.-34.*ak-2.*al+36.*ak*al)*0.5/cl + &
            5.*cl**3*(1.-2.*al+32.*ak*al)*0.5 + 3.*cl**4*(1.-ak-2.*al-30.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*al-1.-30.*ak-4.*ak*al)/(8.*z) + &
            ck**5*(66.*ak-1.-64.*ak*al)/(6.*cl*z) + &
            5.*ck**2*cl**2*(1.+6.*ak-2.*al+52.*ak*al)/(12.*z) + &
            3.*cl**4*(2.*al-1.-64.*ak*al)/(8.*z) + cl**5*(ak-1.+2.*al+62.*ak*al)/(6.*ck*z)
       
    else if ((z > cl-0.5*ck) .and. (z <= ck)) then ! f67(z)
       out = z**5*16.*(1.-3.*ak-3.*al+5.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(4.*ak-1.+al-4.*ak*al)/cl + &
            z**3*10. + &
            z**2*20.*ck**2*(1.-10.*ak-al+10.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + &
            z*5.*ck**3*(18.*ak-1.+al-18.*ak*al)/cl - z*5.*ck**2*(1.+6.*ak) - &
            z*5.*cl**2*(1.+6.*al) + z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.-34.*ak-al+34.*ak*al)*0.5/cl + &
            5.*cl**3*(1+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + &
            ck**5*(66.*ak-1.+al-66.*ak*al)/(6.*cl*z) - &
            3.*ck**4*(1.+30.*ak)/(8.*z) + &
            5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - &
            3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)
       
    else if ((z > ck) .and. (z <= 0.5*(ck+cl))) then ! f68(z)
       out = z**5*16.*(1.-ak-3.*al+3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(ak-1.+4.*al-4.*ak*al)/ck + z**4*8.*(al-1.)/cl + z**3*10. + &
            z**2*20.*ck**2*(1.+6.*ak-al-6.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(1.-ak-10.*al+10.*ak*al)/(3.*ck) + &
            z*5.*ck**3*(al-1.-14.*ak+14.*ak*al)/cl - &
            z*5.*ck**2*(1.+6.*ak) - z*5.*cl**2*(1.+6.*al) + &
            z*5.*cl**3*(ak-1.+18.*al-18.*ak*al)/ck + &
            5.*ck**3*(1.+14.*ak)*0.5 + 3.*ck**4*(1.+30.*ak-al-30.*ak*al)*0.5/cl + &
            5.*cl**3*(1.+14.*al)*0.5 + 3.*cl**4*(1.-ak-34.*al+34.*ak*al)*0.5/ck + &
            ck**5*(al-1.-62.*ak+62.*ak*al)/(6.*cl*z) - 3.*ck**4*(1.+30.*ak)/(8.*z) + &
            5.*ck**2*cl**2*(1.+6.*ak+6.*al+36.*ak*al)/(12.*z) - &
            3.*cl**4*(1.+30.*al)/(8.*z) + cl**5*(ak-1.+66.*al-66.*ak*al)/(6.*ck*z)
       
    else if ((z > 0.5*ck+0.5*cl) .and. (z <= cl)) then ! f69(z) (= f49(z))
       out = z**5*16.*(ak-al-ak*al)/(3.*ck*cl) + &
            z**4*8.*(2.*al-ak)/ck + z**4*8.*(4.*ak*al-al-2.*ak)/cl + &
            z**3*20.*(ak+al-2.*ak*al) + &
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak-8.*al+6.*ak*al)/(3.*ck) + &
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + &
            z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + &
            z*5.*cl**3*(16.*al-ak-14.*ak*al)/ck + &
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + &
            5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak-32.*al+30.*ak*al)*0.5/ck + &
            3.*ck**4*(-16.*ak-al+2.*ak*al)/(4.*z) + &
            ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + &
            10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-ak-16.*al)/(4.*z) + cl**5*(64.*al-ak-62.*ak*al)/(6.*ck*z)
       
    else if ((z > cl) .and. (z <= ck+0.5*cl)) then ! f610(z) ( = f410(z))
       out = z**5*16.*(ak+al-3.*ak*al)/(3.*ck*cl) + &
            z**4*8.*(4.*ak*al-ak-2.*al)/ck + z**4*8.*(4.*ak*al-2.*ak-al)/cl + &
            z**3*20.*(ak+al-2.*ak*al) +&
            z**2*20.*ck**2*(8.*ak+al-10.*ak*al)/(3.*cl) + &
            z**2*20.*cl**2*(ak+8.*al-10.*ak*al)/(3.*ck) + &
            z*10.*ck**2*(2.*ak*al-4.*ak-al) + &
            z*5.*ck**3*(18.*ak*al-16.*ak-al)/cl + z*10.*cl**2*(2.*ak*al-ak-4.*al) + &
            z*5.*cl**3*(18.*ak*al-ak-16.*al)/ck + &
            5.*ck**3*(8.*ak+al-2.*ak*al) + 3.*ck**4*(32.*ak+al-34.*ak*al)*0.5/cl + &
            5.*cl**3*(ak+8.*al-2.*ak*al) + 3.*cl**4*(ak+32.*al-34.*ak*al)*0.5/ck + &
            3.*ck**4*(2.*ak*al-al-16.*ak)/(4.*z) + ck**5*(66.*ak*al-64.*ak-al)/(6.*cl*z) + &
            10.*ck**2*cl**2*(ak+al+4.*ak*al)/(3.*z) + &
            3.*cl**4*(2.*ak*al-16.*al-ak)/(4.*z) + cl**5*(66.*ak*al-ak-64.*al)/(6.*ck*z)
       
    else if ((z > ck+0.5*cl) .and. (z <= cl+0.5*ck)) then ! f611(z) ( = f411(z))
       out = z**5*16.*(al-ak*al)/(3.*ck*cl) + &
            z**4*16.*(ak*al-al)/ck - z**4*8.*al/cl + z**3*20.*al + &
            z**2*20.*ck**2*(al+6.*ak*al)/(3.*cl) + z**2*160.*cl**2*(al-ak*al)/(3.*ck) + &
            z*80.*cl**3*(ak*al-al)/ck - z*10.*ck**2*(al+6.*ak*al) - &
            z*5.*ck**3*(al+14.*ak*al)/cl - z*40.*cl**2*al + &
            5.*ck**3*(al+14.*ak*al) + 3.*ck**4*(al+30.*ak*al)*0.5/cl + &
            40.*cl**3*al +48.*cl**4*(al-ak*al)/ck + &
            10.*ck**2*cl**2*(al+6.*ak*al)/(3.*z) - &
            3.*ck**4*(al+30.*ak*al)/(4.*z) - ck**5*(al+62.*ak*al)/(6.*cl*z) - &
            12.*cl**4*al/z + 32.*cl**5*(ak*al-al)/(3.*ck*z)
       
    else if ((z >= cl+0.5*ck) .and. (z <= cl+ck)) then ! f612(z) ( = f412(z))
       out = ak*al*(z**5*16./(3.*ck*cl) - z**4*16./ck - z**4*16./cl + z**3*40.+ &
            z**2*160.*ck**2/(3.*cl) + z**2*160.*cl**2/(3.*ck) - &
            z*80.*(ck**2 + ck**3/cl + cl**2 + cl**3/ck) + &
            80.*ck**3 + 48.*ck**4/cl + 80.*cl**3 + 48.*cl**4/ck + &
            80.*ck**2*cl**2/(3.*z) - 24.*ck**4/z - 32.*ck**5/(3.*cl*z) - &
            24.*cl**4/z - 32.*cl**5/(3.*ck*z))
       
    else
       out = 0.
    endif
  end subroutine f6

  subroutine f7(z,ak,al,cc,out)
    implicit none
    ! For the case where ck = cl, calls Eq. (33)
    ! of Gaspari et al., (2006). See Appendix C Equations (C.1) and (C.2)
    ! for coefficients

    ! Input: 
    ! z - (scalar) norm(xk,xl) for each grid point, depends on choice of norm
    !     need z >= 0
    ! ak, al - (scalar) corresponding scalars for xk, xl
    ! cc - (scalar) cut-off length parameter, ck = cl, only one 
    !      input necessary
    real*8 :: z, ak, al, cc
    real*8, intent(out) :: out
    real*8 :: rr
    rr = z/cc

    if ((rr >= 0.) .and. (rr <= 0.5)) then ! f1(z) in (C.1)
       out = rr**5*16.*(4.*ak-3.-7.*ak*al+4.*al)/3. + & 
            rr**4*16.*(1.+2.*ak*al-ak-al) + & 
            rr**3*10.*(1.+8.*ak*al-2.*ak-2.*al) - &
            rr**2*40.*(1.+8.*ak*al-ak-al)/3. + &
            2.+44.*ak*al+3.*ak+3.*al
       
    else if ((rr > 0.5) .and. (rr <= 1.)) then ! f2(z) in (C.1)
       out = rr**5*16.*(1.+5.*ak*al-3.*ak-3.*al)/3. + & 
            rr**4*8.*(5.*ak-2.-8.*ak*al+5.*al) + rr**3*10. + &
            rr**2*20.*(2.+20.*ak*al-11.*ak-11.*al)/3. + &
            rr*5.*(13.*ak-4.-36.*ak*al+13.*al) + &
            (16.+204.*ak*al-35.*ak-35.*al)/2. + &
            (29.*ak-8.-84.*ak*al+29.*al)/(12.*rr)
       
    else if ((rr > 1.) .and. (rr <= 1.5)) then ! f3(z) in (C.1)
       out = rr**5*16.*(ak-3.*ak*al+al)/3. + &
            rr**4*8.*(8.*ak*al-3.*ak-3.*al) + &
            rr**3*20.*(ak-2.*ak*al+al) + &
            rr**2*20.*(9.*ak-20.*ak*al+9.*al)/3. + &
            rr*5.*(44.*ak*al-27.*ak-27.*al) + &
            0.5*(189.*ak-244.*ak*al+189.*al) + &
            (460.*ak*al-243.*ak-243.*al)/(12.*rr)
       
    else if ((rr > 1.5) .and. (rr <= 2.)) then ! f4(z) in (C.1)
       out =  8.*ak*al*(rr**5*2. - rr**4*12. + rr**3*15. + &
            rr**2*40. - rr*120. + 96. - 16./rr)/3.
       
    else 
       out = 0.
    endif
  end subroutine f7

end module gengc_function



