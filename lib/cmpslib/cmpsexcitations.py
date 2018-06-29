import sys
import cmpsfunctions as cmf
import warnings
import functools as fct
from scipy.sparse.linalg import LinearOperator,lgmres
import numpy as np
import os
import math
import scipy as sp
from scipy.sparse.linalg import LinearOperator
from scipy.linalg import sqrtm
import numpy as np
import lib.Lanczos.LanczosEngine as LanEn
from cmpsfunctions import mixedTransferOperator,inverseMixedTransferOperator
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))



def densityContributionSpeciesA(minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,\
                                Q1,R1,Q2,R2,ldens,rdens,k,V,W,thresh=1E-10,x0=None,tolerance=1E-12,maxiteration=4000,inner_m=30,outer_k=20):
    l=ldens
    r=rdens
    
    #there are seven terms contributing to the density-part of the Hamiltonian


    #first term, W part
    term1W=minusT11inv_R1bar_l_R1.transpose().dot(W).dot(r)
    #print 'term1 W:',np.tensordot(term1W,np.conj(W),([0,1],[0,1]))
    
    #second term, W part
    term2W=l.transpose().dot(W).dot(minusT22inv_R2_r_R2bar)
    
    #third term, W part
    temporary=np.transpose(minusT11inv_R1bar_l_R1.transpose().dot(V))+np.transpose(herm(R1).dot(minusT11inv_R1bar_l_R1.transpose()).dot(W))
    
    minusT21_plus_ip_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,sigma=-1j*k,direction=1,l=l,r=r,\
                                                                       thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    
    term3W=minusT21_plus_ip_inv_temporary.transpose().dot(R2).dot(r)
    #third term, V part
    term3V=minusT21_plus_ip_inv_temporary.transpose().dot(r)

    #fourth term
    temporary=V.dot(r)+W.dot(r).dot(herm(R2))
    minusT12_minus_ip_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q1,Ru=R1,Ql=Q2,Rl=R2,sigma=1j*k,direction=-1,l=l,r=r,\
                                                                        thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    term4V=minusT11inv_R1bar_l_R1.transpose().dot(minusT12_minus_ip_inv_temporary)
    term4W=minusT11inv_R1bar_l_R1.transpose().dot(R1).dot(minusT12_minus_ip_inv_temporary)


    #fifth term
    term5W=l.transpose().dot(R1).dot(minusT12_minus_ip_inv_temporary)
    #sixth term
    temporary=np.transpose(herm(R1).dot(l.transpose()).dot(W))
    minusT21_plus_ip_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,sigma=-1j*k,direction=1,l=l,r=r,\
                                                                       thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    term6W=minusT21_plus_ip_inv_temporary.transpose().dot(R2).dot(r)
    term6V=minusT21_plus_ip_inv_temporary.transpose().dot(r)
 
    #seventh term
    term7W=l.transpose().dot(W).dot(r)
    termW=term1W+term2W+term3W+term4W+term5W+term6W+term7W
    termV=term3V+term4V+term6V
    return termV,termW

def interactionContributionSpeciesA(minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,\
                            Q1,R1,Q2,R2,ldens,rdens,k,V,W,thresh=1E-10,x0=None,tolerance=1E-12,maxiteration=4000,inner_m=30,outer_k=20):
    l=ldens
    r=rdens
    
    #first term, W part
    term1W=minusT11inv_R1barR1bar_l_R1R1.transpose().dot(W).dot(r)
    #second term, W part
    term2W=l.transpose().dot(W).dot(minusT22inv_R2R2_r_R2barR2bar)
    #third term, W part
    temporary=np.transpose(minusT11inv_R1barR1bar_l_R1R1.transpose().dot(V))+np.transpose(herm(R1).dot(minusT11inv_R1barR1bar_l_R1R1.transpose()).dot(W))
    minusT21_plus_ip_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,sigma=-1j*k,direction=1,l=l,r=r,\
                                                                       thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    
    term3W=minusT21_plus_ip_inv_temporary.transpose().dot(R2).dot(r)
    #third term, V part
    term3V=minusT21_plus_ip_inv_temporary.transpose().dot(r)
    

    #fourth term
    temporary=V.dot(r)+W.dot(r).dot(herm(R2))
    minusT12_minus_ip_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q1,Ru=R1,Ql=Q2,Rl=R2,sigma=1j*k,direction=-1,l=l,r=r,\
                                                                        thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    #print 'overlap 4',np.tensordot(l, minusT12_minus_ip_inv_temporary,([0,1],[0,1]))

    #fourth term , V part
    term4V=minusT11inv_R1barR1bar_l_R1R1.transpose().dot(minusT12_minus_ip_inv_temporary)
    #fourth term , W part
    term4W=minusT11inv_R1barR1bar_l_R1R1.transpose().dot(R1).dot(minusT12_minus_ip_inv_temporary)

    #fifth term
    term5W=herm(R1).dot(l.transpose()).dot(R1).dot(R1).dot(minusT12_minus_ip_inv_temporary)+\
        l.transpose().dot(R1).dot(R1).dot(minusT12_minus_ip_inv_temporary).dot(herm(R2))
    
    #sixth term
    temporary=np.transpose(herm(R1.dot(R1)).dot(l.transpose()).dot(R1.dot(W)+W.dot(R2)))
    minusT21_plus_ip_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,sigma=-1j*k,direction=1,l=l,r=r,\
                                                                       thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    term6W=minusT21_plus_ip_inv_temporary.transpose().dot(R2).dot(r)
    term6V=minusT21_plus_ip_inv_temporary.transpose().dot(r)

    #seventh term
    term7W=herm(R1).dot(l.transpose()).dot(R1.dot(W)+W.dot(R2)).dot(r)+\
        l.transpose().dot(R1.dot(W)+W.dot(R2)).dot(r).dot(herm(R2))
    
    termW=term1W+term2W+term3W+term4W+term5W+term6W+term7W
    termV=term3V+term4V+term6V

    return termV,termW


def kineticContributionSpeciesA(minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,\
                            Q1,R1,Q2,R2,ldens,rdens,k,V,W,thresh=1E-10,x0=None,tolerance=1E-12,maxiteration=4000,inner_m=30,outer_k=20):
    l=ldens
    r=rdens
    
  
    #first term, W part
    term1W=minusT11inv_commQ1barR1bar_l_commQ1R1.transpose().dot(W).dot(r)
    
    #second term, W part
    term2W=l.transpose().dot(W).dot(minusT22inv_commQ2R2_r_commQ2barR2bar)
    
    #third term, W part
    temporary=np.transpose(minusT11inv_commQ1barR1bar_l_commQ1R1.transpose().dot(V))+np.transpose(herm(R1).dot(minusT11inv_commQ1barR1bar_l_commQ1R1.transpose()).dot(W))
    minusT21_plus_ip_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,sigma=-1j*k,direction=1,l=l,r=r,\
                                                                       thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    
    term3W=minusT21_plus_ip_inv_temporary.transpose().dot(R2).dot(r)
    #third term, V part
    term3V=minusT21_plus_ip_inv_temporary.transpose().dot(r)

    #fourth term
    temporary=V.dot(r)+W.dot(r).dot(herm(R2))
    minusT12_minus_ip_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q1,Ru=R1,Ql=Q2,Rl=R2,sigma=1j*k,direction=-1,l=l,r=r,\
                                                                        thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)

    #fourth term , V part
    term4V=minusT11inv_commQ1barR1bar_l_commQ1R1.transpose().dot(minusT12_minus_ip_inv_temporary)
    #fourth term , W part
    term4W=minusT11inv_commQ1barR1bar_l_commQ1R1.transpose().dot(R1).dot(minusT12_minus_ip_inv_temporary)

    #fifth term
    temporary=np.transpose(herm(comm(Q1,R1)).dot(l.transpose()).dot(Q1.dot(W)-W.dot(Q2)+V.dot(R2)-R1.dot(V)+1j*k*W))
    minusT21_plus_ip_inv_temporary=(-1.0)*inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,sigma=-1j*k,direction=1,l=l,r=r,\
                                                                       thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    term5W=minusT21_plus_ip_inv_temporary.transpose().dot(R2).dot(r)
    term5V=minusT21_plus_ip_inv_temporary.transpose().dot(r)

    #sixth term
    term6W=herm(Q1).dot(l.transpose()).dot(comm(Q1,R1)).dot(minusT12_minus_ip_inv_temporary)\
        -l.transpose().dot(comm(Q1,R1)).dot(minusT12_minus_ip_inv_temporary).dot(herm(Q2))\
        -1j*k*l.transpose().dot(comm(Q1,R1)).dot(minusT12_minus_ip_inv_temporary)
        
    term6V=l.transpose().dot(comm(Q1,R1)).dot(minusT12_minus_ip_inv_temporary).dot(herm(R2))\
       -herm(R1).dot(l.transpose()).dot(comm(Q1,R1)).dot(minusT12_minus_ip_inv_temporary)

    #seventh term
    upper=Q1.dot(W)-W.dot(Q2)+V.dot(R2)-R1.dot(V)+1j*k*W
    term7W=herm(Q1).dot(l.transpose()).dot(upper).dot(r)\
        -l.transpose().dot(upper).dot(r).dot(herm(Q2))\
        -1j*k*l.transpose().dot(upper).dot(r)
    term7V=l.transpose().dot(upper).dot(r).dot(herm(R2))\
        -herm(R1).dot(l.transpose()).dot(upper).dot(r)

    termW=term1W+term2W+term3W+term4W+term5W+term6W+term7W
    termV=term3V+term4V+term5V+term6V+term7V
    return termV,termW

def psidag_psidagContributionSpeciesA(minusT11inv_R1barR1bar_l,minusT22inv_r_R2barR2bar,\
                            Q1,R1,Q2,R2,ldens,rdens,k,V,W,thresh=1E-10,x0=None,tolerance=1E-12,maxiteration=4000,inner_m=30,outer_k=20):
    l=ldens
    r=rdens
    
    #there are seven terms contributing to the density-part of the Hamiltonian


    #first term, W part
    term1W=minusT11inv_R1barR1bar_l.transpose().dot(W).dot(r)
    
    #second term, W part
    term2W=l.transpose().dot(W).dot(minusT22inv_r_R2barR2bar)
    
    #third term, W part
    temporary=np.transpose(minusT11inv_R1barR1bar_l.transpose().dot(V))+\
        np.transpose(herm(R1).dot(minusT11inv_R1barR1bar_l.transpose()).dot(W))
    minusT21_plus_ip_inv_temporary=(-1.0)*\
        inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,\
                                     sigma=-1j*k,direction=1,l=l,r=r,\
                                     thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    term3W=minusT21_plus_ip_inv_temporary.transpose().dot(R2).dot(r)
    #third term, V part
    term3V=minusT21_plus_ip_inv_temporary.transpose().dot(r)

    #fourth term
    temporary=V.dot(r)+W.dot(r).dot(herm(R2))
    minusT12_minus_ip_inv_temporary=(-1.0)*\
        inverseMixedTransferOperator(ih=temporary,Qu=Q1,Ru=R1,Ql=Q2,Rl=R2,\
                                     sigma=1j*k,direction=-1,l=l,r=r,\
                                     thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    term4V=minusT11inv_R1barR1bar_l.transpose().dot(minusT12_minus_ip_inv_temporary)
    term4W=minusT11inv_R1barR1bar_l.transpose().dot(R1).dot(minusT12_minus_ip_inv_temporary)


    #fifth term
    term5W=herm(R1).dot(l.transpose()).dot(minusT12_minus_ip_inv_temporary)+\
        l.transpose().dot(minusT12_minus_ip_inv_temporary).dot(herm(R2))

    termW=term1W+term2W+term3W+term4W+term5W
    termV=term3V+term4V
    return termV,termW

def psi_psiContributionSpeciesA(minusT11inv_l_R1R1,minusT22inv_R2R2_r,\
                                Q1,R1,Q2,R2,ldens,rdens,k,V,W,thresh=1E-10,x0=None,tolerance=1E-12,maxiteration=4000,inner_m=30,outer_k=20):
    l=ldens
    r=rdens
    
    #there are seven terms contributing to the density-part of the Hamiltonian


    #first term, W part
    term1W=minusT11inv_l_R1R1.transpose().dot(W).dot(r)
    
    #second term, W part
    term2W=l.transpose().dot(W).dot(minusT22inv_R2R2_r)
    
    #third term, W part
    temporary=np.transpose(minusT11inv_l_R1R1.transpose().dot(V))+\
        np.transpose(herm(R1).dot(minusT11inv_l_R1R1.transpose()).dot(W))
    minusT21_plus_ip_inv_temporary=(-1.0)*\
        inverseMixedTransferOperator(ih=temporary,Qu=Q2,Ru=R2,Ql=Q1,Rl=R1,\
                                     sigma=-1j*k,direction=1,l=l,r=r,\
                                     thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    #print 'overlap 3',np.tensordot(minusT21_plus_ip_inv_temporary,r,([0,1],[0,1]))
    
    term3W=minusT21_plus_ip_inv_temporary.transpose().dot(R2).dot(r)
    #third term, V part
    term3V=minusT21_plus_ip_inv_temporary.transpose().dot(r)

    #fourth term
    temporary=V.dot(r)+W.dot(r).dot(herm(R2))
    minusT12_minus_ip_inv_temporary=(-1.0)*\
        inverseMixedTransferOperator(ih=temporary,Qu=Q1,Ru=R1,Ql=Q2,Rl=R2,\
                                     sigma=1j*k,direction=-1,l=l,r=r,\
                                     thresh=thresh,x0=x0,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    term4V=minusT11inv_l_R1R1.transpose().dot(minusT12_minus_ip_inv_temporary)
    term4W=minusT11inv_l_R1R1.transpose().dot(R1).dot(minusT12_minus_ip_inv_temporary)


    #fifth term
    term5W=herm(R1).dot(l.transpose()).dot(R1).dot(R1).dot(minusT12_minus_ip_inv_temporary)+\
        l.transpose().dot(R1).dot(R1).dot(minusT12_minus_ip_inv_temporary).dot(herm(R2))

    termW=term1W+term2W+term3W+term4W+term5W
    termV=term3V+term4V
    return termV,termW   


def HAproduct(n,Q1,R1,Q2,R2,ldens,invl,sqrtl,invsqrtl,rdens,invsqrtr,k,\
              mu,mass,g,\
              minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,\
              minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,\
              minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,\
              thresh,tolerance,maxiteration,inner_m,outer_k,verbosity,\
              vector):
    
    n[0]+=1
    if verbosity>=1:
        sys.stdout.write("\rCalling HA product: %i" % n[0])
        sys.stdout.flush()

    D=Q1.shape[0]

    Y=np.reshape(vector,(D,D))
    W=invsqrtl.dot(Y).dot(invsqrtr)
    V=-invl.dot(herm(R1)).dot(sqrtl).dot(Y).dot(invsqrtr)

    Vd,Wd=densityContributionSpeciesA(minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,Q1=Q1,R1=R1,Q2=Q2,R2=R2,ldens=ldens,rdens=rdens,k=k,V=V,W=W,\
                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    Vi,Wi=interactionContributionSpeciesA(minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,Q1=Q1,R1=R1,Q2=Q2,R2=R2,ldens=ldens,rdens=rdens,k=k,V=V,W=W,\
                                          thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    Vk,Wk=kineticContributionSpeciesA(minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,Q1=Q1,R1=R1,Q2=Q2,R2=R2,ldens=ldens,rdens=rdens,k=k,V=V,W=W,\
                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    #V part of Y-opt:
    
    Wopt=1.0/(2*mass)*Wk+g*Wi+mu*Wd
    Vopt=1.0/(2*mass)*Vk+g*Vi+mu*Vd
    
    Yopt=-herm(sqrtl).dot(R1).dot(herm(invl)).dot(Vopt).dot(herm(invsqrtr))+\
        herm(invsqrtl).dot(Wopt).dot(herm(invsqrtr))
    return np.reshape(Yopt,D*D)



def HAproductDelta(n,Q1,R1,Q2,R2,ldens,invl,sqrtl,invsqrtl,rdens,invsqrtr,k,\
                   mu,mass,g,Delta,\
                   minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,\
                   minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,\
                   minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,\
                   minusT11inv_l_R1R1,minusT22inv_R2R2_r,\
                   minusT11inv_R1barR1bar_l,minusT22inv_r_R2barR2bar,\
                   thresh,tolerance,maxiteration,inner_m,outer_k,verbosity,\
                   vector):
    n[0]+=1
    if verbosity>=1:
        sys.stdout.write("\rCalling HA product: %i" % n[0])
        sys.stdout.flush()

    D=Q1.shape[0]
    Y=np.reshape(vector,(D,D))
    W=invsqrtl.dot(Y).dot(invsqrtr)
    V=-invl.dot(herm(R1)).dot(sqrtl).dot(Y).dot(invsqrtr)

    Vd,Wd=densityContributionSpeciesA(minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,Q1=Q1,R1=R1,Q2=Q2,R2=R2,ldens=ldens,rdens=rdens,k=k,V=V,W=W,\
                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    Vi,Wi=interactionContributionSpeciesA(minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,Q1=Q1,R1=R1,Q2=Q2,R2=R2,ldens=ldens,rdens=rdens,k=k,V=V,W=W,\
                                          thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    Vk,Wk=kineticContributionSpeciesA(minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,Q1=Q1,R1=R1,Q2=Q2,R2=R2,ldens=ldens,rdens=rdens,k=k,V=V,W=W,\
                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    Vpp,Wpp=psi_psiContributionSpeciesA(minusT11inv_l_R1R1,\
                                        minusT22inv_R2R2_r,Q1,R1,Q2,R2,\
                                        ldens,rdens,k,V,W,\
                                        thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    Vpdpd,Wpdpd=psidag_psidagContributionSpeciesA(minusT11inv_R1barR1bar_l,\
                                                  minusT22inv_r_R2barR2bar,\
                                                  Q1,R1,Q2,R2,ldens,rdens,k,\
                                                  V,W,\
                                                  thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    #V part of Y-opt:
    
    Wopt=1.0/(2*mass)*Wk+g*Wi+mu*Wd+Delta*(Wpp+Wpdpd)
    Vopt=1.0/(2*mass)*Vk+g*Vi+mu*Vd+Delta*(Vpp+Vpdpd)
    
    Yopt=-herm(sqrtl).dot(R1).dot(herm(invl)).dot(Vopt).dot(herm(invsqrtr))+\
        herm(invsqrtl).dot(Wopt).dot(herm(invsqrtr))
    return np.reshape(Yopt,D*D)

#init has to be a vector
def eigsh(Q1,R1,Q2,R2,ldens,rdens,k,mu,mass,g,Delta,init,accuracy=1e-8,numvecs=1,numcv=50,\
          thresh=1E-10,tolerance=1E-12,maxiteration=4000,inner_m=30,outer_k=20,verbosity=0):
    D=Q1.shape[0]
    n=[0]
    l=ldens
    r=rdens
    
    R1bar_l_R1=np.transpose(herm(R1).dot(l.transpose()).dot(R1))
    R2_r_R2bar=R2.dot(r).dot(herm(R2))
    
    minusT11inv_R1bar_l_R1=(-1.0)*inverseMixedTransferOperator(ih=R1bar_l_R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    minusT22inv_R2_r_R2bar=(-1.0)*inverseMixedTransferOperator(ih=R2_r_R2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    
    R1barR1bar_l_R1R1=np.transpose(herm(R1.dot(R1)).dot(l.transpose()).dot(R1).dot(R1))
    R2R2_r_R2barR2bar=R2.dot(R2).dot(r).dot(herm(R2.dot(R2)))
    
    minusT11inv_R1barR1bar_l_R1R1=(-1.0)*inverseMixedTransferOperator(ih=R1barR1bar_l_R1R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    minusT22inv_R2R2_r_R2barR2bar=(-1.0)*inverseMixedTransferOperator(ih=R2R2_r_R2barR2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
  
    commQ1barR1bar_l_commQ1R1=np.transpose(herm(comm(Q1,R1)).dot(l.transpose()).dot(comm(Q1,R1)))
    commQ2R2_r_commQ2barR2bar=comm(Q2,R2).dot(r).dot(herm(comm(Q2,R2)))
    
    minusT11inv_commQ1barR1bar_l_commQ1R1=(-1.0)*inverseMixedTransferOperator(ih=commQ1barR1bar_l_commQ1R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                                              thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    minusT22inv_commQ2R2_r_commQ2barR2bar=(-1.0)*inverseMixedTransferOperator(ih=commQ2R2_r_commQ2barR2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                                              thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)

    
    sqrtl=sqrtm(ldens)
    sqrtr=sqrtm(rdens)
    invl=np.linalg.pinv(ldens)
    invr=np.linalg.pinv(rdens)
    invsqrtl=np.linalg.pinv(sqrtl)
    invsqrtr=np.linalg.pinv(sqrtr)
    
    if abs(Delta)>1E-10:
        #pairing term
        l_R1R1=np.transpose(l.transpose().dot(R1.dot(R1)))
        R2R2_r=R2.dot(R2).dot(r)
        
        minusT11inv_l_R1R1=(-1.0)*inverseMixedTransferOperator(ih=l_R1R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        minusT22inv_R2R2_r=(-1.0)*inverseMixedTransferOperator(ih=R2R2_r,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        
        R1barR1bar_l=np.transpose(herm(R1).dot(herm(R1)).dot(l.transpose()))
        r_R2barR2bar=r.dot(herm(R2).dot(herm(R2)))
        
        minusT11inv_R1barR1bar_l=(-1.0)*inverseMixedTransferOperator(ih=R1barR1bar_l,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                                     thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        minusT22inv_r_R2barR2bar=(-1.0)*inverseMixedTransferOperator(ih=r_R2barR2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                                     thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        
        mv=fct.partial(HAproductDelta,*[n,Q1,R1,Q2,R2,ldens,invl,sqrtl,invsqrtl,\
                                        rdens,invsqrtr,\
                                        k,mu,mass,g,Delta,\
                                        minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,\
                                        minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,\
                                        minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,\
                                        minusT11inv_l_R1R1,minusT22inv_R2R2_r,\
                                        minusT11inv_R1barR1bar_l,minusT22inv_r_R2barR2bar,\
                                        thresh,tolerance,maxiteration,inner_m,outer_k,verbosity])
    else:
        mv=fct.partial(HAproduct,*[n,Q1,R1,Q2,R2,ldens,invl,sqrtl,invsqrtl,\
                                   rdens,invsqrtr,\
                                   k,mu,mass,g,\
                                   minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,\
                                   minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,\
                                   minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,\
                                   thresh,tolerance,maxiteration,inner_m,outer_k,verbosity])
        
        
    LOP=LinearOperator((D**2,D**2),matvec=mv,rmatvec=None,matmat=None,dtype=Q1.dtype)
    e,v=sp.sparse.linalg.eigsh(LOP,k=numvecs,which='SA',maxiter=1000000,tol=accuracy,v0=init,ncv=numcv)
    
    return [e,v]


def lan(Q1,R1,Q2,R2,ldens,rdens,k,mu,mass,g,Delta,init,accuracy=1E-5,Ndiag=10,nmax=100,numeig=1,delta=1E-9,\
        thresh=1E-10,tolerance=1E-12,maxiteration=4000,inner_m=30,outer_k=20,verbosity=0):
    D=Q1.shape[0]
    n=[0]
    l=ldens
    r=rdens
    R1bar_l_R1=np.transpose(herm(R1).dot(l.transpose()).dot(R1))
    R2_r_R2bar=R2.dot(r).dot(herm(R2))
    minusT11inv_R1bar_l_R1=(-1.0)*inverseMixedTransferOperator(ih=R1bar_l_R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    minusT22inv_R2_r_R2bar=(-1.0)*inverseMixedTransferOperator(ih=R2_r_R2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    
    R1barR1bar_l_R1R1=np.transpose(herm(R1.dot(R1)).dot(l.transpose()).dot(R1).dot(R1))
    R2R2_r_R2barR2bar=R2.dot(R2).dot(r).dot(herm(R2.dot(R2)))
    
    minusT11inv_R1barR1bar_l_R1R1=(-1.0)*inverseMixedTransferOperator(ih=R1barR1bar_l_R1R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    minusT22inv_R2R2_r_R2barR2bar=(-1.0)*inverseMixedTransferOperator(ih=R2R2_r_R2barR2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
  
    commQ1barR1bar_l_commQ1R1=np.transpose(herm(comm(Q1,R1)).dot(l.transpose()).dot(comm(Q1,R1)))
    commQ2R2_r_commQ2barR2bar=comm(Q2,R2).dot(r).dot(herm(comm(Q2,R2)))
    
    minusT11inv_commQ1barR1bar_l_commQ1R1=(-1.0)*inverseMixedTransferOperator(ih=commQ1barR1bar_l_commQ1R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                                              thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    minusT22inv_commQ2R2_r_commQ2barR2bar=(-1.0)*inverseMixedTransferOperator(ih=commQ2R2_r_commQ2barR2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                                              thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    
        
    sqrtl=sqrtm(ldens)
    sqrtr=sqrtm(rdens)
    invl=np.linalg.pinv(ldens)
    invr=np.linalg.pinv(rdens)
    invsqrtl=np.linalg.pinv(sqrtl)
    invsqrtr=np.linalg.pinv(sqrtr)


    
    if abs(Delta)>1E-10:
        #pairing term
        l_R1R1=np.transpose(l.transpose().dot(R1.dot(R1)))
        R2R2_r=R2.dot(R2).dot(r)
        
        minusT11inv_l_R1R1=(-1.0)*inverseMixedTransferOperator(ih=l_R1R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        minusT22inv_R2R2_r=(-1.0)*inverseMixedTransferOperator(ih=R2R2_r,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        
        R1barR1bar_l=np.transpose(herm(R1).dot(herm(R1)).dot(l.transpose()))
        r_R2barR2bar=r.dot(herm(R2).dot(herm(R2)))
        
        minusT11inv_R1barR1bar_l=(-1.0)*inverseMixedTransferOperator(ih=R1barR1bar_l,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                                     thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        minusT22inv_r_R2barR2bar=(-1.0)*inverseMixedTransferOperator(ih=r_R2barR2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                                     thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
        
        mv=fct.partial(HAproductDelta,*[n,Q1,R1,Q2,R2,ldens,invl,sqrtl,invsqrtl,\
                                        rdens,invsqrtr,\
                                        k,mu,mass,g,Delta,\
                                        minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,\
                                        minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,\
                                        minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,\
                                        minusT11inv_l_R1R1,minusT22inv_R2R2_r,\
                                        minusT11inv_R1barR1bar_l,minusT22inv_r_R2barR2bar,\
                                        thresh,tolerance,maxiteration,inner_m,outer_k,verbosity])
    else:
        mv=fct.partial(HAproduct,*[n,Q1,R1,Q2,R2,ldens,invl,sqrtl,invsqrtl,\
                                   rdens,invsqrtr,\
                                   k,mu,mass,g,\
                                   minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,\
                                   minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,\
                                   minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,\
                                   thresh,tolerance,maxiteration,inner_m,outer_k,verbosity])
                                   
    lanczos=LanEn.LanczosEngine(mv,Ndiag=Ndiag,nmax=nmax,numeig=numeig,delta=delta,deltaEta=accuracy)
    if init==None:
        init=np.random.rand(D**2).astype(Q1.dtype)
    e,v=lanczos.__simulate__(init)
    vecs=np.zeros((D*2,numeig)).astype(Q1.dtype)
    for n in range(len(numeig)):
        vecs[:,n]=np.reshape(v[n],D*D)
    return [e,vecs]




def buildHamiltonian(Q1,R1,Q2,R2,ldens,rdens,k,mu,mass,g,Delta,\
                     thresh=1E-10,tolerance=1E-12,maxiteration=4000,inner_m=30,outer_k=20,verbosity=0):
    D=Q1.shape[0]
    n=[0]
    l=ldens
    r=rdens
    R1bar_l_R1=np.transpose(herm(R1).dot(l.transpose()).dot(R1))
    R2_r_R2bar=R2.dot(r).dot(herm(R2))
    minusT11inv_R1bar_l_R1=(-1.0)*inverseMixedTransferOperator(ih=R1bar_l_R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    minusT22inv_R2_r_R2bar=(-1.0)*inverseMixedTransferOperator(ih=R2_r_R2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                               thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    
    R1barR1bar_l_R1R1=np.transpose(herm(R1.dot(R1)).dot(l.transpose()).dot(R1).dot(R1))
    R2R2_r_R2barR2bar=R2.dot(R2).dot(r).dot(herm(R2.dot(R2)))
    
    minusT11inv_R1barR1bar_l_R1R1=(-1.0)*inverseMixedTransferOperator(ih=R1barR1bar_l_R1R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    minusT22inv_R2R2_r_R2barR2bar=(-1.0)*inverseMixedTransferOperator(ih=R2R2_r_R2barR2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                                      thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
  
    commQ1barR1bar_l_commQ1R1=np.transpose(herm(comm(Q1,R1)).dot(l.transpose()).dot(comm(Q1,R1)))
    commQ2R2_r_commQ2barR2bar=comm(Q2,R2).dot(r).dot(herm(comm(Q2,R2)))
    
    minusT11inv_commQ1barR1bar_l_commQ1R1=(-1.0)*inverseMixedTransferOperator(ih=commQ1barR1bar_l_commQ1R1,Qu=Q1,Ru=R1,Ql=Q1,Rl=R1,sigma=0.0,direction=1,l=l,r=r,\
                                                                              thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    minusT22inv_commQ2R2_r_commQ2barR2bar=(-1.0)*inverseMixedTransferOperator(ih=commQ2R2_r_commQ2barR2bar,Qu=Q2,Ru=R2,Ql=Q2,Rl=R2,sigma=0.0,direction=-1,l=l,r=r,\
                                                                              thresh=thresh,x0=None,tolerance=tolerance,maxiteration=maxiteration,inner_m=inner_m,outer_k=outer_k)
    
        
    sqrtl=sqrtm(ldens)
    sqrtr=sqrtm(rdens)
    invl=np.linalg.pinv(ldens)
    invr=np.linalg.pinv(rdens)
    invsqrtl=np.linalg.pinv(sqrtl)
    invsqrtr=np.linalg.pinv(sqrtr)
    mv=fct.partial(HAproduct,*[n,Q1,R1,Q2,R2,ldens,invl,sqrtl,invsqrtl,\
                               rdens,invsqrtr,\
                               k,mu,mass,g,\
                               minusT11inv_R1bar_l_R1,minusT22inv_R2_r_R2bar,\
                               minusT11inv_R1barR1bar_l_R1R1,minusT22inv_R2R2_r_R2barR2bar,\
                               minusT11inv_commQ1barR1bar_l_commQ1R1,minusT22inv_commQ2R2_r_commQ2barR2bar,\
                               thresh,tolerance,maxiteration,inner_m,outer_k,verbosity])
    
    H=np.zeros((D**2,D**2)).astype(Q1.dtype)
    for n2 in range(D**2):
        sys.stdout.write("\r calculating column %i " % n2)
        sys.stdout.flush()
        x=np.zeros(D**2).astype(Q1.dtype)
        x[n2]=1.0
        H[:,n2]=mv(x) 
    return H

