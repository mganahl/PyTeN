#!/usr/bin/env python
import sys
from sys import stdout
import numpy as np
import time
import functools as fct
import matplotlib.pyplot as plt
from scipy.interpolate import splrep
from scipy.interpolate import splev
from scipy.signal import savgol_filter
from scipy.optimize import minimize_scalar
import functools as fct
import random
import math
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))



def SBwick(Mij,cdag,c):
    if len(cdag)==1:
        assert(len(c)==1)
        return Mij[cdag[0],c[0]]
    else:
        if isinstance(cdag,list):
            cdag=np.array(cdag).astype(int)
        if isinstance(c,list):
            c=np.array(c).astype(int)
            
        corr=0.0
        for n1 in range(len(cdag)):
            for n2 in range(len(c)):
                cdag_p=cdag[cdag!=cdag[n1]]
                c_p=c[c!=c[n2]]
                corr=corr+Mij[n1,n2]*SFwick(Mij,cdag_p,c_p)
        return corr


def SFwick(Mij,cdag,c):
    if len(cdag)==1:
        assert(len(c)==1)
        return Mij[cdag[0],c[0]]
    else:
        if isinstance(cdag,list):
            cdag=np.array(cdag).astype(int)
        if isinstance(c,list):
            c=np.array(c).astype(int)
            
        corr=0.0
        for n1 in range(len(cdag)):
            for n2 in range(len(c)):
                cdag_p=cdag[cdag!=cdag[n1]]
                c_p=c[c!=c[n2]]
                par=(-1)**(len(cdag)-n1+n2)
                corr=corr+par*Mij[n1,n2]*SFwick(Mij,cdag_p,c_p)
        return corr


#x0 and xN are the left and right boundaries
def computedx(x0,x,xN):
    assert(x[0]>=x0)
    assert(x[-1]<=xN)
    dxs=np.zeros(len(x))
    for n in range(len(x)):
        #mat=np.zeros((self.D,self.D,2)).astype(self.type)
        if n==0:
            dx=(x[n+1]+x[n])/2.0
        if n>0 and n<(len(x)-1):
            dx=(x[n+1]-x[n-1])/2.0
        if n==len(x)-1:
            dx=xN-x0-(x[n]+x[n-1])/2.0
        dxs[n]=dx
    assert(np.abs(np.sum(dxs)-(xN-x0))<1E-12)
    return dxs

def funK(x,period,dtype=float,const=False):
    #print ('in funk: ',seed)
    #if seed!=None:
    #    np.random.seed(seed)
    #    random.seed(seed)
    N=2
    if dtype==float:
        if not const:
            f=np.zeros(len(x))
            a=np.random.rand(N)-0.5
            b=np.random.rand(N)-0.5
            c=np.random.rand(N)-0.5
            d=np.random.rand(N)-0.5
            k1=random.sample(range(0,N,1),N)
            k2=random.sample(range(0,N,1),N)
            phi=np.random.rand(N)
            for n in range(len(a)):
                f+=a[n]*np.cos(2*k1[n]*math.pi/period*x+phi[n])+b[n]*np.sin(2*k2[n]*math.pi/period*x+phi[n])
        elif const:
            f=(np.random.rand(1)[0]-0.5)*np.ones(len(x))
        return f

    if dtype==complex:
        if not const:
            f=np.zeros(len(x)).astype(complex)
            a=np.random.rand(N)-0.5
            b=np.random.rand(N)-0.5
            c=np.random.rand(N)-0.5
            d=np.random.rand(N)-0.5
            k1=random.sample(range(0,N,1),N)
            k2=random.sample(range(0,N,1),N)
            phi1=np.random.rand(N)
            phi2=np.random.rand(N)

            #k1=np.random.rand(4)*5
            #k2=np.random.rand(4)*5
            for n in range(len(a)):
                f+=a[n]*np.cos(2*k1[n]*math.pi/period*x+phi1[n])+b[n]*np.sin(2*k2[n]*math.pi/period*x+phi1[n])+1j*(c[n]*np.cos(2*k1[n]*math.pi/period*x+phi2[n])+d[n]*np.sin(2*k2[n]*math.pi/period*x+phi2[n]))

        elif const:
            f=(np.random.rand(1)[0]-0.5+1j*(np.random.rand(1)[0]-0.5))*np.ones(len(x))

        return f

def funR(x,period,dtype=float,const=False):
    #print ('in funR: ',seed)
    #if seed!=None:
    #    np.random.seed(seed)
    #    random.seed(seed)
    N=2
    if dtype==float:
        if not const:
            f=np.zeros(len(x))
            a=np.random.rand(N)-0.5
            b=np.random.rand(N)-0.5
            k1=random.sample(range(0,N,1),N)
            k2=random.sample(range(0,N,1),N)
            phi=np.random.rand(4)
            for n in range(len(a)):
                f+=a[n]*np.sin(2*k1[n]*math.pi/period*x+phi[n])
        elif const:
            f=(np.random.rand(1)[0]-0.5)*np.ones(len(x))
        
        return f
    if dtype==complex:
        if not const:
            f=np.zeros(len(x)).astype(complex)
            a=np.random.rand(N)-0.5
            b=np.random.rand(N)-0.5
            k1=random.sample(range(0,N,1),N)
            k2=random.sample(range(0,N,1),N)
            phi1=np.random.rand(N)
            phi2=np.random.rand(N)
            for n in range(len(a)):
                f+=a[n]*np.sin(2*k1[n]*math.pi/period*x+phi1[n])+1j*b[n]*np.sin(2*k2[n]*math.pi/period*x+phi2[n])
        elif const:
            f=(np.random.rand(1)[0]-0.5+1j*(np.random.rand(1)[0]-0.5))*np.ones(len(x))
        
        return f
    
def bisection(fun,a,b,tol=1E-10,maxit=1000):
    if fun(a)==0.0:
        return a
    if fun(b)==0.0:
        return b
    assert(a<b)
    assert((fun(a)<0.0 and fun(b)>0.0) or (fun(a)>0 and fun(b)<0.0))
    it=0
    while it <maxit:
        c=(a+b)/2.0
        if fun(c)==0 or ((b-a)/2.0<tol):
            return c
        it+=1
        if np.sign(fun(c))==np.sign(fun(a)):
            a=c
        else:
            b=c

            
def placeBraket(f,a,b,Delta,thresh):
    #converged=False
    a0=a
    b0=b
    while True:
        if (np.sign(f(a))!=np.sign(f(b))):
            return a,b,True
        elif (np.sign(f(a))==np.sign(f(b))):
            a-=Delta
            b+=Delta
            #if a and b have moved too far apart, its likely that there is no root; you have to change funr 
            #in this case
        if np.abs(a-a0)>thresh or np.abs(b-b0)>thresh:
            return a,b,False

#tries to return a periodic, smooth version of ar, where ar is a D x D x len(xin) array.
#windowlength and polyorder are parameters for the savgol smoothing function of scipy. 
#Delta determines an interval I=[xin[-1]-Delta,xin[-1]+Delta] in which the function looks for a value arr[d1,d2,n], with xin[n] in I such that
#abs(arr[d1,d2,n]-arr[d1,d2,0])<tol. It then patches these pieces together to produce a smooth and periodic function for all d1,d2.
def periodicSmooth(xin,ar,order,window_length, polyorder,Delta=0.01,tol=1E-10,imax=1000,thresh=0.2,method='bounded',plot=False):
    D=ar.shape[0]
    x=np.concatenate((xin[0:-1]-xin[-1],xin,xin[1::]+xin[-1]),axis=0)
    concar=np.concatenate((ar[:,:,0:-1],ar,ar[:,:,1::]),axis=2)
    out=np.zeros(ar.shape).astype(ar.dtype)
    #sm_ar=savgol_filter(np.real(concar[d1,d2,:]),window_length, polyorder)
    if ar.dtype==np.complex128:
        for d1 in range(D):
            for d2 in range(D):
                #sm_ar=savgol_filter(np.real(concar[d1,d2,:]),window_length, polyorder)+1j*savgol_filter(np.imag(concar[d1,d2,:]),window_length, polyorder)
                sm_arr=savgol_filter(np.real(concar[d1,d2,:]),window_length, polyorder)
                sm_ari=savgol_filter(np.imag(concar[d1,d2,:]),window_length, polyorder)
                if plot==True:
                    plt.figure(100)
                    plt.plot(x,sm_arr)
                    plt.ylim([-10,10])
                    plt.figure(101)
                    plt.plot(x,sm_ari)

                splr=splrep(x,sm_arr,k=order,per=0)
                spli=splrep(x,sm_ari,k=order,per=0)

                #place the braket at the right boundary of the unit-cell
                a0=xin[-1]-Delta
                b0=xin[-1]+Delta
                
                xL=xin[0]
                funr=lambda z:splev(z,splr)-splev(xL,splr)
                minusfunr=lambda z:-splev(z,splr)+splev(xL,splr)
                d_dx_funr=lambda z:splev(z,splr,der=1)
                if np.abs(funr(a0))<tol or np.abs(funr(b0))<tol:
                    if np.abs(funr(a0))<tol:
                        xR=a0
                    elif np.abs(funr(b0))<tol:
                        xR=b0
                else:
                    if np.sign(d_dx_funr(a0))==np.sign(d_dx_funr(b0)):
                        conv=False
                        it=0
                        #if the sign of the derivative is the same on either side, 
                        #make sure that there is a root of funr between a and b
                        while conv==False:
                            a,b,conv=placeBraket(funr,a0,b0,Delta,thresh=thresh)
                            if conv==True:
                                #there is a root
                                xR=bisection(funr,a,b,tol=tol,maxit=imax)
                                break
                            elif conv==False:
                                #there is no root between a0,b0; shift the points a bit and redefine funr
                                xL-=Delta
                                funr=lambda z:splev(z,splr)-splev(xL,splr)
                                minusfunr=lambda z:-splev(z,splr)+splev(xL,splr)
                            it+=1
                            if it>imax:
                                sys.exit('in utilities.py: function periodicSmooth could not find a periodic slice for d1={0},d2={1}'.format(d1,d2))
                    elif np.sign(d_dx_funr(a0))!=np.sign(d_dx_funr(b0)):
                        if np.sign(d_dx_funr(a0))<0.0:
                            opt=minimize_scalar(funr,method=method,bounds=(a0,b0))
                            xR=opt.x
                            if opt.success==False:
                                print(opt.message)
                                sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum in [{2},{3}] for d1={0},d2={1}'.format(d1,d2,a0,b0))
                            #xR=1E10
                            #ainit=a0
                            #it=0
                            #while xR>b0+thresh or xR<a0-thresh:
                            #    #xR=minimize(funr,np.array([a0]),method=method,bounds=((a0-thresh),(b0+thresh))).x
                            #    
                            #    print xR
                            #    ainit+=np.random.rand(1)[0]*1E-4
                            #    it+=1
                            #    if it>imax:
                            #        xdense=np.linspace(xL-1,xR+1.0,5000)
                            #        plt.figure(89)
                            #        plt.plot(xdense,funr(xdense),a0,funr(a0),'o',b0,funr(b0),'o',xR,funr(xR),'d')
                            #        plt.draw()
                            #        plt.show()
                            #        #raw_input()
                            #        sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum for d1={0},d2={1}'.format(d1,d2))

                        elif np.sign(d_dx_funr(a0))>0.0:
                            opt=minimize_scalar(minusfunr,method=method,bounds=(a0,b0))
                            xR=opt.x
                            if opt.success==False:
                                print(opt.message)
                                sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum in [{2},{3}] for d1={0},d2={1}'.format(d1,d2,a0,b0))


                            #xR=1E10
                            #ainit=a0
                            #it=0
                            #while xR>b0+thresh or xR<a0-thresh:
                            #    #xR=minimize_scalar(minusfunr,np.array([a0]),method=method,bounds=bnds).x
                            #    xR=minimize_scalar(minusfunr,method=method,bounds=(a0,b0)).x
                            #    ainit+=np.random.rand(1)[0]*1E-4
                            #    it+=1
                            #    if it>imax:
                            #        sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum for d1={0},d2={1}'.format(d1,d2))


                xtempr=np.linspace(xL,xR,len(xin))


                a0=xin[-1]-Delta
                b0=xin[-1]+Delta

                xL=xin[0]
                funi=lambda z:splev(z,spli)-splev(xL,spli)
                minusfuni=lambda z:-splev(z,spli)+splev(xL,spli)
                d_dx_funi=lambda z:splev(z,spli,der=1)
                if np.abs(funi(a0))<tol or np.abs(funi(b0))<tol:
                    if np.abs(funi(a0))<tol:
                        xR=a0
                    elif np.abs(funi(b0))<tol:
                        xR=b0
                else:
                    if np.sign(d_dx_funi(a0))==np.sign(d_dx_funi(b0)):
                        conv=False
                        it=0
                        #if the sign of the derivative is the same on either side, 
                        #make sure that there is a root of funi between a and b
                        #if plot==True:
                        #    print ('doing bisection')
                        while conv==False:
                            a,b,conv=placeBraket(funi,a0,b0,Delta,thresh=thresh)
                            if conv==True:
                                #there is a root
                                xR=bisection(funi,a,b,tol=tol,maxit=imax)
                                if xR>x[-1]:
                                    conv=False
                                if xR<x[-1]:
                                    break
                            elif conv==False:
                                #there is no root between a0,b0; shift the points a bit and redefine funi
                                xL-=Delta
                                funi=lambda z:splev(z,spli)-splev(xL,spli)
                                minusfuni=lambda z:-splev(z,spli)+splev(xL,spli)
                            it+=1
                            if it>imax:
                                sys.exit('in utilities.py: function periodicSmooth could not find a periodic slice for d1={0},d2={1}'.format(d1,d2))
                    elif np.sign(d_dx_funi(a0))!=np.sign(d_dx_funi(b0)):
                        if np.sign(d_dx_funi(a0))<0.0:
                            opt=minimize_scalar(funi,method=method,bounds=(a0,b0))
                            xR=opt.x
                            if opt.success==False:
                                print(opt.message)
                                sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum in [{2},{3}] for d1={0},d2={1}'.format(d1,d2,a0,b0))

                            #xR=1E10
                            #ainit=a0
                            #it=0
                            #while xR>b0+thresh or xR<a0-thresh:
                            #    #xR=minimize_scalar(funi,np.array([ainit]),method=method,bounds=((a0-thresh),(b0+thresh))).x
                            #    xR=minimize_scalar(funi,method=method,bounds=(a0,b0)).x
                            #    ainit+=np.random.rand(1)[0]*1E-4
                            #    it+=1
                            #    if it>imax:
                            #        sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum for d1={0},d2={1}'.format(d1,d2))
                            #if plot==True:
                            #    #print ('doing minimization')
                            #    plt.figure(189)
                            #    xdense=np.linspace(0.5,1.5,100000)
                            #    plt.plot(xdense,funi(xdense),a0,funi(a0),'o',b0,funi(b0),'o')
                            #    print xR
                            #    print funi(xR)
                            #    plt.draw()
                            #    plt.show()

                        elif np.sign(d_dx_funi(a0))>0.0:
                            opt=minimize_scalar(minusfuni,method=method,bounds=(a0,b0))
                            xR=opt.x
                            if opt.success==False:
                                print(opt.message)
                                sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum in [{2},{3}] for d1={0},d2={1}'.format(d1,d2,a0,b0))

                            #xR=1E10
                            #ainit=a0
                            #it=0
                            #while xR>b0+thresh or xR<a0-thresh:
                            #    #xR=minimize_scalar(minusfuni,np.array([ainit]),method=method,bounds=((a0-thresh),(b0+thresh))).x
                            #    xR=minimize_scalar(minusfuni,method=method,bounds=(a0,b0)).x
                            #    ainit+=np.random.rand(1)[0]*1E-4
                            #    it+=1
                            #    if it>imax:
                            #        sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum for d1={0},d2={1}'.format(d1,d2))


                xtempi=np.linspace(xL,xR,len(xin))

                   
                out[d1,d2,:]=splev(xtempr,splr)+1j*splev(xtempi,spli)
                #if plot==True:
                #    if d1==0 and d2==0:
                #        plt.figure(103)
                #        plt.title('d1={0}, d2={1}'.format(d1,d2))
                #        plt.plot(xtempi,np.imag(out[d1,d2,:]))
                #        
                #        plt.figure(104)
                #        plt.title('d1={0}, d2={1}'.format(d1,d2))
                #        plt.plot(x,sm_ari)
                #
                #        print xtempi
                #        plt.draw()
                #        plt.show()


        if plot==True:
            plt.figure(100)
            plt.plot(xtempr,np.transpose(np.real(np.reshape(out,(D*D,len(xtempr))))),'o')
            plt.ylim([-10,10])
            plt.figure(101)
            plt.plot(xtempi,np.transpose(np.imag(np.reshape(out,(D*D,len(xtempr))))),'o')
            plt.ylim([-10,10])

            plt.figure(102)
            plt.plot(xtempr,np.transpose(np.real(np.reshape(out,(D*D,len(xtempr))))),'o')

            plt.figure(103)
            plt.plot(xtempi,np.transpose(np.imag(np.reshape(out,(D*D,len(xtempr))))),'o')

            plt.draw()
            plt.show()

            

        return out,order
        
    if ar.dtype==np.float64:
        for d1 in range(D):
            for d2 in range(D):
                #sm_ar=savgol_filter(np.real(concar[d1,d2,:]),window_length, polyorder)+1j*savgol_filter(np.imag(concar[d1,d2,:]),window_length, polyorder)
                sm_arr=savgol_filter(np.real(concar[d1,d2,:]),window_length, polyorder)
                splr=splrep(x,sm_arr,k=order,per=0)

                #place the braket at the right boundary of the unit-cell
                a0=xin[-1]-Delta
                b0=xin[-1]+Delta

                xL=xin[0]
                funr=lambda z:splev(z,splr)-splev(xL,splr)
                minusfunr=lambda z:-splev(z,splr)+splev(xL,splr)
                d_dx_funr=lambda z:splev(z,splr,der=1)


                if np.abs(funr(a0))<tol or np.abs(funr(b0))<tol:
                    if np.abs(funr(a0))<tol:
                        xR=a0
                    elif np.abs(funr(b0))<tol:
                        xR=b0
                else:
                    if np.sign(d_dx_funr(a0))==np.sign(d_dx_funr(b0)):
                        conv=False
                        it=0
                        #if the sign of the derivative is the same on either side, 
                        #make sure that there is a root of funr between a and b
                        while conv==False:
                            a,b,conv=placeBraket(funr,a0,b0,Delta,thresh=thresh)
                            if conv==True:
                                #there is a root
                                xR=bisection(funr,a,b,tol=tol,maxit=imax)
                                break
                            elif conv==False:
                                #there is no root between a0,b0; shift the points a bit and redefine funr
                                xL-=Delta
                                funr=lambda z:splev(z,splr)-splev(xL,splr)
                                minusfunr=lambda z:-splev(z,splr)+splev(xL,splr)
                            it+=1
                            if it>imax:
                                sys.exit('in utilities.py: function periodicSmooth could not find a periodic slice for d1={0},d2={1}'.format(d1,d2))
                    elif np.sign(d_dx_funr(a0))!=np.sign(d_dx_funr(b0)):
                        if np.sign(d_dx_funr(a0))<0.0:
                            opt=minimize_scalar(funr,method=method,bounds=(a0,b0))
                            xR=opt.x
                            if opt.success==False:
                                print(opt.message)
                                sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum in [{2},{3}] for d1={0},d2={1}'.format(d1,d2,a0,b0))

                            #xR=1E10
                            #ainit=a0
                            #it=0
                            #while xR>b0+thresh or xR<a0-thresh:
                            #    #xR=minimize_scalar(funr,np.array([a0]),method=method,bounds=((a0-thresh),(b0+thresh))).x
                            #    xR=minimize_scalar(funr,method=method,bounds=(a0,b0)).x
                            #    ainit+=np.random.rand(1)[0]*1E-4
                            #    it+=1
                            #    if it>imax:
                            #        sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum for d1={0},d2={1}'.format(d1,d2))

                        elif np.sign(d_dx_funr(a0))>0.0:
                            opt=minimize_scalar(minusfunr,method=method,bounds=(a0,b0))
                            xR=opt.x
                            if opt.success==False:
                                print(opt.message)
                                sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum in [{2},{3}] for d1={0},d2={1}'.format(d1,d2,a0,b0))

                            #xR=1E10
                            #ainit=a0
                            #it=0
                            #while xR>b0+thresh or xR<a0-thresh:
                            #    #xR=minimize_scalar(minusfunr,np.array([a0]),method=method,bounds=((a0-thresh),(b0+thresh))).x
                            #    xR=minimize_scalar(minusfunr,method=method,bounds=(a0,b0)).x
                            #    ainit+=np.random.rand(1)[0]*1E-4
                            #    it+=1
                            #    if it>imax:
                            #        sys.exit('in utilities.py.periodicSmooth: minimize_scalar could not find a minimum for d1={0},d2={1}'.format(d1,d2))

                xtempr=np.linspace(xL,xR,len(xin))

                out[d1,d2,:]=splev(xtempr,splr)
        return out,order



def smoothArray(ar,window_length, polyorder):
    D=ar.shape[0]
    out=np.zeros(ar.shape).astype(ar.dtype)
    #sm_ar=savgol_filter(np.real(concar[d1,d2,:]),window_length, polyorder)
    if ar.dtype==np.complex128:
        for d1 in range(D):
            for d2 in range(D):
                #sm_ar=savgol_filter(np.real(concar[d1,d2,:]),window_length, polyorder)+1j*savgol_filter(np.imag(concar[d1,d2,:]),window_length, polyorder)
                sm_arr=savgol_filter(np.real(ar[d1,d2,:]),window_length, polyorder)
                sm_ari=savgol_filter(np.imag(ar[d1,d2,:]),window_length, polyorder)
                out[d1,d2,:]=sm_arr+1j*sm_ari
        return out
        
    if ar.dtype==np.float64:
        for d1 in range(D):
            for d2 in range(D):
                sm_arr=savgol_filter(np.real(ar[d1,d2,:]),window_length, polyorder)
                out[d1,d2,:]=sm_arr
        return out

                


#def findPhase(diag0,diag1):
#    method='bounded'
#    a0=-math.pi
#    b0=math.pi
#    phi=np.zeros(diag0.shape)
#    #x=np.linspace(-math.pi,math.pi,100)
#    for n in range(len(diag0)):
#        fun=lambda phi: (np.real(diag0[n])-np.cos(phi)*np.real(diag1[n])-np.sin(phi)*np.imag(diag1[n]))**2.0
#        
#        phi[n]=minimize_scalar(fun,method=method,bounds=(a0,b0)).x
#        #plt.plot(x,fun(x),phi[n],fun(phi[n]),'o')
#        #plt.show()
#
#    return phi


def fun1(U0,U1,V0,V1,phi):
    x=np.real(U1)
    y=np.imag(U1)        
    x0=np.real(U0)
    y0=np.imag(U0)        
    
    v=np.real(V1)
    w=np.imag(V1)        
    v0=np.real(V0)
    w0=np.imag(V0)        

    return (x0-x*np.cos(phi)+np.sin(phi)*y)**2.0+(y0-y*np.cos(phi)-x*np.sin(phi))**2+(v0-v*np.cos(phi)+w*np.sin(phi))**2.0+(w0-w*np.cos(phi)-v*np.sin(phi))**2

def fun2(U0,U1,V0,V1,phi):
    z1=U0-np.exp(1j*phi)*U1
    z2=V0-np.exp(-1j*phi)*V1
    return np.real(np.conj(z1)*z1+np.conj(z2)*z2)


def fun3(U0,U1,V0,V1,phi):
    z1=U0-np.exp(1j*phi)*U1
    z2=V0-np.exp(-1j*phi)*V1
    return np.real(np.conj(z1).dot(z1)+np.conj(z2).dot(z2))


#uses diagonals of matrices to do the phase matching
def findPhase(diagU0,diagU1,diagV0,diagV1):
    method='bounded'
    a0=-2.0*math.pi
    b0=2.0*math.pi
    phi=np.zeros(diagU0.shape)
    #x=np.linspace(-math.pi,math.pi,100)
    for n in range(len(diagU0)):
        fun=fct.partial(fun1,*[diagU0[n],diagU1[n],diagV0[n],diagV1[n]])
        phi[n]=minimize_scalar(fun,method=method,bounds=(a0,b0)).x

    return phi


#assumes that U0,U1 are left unitary matrices and V0,V1 right unitary matrices
#uses full matrices to do the phase matching
def findPhase2(U0,U1,V0,V1):
    D=U0.shape[0]
    method='bounded'
    a0=-2.0*math.pi
    b0=2.0*math.pi
    phi=np.zeros(D)
    #x=np.linspace(-math.pi,math.pi,10000)
    for n in range(D):
        #fun=fct.partial(fun2,*[U0[n,n],U1[n,n],V0[n,n],V1[n,n]])
        fun=fct.partial(fun3,*[U0[:,n],U1[:,n],V0[n,:],V1[n,:]])
        phi[n]=minimize_scalar(fun,method=method,bounds=(a0,b0)).x

    return phi


def matchState(U0,U1):
    #use columns of U0 as reference states:
    Ov=herm(U0).dot(U1)
    #for each vectpr U0[:,n], find the state in U1[:,m] with largest overlap , and bring it to position n
    indy=np.argmax(Ov,1)
    inds=tuple(np.arange(D),indy)
    Uout=np.zeros(U1.shape).astype(U1.dtype)
    for n in range(len(indy)):
        Uout[n,:]=U1[indy[n],:]
    return Uout
        
    

#########################################################################################################################

#check if matrix is diagonal or not
def isDiag(matrix,err=1E-8):
    assert(np.shape(matrix)[0]==np.shape(matrix)[1])
    D=np.shape(matrix)[0]
    diff=np.linalg.norm(np.diag(np.diag(matrix))-matrix)
    if diff<err:
        return True,diff
    if diff>=err:
        return False,diff
    
#check if matrix is a diagonal unitary (i.e. a phase matrix)
def isDiagUnitary(matrix,err=1E-8):
    assert(np.shape(matrix)[0]==np.shape(matrix)[1])
    D=np.shape(matrix)[0]
    diff=np.linalg.norm(np.abs(matrix)-np.eye(D))
    if diff<err:
        return True,diff
    if diff>=err:
        return False,diff


#N is the lenght of the fine grid on which grid is defined
def bisection(grid,position):
    g=np.copy(grid)
    while  True:
        N=len(g)
        i0=(N-N%2)/2
        lgrid=np.copy(g[0:i0])
        rgrid=np.copy(g[i0::])
        if (position>=lgrid[-1])&(position<=rgrid[0]):
            closestleft=np.nonzero(np.array(grid)==lgrid[-1])[0][0]
            closestright=np.nonzero(np.array(grid)==rgrid[0])[0][0]
            break
        if position<lgrid[-1]:
            g=np.copy(lgrid)
            
        if position>rgrid[0]:
            g=np.copy(rgrid)
    return closestleft,closestright



def determineNewStepsize(alpha_,alpha,alphas,nxdots,normxopt,normxoptold,normtol,warmup,it,rescaledepth,factor,itPerDepth,itreset,reset):
    if (normxopt-normxoptold)/normxopt>normtol and it>1:
        rescaledepth+=1
        alpha_/=factor
        itPerDepth=0
        reset=False
        reject=True

    elif (normxopt-normxoptold)/normxopt<=normtol or it==1:
        reject=False
        if reset==True:
            itPerDepth=0
            if rescaledepth>0:
                rescaledepth-=1
            if rescaledepth>0:
                reset=False
            if warmup==True:
                alpha_=alpha/(factor**(rescaledepth))
                if nxdots!=None:
                    if normxopt<nxdots[0]:
                        warmup=False
            if warmup==False:
                if nxdots!=None:
                    if (normxopt>=nxdots[0]):
                        alpha_=alpha/(factor**(rescaledepth))
                    elif (normxopt<nxdots[0]):
                        ind=0
                        while (normxopt<nxdots[ind]):
                            if ind==(len(nxdots)-1):
                                alpha_=alphas[-1]/(factor**(rescaledepth))
                                break
                            if nxdots[ind+1]<=normxopt:
                                alpha_=alphas[ind]/(factor**(rescaledepth))
                                break
                            if nxdots[ind+1]>normxopt:
                                ind+=1


        if reset==False:
            itPerDepth+=1
            if itPerDepth%itreset==0:
                reset=True
        if reject==False:
            normxoptold=normxopt
    return alpha_,rescaledepth,itPerDepth,reset,reject,warmup



def determineNonLinearCGBeta(nlcgupperthresh,nlcglowerthresh,nlcgnormtol,nlcgreset,normxopt,normxoptold,it,itstde,stdereset,dostde,itbeta,printnlcgmessage,printstdemessage):
    if (nlcgupperthresh!=None) and (normxopt<nlcgupperthresh) and (nlcglowerthresh<normxopt):
        if printnlcgmessage==True:
            print
            print ('###############################             doing non-linear conjugate gradient              ####################################')
            printnlcgmessage=False
            printstdemessage=True

        if it>1:
            #fletcher reeves non-linear conjugate gradient
            if (normxopt-normxoptold)/normxopt<=nlcgnormtol:
                beta=normxopt**2/normxoptold**2
            elif (normxopt-normxoptold)/normxopt>nlcgnormtol:
                beta=0.0
            if beta>1.0:
                beta=0.0
            #elif np.abs(normxopt-normxoptold)/normxopt>normtol:
        if it==1:
            beta=0.0

        if beta==0:
            itstde+=1
            dostde=True
            betanew=0.0
        elif beta!=0:
            if (itstde<stdereset) and (dostde==True):
                itstde+=1
                betanew=0.0
            elif (itstde>=stdereset) and (dostde==True):
                betanew=beta
                dostde=False
                itstde=0
            elif (itstde<stdereset) and (dostde==False):
                betanew=beta

        
        if itbeta==nlcgreset:
            betanew=0.0
            itbeta=0
            dostde=True
            itstde=0
        
        if betanew!=0.0:
            itbeta+=1
        if betanew==0.0:
                itbeta=0
    else:
        if printstdemessage==True:
            print
            print ('###############################             doing steepest descent                          ####################################')
            printnlcgmessage=True
            printstdemessage=False
        betanew=0.0

    return betanew,itstde,itbeta,dostde,printnlcgmessage,printstdemessage
                    



