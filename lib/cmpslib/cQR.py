#!/usr/bin/env python
import numpy as np

#this is a tenth order series expansion of 1/sqrt(1+ax+bx**2)
fun=lambda a,b,x:-a*x/2.0 + 1./8.*(3.*a**2 - 4.*b)*x**2 + (-5.*a**3/16.0 + 3.*a*b/4.)*x**3 + 1./128.* (35.*a**4 - 120.*a**2*b + 48.*b**2)*x**4     +    1./256.*(-63*a**5 + 280*a**3*b - 240*a*b**2)*x**5\
     + (231.*a**6 - 1260.*a**4.*b + 1680.*a**2.*b**2. - 320.*b**3.)/1024.0*x**6  + (-429.*a**7. + 2772.*a**5.*b - 5040.*a**3.*b**2. + 2240.*a*b**3.)/2048.0*x**7. +  (6435.*a**8. - 48048.*a**6.*b + 110880.*a**4.*b**2 - 80640.*a**2.*b**3. + 8960.*b**4.)/32768.0*x**8.+\
     (-12155.*a**9. + 102960.*a**7.*b - 288288.*a**5.*b**2. + 295680.*a**3.*b**3. - 80640.*a*b**4.)/65536.0*x**9.+ (46189.*a**10 - 437580.*a**8.*b + 1441440.*a**6.*b**2. - 1921920.*a**4.*b**3. +  887040.*a**2.*b**4. - 64512.*b**5.)/262144.0*x**10.

fun2=lambda a,b,x:-a/2.0 + 1./8.*(3.*a**2 - 4.*b)*x + (-5.*a**3/16.0 + 3.*a*b/4.)*x**2 + 1./128.* (35.*a**4 - 120.*a**2*b + 48.*b**2)*x**3     +    1./256.*(-63*a**5 + 280*a**3*b - 240*a*b**2)*x**4\
     + (231.*a**6 - 1260.*a**4.*b + 1680.*a**2.*b**2. - 320.*b**3.)/1024.0*x**5  + (-429.*a**7. + 2772.*a**5.*b - 5040.*a**3.*b**2. + 2240.*a*b**3.)/2048.0*x**6. +  (6435.*a**8. - 48048.*a**6.*b + 110880.*a**4.*b**2 - 80640.*a**2.*b**3. + 8960.*b**4.)/32768.0*x**7.+\
     (-12155.*a**9. + 102960.*a**7.*b - 288288.*a**5.*b**2. + 295680.*a**3.*b**3. - 80640.*a*b**4.)/65536.0*x**8.+ (46189.*a**10 - 437580.*a**8.*b + 1441440.*a**6.*b**2. - 1921920.*a**4.*b**3. +  887040.*a**2.*b**4. - 64512.*b**5.)/262144.0*x**9.

#this is a tenth order series expansion of 1/sqrt(1+ax+bx**2)
#returns the first 11 coefficients of the taylor expansion of  1/sqrt(1+ax+bx**2) in a vector h , around x=0
#the evaluation is not optimal;
def taylor(a,b,dtype):
    h=np.zeros(10,dtype=dtype)
    h[0]=-a/2.0 
    h[1]=1./8.*(3.*a**2 - 4.*b) 
    h[2]=(-5.*a**3/16.0 + 3.*a*b/4.)
    h[3]=(35./128.*a**4 - 120./128.*a**2*b + 48./128.*b**2)
    h[4]=(-63./256.*a**5. + 280./256.*a**3.*b - 240./256.*a*b**2.)
    h[5]=(231./1024.0*a**6. - 1260./1024.0*a**4.*b + 1680./1024.0*a**2.*b**2. - 320./1024.0*b**3.)
    h[6]=(-429./2048.0*a**7. + 2772./2048.0*a**5.*b - 5040./2048.0*a**3.*b**2. + 2240./2048.0*a*b**3.)
    h[7]=(6435./32768.0*a**8. - 48048./32768.0*a**6.*b + 110880./32768.0*a**4.*b**2. - 80640./32768.0*a**2.*b**3. + 8960./32768.0*b**4.)
    h[8]=(-12155./65536.0*a**9. + 102960./65536.0*a**7.*b - 288288./65536.0*a**5.*b**2. + 295680./65536.0*a**3.*b**3. - 80640./65536.0*a*b**4.)
    h[9]=(46189./262144.0*a**10. - 437580./262144.0*a**8.*b + 1441440./262144.0*a**6.*b**2 - 1921920./262144.0*a**4.*b**3. +  887040./262144.0*a**2.*b**4. - 64512./262144.0*b**5.)
    return h

def taylor2(a,b,dtype):
    rest=np.zeros(10,dtype=dtype)
    pa=np.zeros((10,10),dtype=dtype)
    pb=np.zeros(10,dtype=dtype)
    pa_b_and_bb=np.zeros(10,dtype=dtype)

    pa[1,0]=-1./2.0
    pa[2,1]=3./8.,
    pa[3,2]=-5./16.0
    pa[4,3]=35./128.
    pa[5,4]=-63./256.
    pa[6,5]=231./1024.
    pa[7,6]=-429./2048.
    pa[8,7]=6435./32768.
    pa[9,8]=-12155./65536.0
    pa[10,9]=46189./262144.0
    
    pb[1,1]=-1./2.
    pb[2,3]=48./128.
    pb[3,5]=-320./1024.
    pb[4,7]=8960./32768.0
    pb[5,9]=-64512./262144.0

    pa_b_and_bb[1,2]=3./4.*b
    pa_b_and_bb[1,4]=-240./256.*b**2.
    pa_b_and_bb[2,3]=-120./128.*b
    pa_b_and_bb[2,5]=+1680./1024.0*b**2.
    pa_b_and_bb[3,4]=280./256.*b
    pa_b_and_bb[3,6]=-5040./2048.*b**2.
    pa_b_and_bb[4,5]=-1260./1024.*b
    pa_b_and_bb[4,7]=+110880./32768.0*b**2.
    pa_b_and_bb[5,6]=2772./2048.*b
    pa_b_and_bb[5,8]=-288288./65536.0*b**2.
    pa_b_and_bb[6,7]=-48048./32768.0*b
    pa_b_and_bb[6,9]=+1441440./262144.0*b**2
    pa_b_and_bb[7,8]=102960./65536.0*b
    pa_b_and_bb[8,9]=-437580./262144.0*b

    h[6]=(+2240./2048.0*a*b**3.)
    h[7]=(-80640./32768.0*a**2.*b**3.)
    h[8]=(+295680./65536.0*a**3.*b**3. - 80640./65536.0*a*b**4.)
    h[9]=(-1921920./262144.0*a**4.*b**3. +  887040./262144.0*a**2.*b**4. )
    return 



def normalize(cQ,cR,dx,c):
    a,b=scalarProduct(cQ,cR,c,cQ,cR,c)
    h=taylor(a,b,type(a))
    show=False
    if np.abs(a)>1E10 or np.abs(b)>1E10:
        show=True
        #raw_input()
        #return None,None
    eps1=dx**(np.arange(1,len(h)+1))
    eps2=dx**(np.arange(len(h)))
    #t2=h[-1]
    #for n in range(len(h)-2,-1,-1):
    #    t2=t1*dx+h[n]
    #t1=h[-1]*dx
    #for n in range(len(h)-2,-1,-1):
    #    t1=(t1+h[n])*dx

    t1=np.dot(eps1,h) #NEVER DO THAT
    t2=np.dot(eps2,h) #NEVER DO THAT
    cQnormdiag=cQ[c]+t2+cQ[c]*t1
    cQnorm=np.copy(cQ*(1.0+t1*np.ones(len(cQ))))
    cQnorm[c]=cQnormdiag
    cRnorm=np.copy(cR*(1.0+t1*np.ones(len(cQ))))
    return cQnorm,cRnorm,show


#i1 and 1i are the indices where the 1 has to be put in the mps matrix    
#returns the two leading order coefficients (dx, dx**2) of the scalar product of two columns of a cMPS  matrix (1+dx Q;sqrt(dx)R); 
#Q1,R1 is complex conjugated in the operation
#checked: works correctly for real and complex Q,R
def scalarProduct(Q1,R1,i1,Q2,R2,i2):
    r1=np.conj(Q1[i2])+Q2[i1]+np.dot(np.conj(R1),R2)
    r2=np.dot(np.conj(Q1),Q2)
    return r1,r2


def subtract(Q1,R1,Q2,R2,r1,r2,dx,index):
    vecQ=dx*np.copy(Q2)
    vecQ[index]=1.0+dx*Q2[index]
    outQ=Q1-(r1+dx*r2)*vecQ
    outR=R1-dx*(r1+dx*r2)*R2
    return outQ,outR


#this is a routine for QR decomposition of cMPS matrices Q,R, defined on a grid dx, that is the corresponding MPS matrix 
#has the form (1+dxQ;sqrt(dx)R); as opposed to using a regular QR decomposition as provided by the numpy method, this method
#works for ARBITRARILY SMALL dx (1E-100000 is possible) by exploiting the cMPS structure. It becomes less accurate for dx large, 
#due to the use of a taylor expansion of 1/sqrt(1+a*dx+b*dx**2) of tenth order. The accuracy for large dx gets worse
#when increasing the bond dimension. for D=100, and dx=0.01, the e.g. left normalization of the resulting matrices is already quite bad
#we use the following shorthand notation: V(c,Q,R)=(1(c)+dx*Q[:,c];sqrt(dx)*R[:,c]), where 1(c) is a vector with 1.0 at position c and 0 else where
#INPUT: two cMPS matrices Q,R, and a possible lattice spacing dx. dx=0.0 by default.
#RETURNS: matrices Qn,Rn and upper triangular UT, such that Q=Qn+UT+dx*Qn.dot(UT), R=Rl*(1.0+dx*UT) and Qn+herm(Qn)+herm(Rn)*Rn+dx*herm(Qn)*Qn=0 (orthonormality);
#         in other words: (1.0+dx*Q;sqrt(dx)*R)=(1.0+dx*Qn;sqrt(dx)*Rn)*(1.0+dx*UT)
def cQR(Q,R,dx=0.0):
    dtype=type(Q[0,0])
    D=np.shape(Q)[0]
    Qnorm=np.zeros((D,D),dtype=dtype)
    Rnorm=np.zeros((D,D),dtype=dtype)
    UTMatrix=np.zeros((D,D),dtype=dtype)
    column=0

    Qn,Rn,show=normalize(Q[:,0],R[:,0],dx,column)

    Qnorm[:,column]=np.copy(Qn)
    Rnorm[:,column]=np.copy(Rn)
    r1,r2=scalarProduct(Qn,Rn,0,Q[:,0],R[:,0],0)
    UTMatrix[0,0]=r1+dx*r2
    if show==True:
        return None,None,None,False
        #print (Q)
        #print (R)
        #raw_input()
    for column in range(1,D):
        #grab a new column vector from the matrix (1+dx Q;sqrt(dx) R):
        cQ=(Q[:,column])
        cR=(R[:,column])
        #cQ=np.copy(Q[:,column])
        #cR=np.copy(R[:,column])

        cQtemp=np.copy(Q[:,column])
        cRtemp=np.copy(R[:,column])
        #make it orthonormal to all previously calculated columns:
        for c2 in range(column):
            #take the current column V(column,Q,R) and calculate the overlap
            #with the previously calculated (orthonormal) vectors V(c2,Qnorm,Rnorm)
            #scalarProduct returns the coefficients of the two leading orders r1 and r2 of the scalar product (dx*r1 and dx**2*r2).
            #if column<>c2, this is already the full scalar product; if column==c2, then
            #the user has to add 1.0 to get the true scalar product 1.0+r1*dx+r2*dx**2
            r1,r2=scalarProduct(Qnorm[:,c2],Rnorm[:,c2],c2,cQ,cR,column)
            UTMatrix[c2,column]=r1+dx*r2
            #now subtract dx*(r1+r2*dx)*V(c2,Qnorm,Rnorm) to make V(column,Q,R) orthogonal to it; 
            #Here lies the advantage of using Q and R, instead of the MPS matrix: V(column,Q,R) has one big entry 1.0+dx cQ(column) at column, and 
            #infinitesimal small ones every where else. The goal is changing the corrections of order dx of V(column,Q,R) so that V becomes
            #orthogonal to the previous vectors, but in fact, it is almost orthogonal already (up to corrections of order dx).
            #Operating directly on V(column,Q,R) by adding numbers of order of dx or smaller, and the subtracting out 1.0 and dividing by dx
            #is sensitive to round off errors at machine precision. Instead, we directly perform all opertation on the corrections of
            #order dx, since we know them already
            
            #subtract returns
            #the order dx and sqrt(dx) components of the resulting column, disregarding any 1.0 terms
            #cQtemp,cRtemp=subtract(cQtemp,cRtemp,Qnorm[:,c2],Rnorm[:,c2],r1,r2,dx,c2)

            cQtemp=cQtemp-dx*(r1+dx*r2)*Qnorm[:,c2]
            cQtemp[c2]=cQtemp[c2]-(r1+dx*r2)
            cRtemp=cRtemp-dx*(r1+dx*r2)*Rnorm[:,c2]
            #vecQ=dx*np.copy(Q2)
            #vecQ[index]=1.0+dx*Q2[index]
            #outQ=Q1-(r1+dx*r2)*vecQ
            #outR=R1-dx*(r1+dx*r2)*R2

        #now normalize cQtemp and cRtemp
        Qn,Rn,show=normalize(cQtemp,cRtemp,dx,column)
        if show==True:
            return None,None,None,False            
            #print (Q)
            #print (R)
            #raw_input()

        r1,r2=scalarProduct(Qn,Rn,column,cQ,cR,column)
        UTMatrix[column,column]=r1+dx*r2

        Qnorm[:,column]=np.copy(Qn)
        Rnorm[:,column]=np.copy(Rn)
    return Qnorm,Rnorm,UTMatrix,True
