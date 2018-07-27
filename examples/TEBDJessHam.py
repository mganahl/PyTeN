"""
@author: Martin Ganahl
"""
import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)

import numpy as np
import matplotlib.pyplot as plt
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as H
import argparse
import lib.mpslib.mps as mpslib

comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

if __name__ == "__main__":

    parser = argparse.ArgumentParser('TEBDJessHam.py: time evolution for thre Jess Hamiltonian')
    parser.add_argument('--N', help='system size (10)',type=int,default=8)
    parser.add_argument('--switchingstep', help='step at which to switch from TEBD to TDVP (50)',type=int,default=50)        
    parser.add_argument('--D', help='MPS bond dimension (32)',type=int,default=32)
    parser.add_argument('--J', help='S-S interaction (3.0)',type=float,default=3.0)
    parser.add_argument('--chi', help='chi parameter (2.0)',type=float,default=2.0)
    parser.add_argument('--solver', help='TDVP solver (2.0)',type=str,default='RK45')    
    parser.add_argument('--w', help='omega parameter (1.0)',type=float,default=1.0)
    parser.add_argument('--debug',action='store_true', help='debug (1.0)')
    parser.add_argument('--theta1', help='theta1 parameter (1.0)',type=float,default=0.37619240903033213)
    parser.add_argument('--theta2', help='omega parameter (1.0)',type=float,default=2.2815584984041757)
    parser.add_argument('--phi1', help='omega parameter (1.0)',type=float,default=3.676699204438477)
    parser.add_argument('--phi2', help='omega parameter (1.0)',type=float,default=5.569523747854273)        
    parser.add_argument('--dt', help='time step for time evolution (0.01)',type=float,default=0.01)
    parser.add_argument('--thresh', help='truncation threshold (1E-8)',type=float,default=1E-8)    
    parser.add_argument('--Nsteps', help='number of timesteps (1000)',type=int,default=1000)
    parser.add_argument('--msteps', help='measurement step (every 20)',type=int,default=20)
    parser.add_argument('--mstepsTDVP', help='measurement step (every 20)',type=int,default=2)        
    parser.add_argument('--filename', help='filename for output (_JessHamiltonian)',type=str,default='_JessHamiltonian')
    args=parser.parse_args()

    N=args.N
    #initializes a product state with a specific arrangement of up and down spins
    #state is an array of length N defining which state should be put at each site;
    #values at state[n] can be in {0,1,..,d-1}
    def blochstate(theta,phi):
        return np.asarray([np.cos(theta/2.)+0j,np.sin(theta/2.)*np.exp(1j*phi)])

    if not args.debug:
        #state=[np.kron(blochstate(args.theta1,args.phi1),blochstate(args.theta2,args.phi2))]*args.N
        state=[np.kron([1,0],[1,0])]*N
        state[int(N/2-5)]=np.kron([0,1],[0,1])
        state[int(N/2+5)]=np.kron([0,1],[0,1])
        
    else:
        state=[np.kron([1,0],[1,0])]*N
        state[int(N/2-5)]=np.kron([0,1],[0,1])
        state[int(N/2+5)]=np.kron([0,1],[0,1])
        statexxz=[np.asarray([1,0])]*N
        statexxz[int(N/2-5)]=np.asarray([0,1])
        statexxz[int(N/2+5)]=np.asarray([0,1])        
        mpsxxz=mpslib.MPS.productState(statexxz,obc=True,dtype=complex) #this mps for now has a maximally allowed bond dimension mps._D=1;
        mpsxxz.position(N)
        mpsxxz.position(0)
        mpoxxz=H.XXZ(Jz=args.J*np.ones(args.N-1),Jxy=np.ones(args.N-1),Bz=np.ones(N)*args.w,obc=True)
        enginexxz=en.TimeEvolutionEngine(mpsxxz,mpoxxz,args.filename)
        Sxxz=np.zeros((args.Nsteps,N,3)).astype(complex) #container for holding the measurements
        
    mps=mpslib.MPS.productState(state,obc=True,dtype=complex) #this mps for now has a maximally allowed bond dimension mps._D=1;
    #normalize the state by sweeping the orthogonalizty center once back and forth through the system
    mps.position(N)
    mps.position(0)

    #initialize an MPO (MPOs are defined in lib.mpslib.Hamiltonians)
    #the MPO class in Hamiltonians implements a routine MPO.twoSiteGate(m,n,dt), which 
    #returns the exponential exp(dt*h(m,n)), where h(m,n) is the local Hamiltonian contribution 
    #acting on sites m and n
    mpo=H.JessHamiltonian(J=args.J*np.ones(args.N-1),w=np.ones(args.N)*args.w,chi=np.ones(args.N)*args.chi,obc=True)


    #initialize a TEBDEngine with an mps and an mpo
    #you don't have to pass an mpo here; the engine mererly assumes that
    #the object passed implements the memberfunction object.twoSiteGate(m,n,dt)
    #which should return an twosite gate
    engine=en.TimeEvolutionEngine(mps,mpo,args.filename)

    dt=-1j*args.dt
    numsteps=args.msteps   #numnber of steps to be taken in between measurements
    Dmax=args.D      #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
                 #adapted to this value in the TEBDEngine
    thresh=args.thresh  #truncation threshold
    Nmax=args.Nsteps    #number of measurements
    S=np.zeros((Nmax,N,3)).astype(complex) #container for holding the measurements

    T=np.zeros((Nmax,N,3)).astype(complex) #container for holding the measurements    
    plt.ion()
    sigma_x=np.asarray([[0,1],[1,0]]).astype(complex)
    sigma_y=np.asarray([[0,-1j],[1j,0]]).astype(complex)
    sigma_z=np.diag([1,-1]).astype(complex)

    #comm_sx_sy=[comm(np.kron(sigma_x,np.eye(2)),np.kron(sigma_y,np.eye(2))) for n in range(N)]  #a list of local operators to be measured
    #comm_sz_sx=[comm(np.kron(sigma_z,np.eye(2)),np.kron(sigma_x,np.eye(2))) for n in range(N)]  #a list of local operators to be measured    
    sx=[np.kron(sigma_x,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    taux=[np.kron(np.eye(2),sigma_x) for n in range(N)]  #a list of local operators to be measured
    sy=[np.kron(sigma_y,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    tauy=[np.kron(np.eye(2),sigma_y) for n in range(N)]  #a list of local operators to be measured    
    sz=[np.kron(sigma_z,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    tauz=[np.kron(np.eye(2),sigma_z) for n in range(N)]  #a list of local operators to be measured    

    #Tyexact=[np.conj(state[0]).dot(tauy[0]).dot(state[0])]*N
    #Syexact=[np.conj(state[0]).dot(sy[0]).dot(state[0])]*N
    it=0  #counts the total iteration number
    tw=0  #accumulates the truncated weight (see below)
    itxxz=0  #counts the total iteration number
    twxxz=0  #accumulates the truncated weight (see below)
    
    #data=np.load('ZY_07-26_10h57m52s_complete.npy')
    data=np.load('ZY_07-26_10h57m52s_complete.npy')
    #commdata=np.load('ZY_07-26_12h27m04s_complete.npy')
    #data=np.load('magnetization-data-n8-d30.npy')
    
    for n in range(Nmax):
        #do numsteps TEBD steps 
        #measure the operators 
        #S[n,:,0],S[n,:,1],S[n,:,2],CSxSy,CSzSx=engine._mps.measureList(sx),engine._mps.measureList(sy),engine._mps.measureList(sz),engine._mps.measureList(comm_sx_sy),engine._mps.measureList(comm_sz_sx)



        S[n,:,1],S[n,:,0],S[n,:,2]=engine._mps.measureList(sy),engine._mps.measureList(sx),engine._mps.measureList(sz)
        T[n,:,0],T[n,:,1],T[n,:,2]=engine._mps.measureList(taux),engine._mps.measureList(tauy),engine._mps.measureList(tauz)        
        if args.debug:        
            Sxxz[n,:,1],Sxxz[n,:,0],Sxxz[n,:,2]=enginexxz._mps.measureList([sigma_y]*N),enginexxz._mps.measureList([sigma_x]*N),enginexxz._mps.measureList([sigma_z]*N)
        if not args.debug:                    
            #plt.figure(2)
            #plt.clf()
            #plt.title('[Sx,Sy]')
            #plt.subplot(2,2,1)
            #plt.plot(range(N),np.real(CSxSy),range(N),np.real(2j*S[n,:,2]),'x')
            #plt.ylabel(r'$\Re\langle[\sigma_x,\sigma_y] \rangle$')
            #plt.legend([r'$\langle[\sigma_x,\sigma_y]\rangle$',r'$\langle 2j\sigma_z\rangle$'])
            #
            #plt.subplot(2,2,2)        
            #plt.plot(range(N),np.imag(CSxSy),range(N),np.imag(2j*S[n,:,2]),'x')
            #plt.ylabel(r'$\Im\langle[\sigma_x,\sigma_y] \rangle$')        
            #plt.legend([r'$\langle[\sigma_x,\sigma_y]\rangle$',r'$\langle 2j\sigma_z\rangle$'])        
            #plt.subplot(2,2,3)
            #plt.plot(range(N),np.real(CSzSx),range(N),np.real(2j*S[n,:,1]),'x')
            #plt.ylabel(r'$\Re\langle[\sigma_z,\sigma_x] \rangle$')                
            #plt.legend([r'$\langle[\sigma_z,\sigma_x]\rangle$',r'$\langle 2j\sigma_y\rangle$'])                        
            #plt.subplot(2,2,4)        
            #plt.plot(range(N),np.imag(CSzSx),range(N),np.imag(2j*S[n,:,1]),'x')
            #plt.ylabel(r'$\Im\langle[\sigma_z,\sigma_x] \rangle$')                        
            #plt.legend([r'$\langle[\sigma_z,\sigma_x]\rangle$',r'$\langle 2j\sigma_y\rangle$'])                
            
            plt.figure(1)
            plt.clf()
            plt.subplot(3,2,1)
            plt.title(r'$\sigma_x$')
            plt.plot(range(N),np.real(S[n,:,0]))#,range(N),data[n,:,0],'x')
            #plt.ylim([-1,1])
            plt.subplot(3,2,2)
            plt.title('Tx')        
            plt.plot(range(N),np.real(T[n,:,0]))#,range(N),data[n,:,3],'x')        
            #plt.ylim([-1,1])
            plt.subplot(3,2,3)
            plt.title('Sy')
            plt.plot(range(N),np.real(S[n,:,1]))#,range(N))#,data[n,:,1],'x')
            #plt.ylim([-1,1])
            plt.subplot(3,2,4)
            plt.title('Ty')
            plt.plot(range(N),np.real(T[n,:,1]))#,range(N))#,data[n,:,4],'x')
            #plt.ylim([-1,1])
            plt.subplot(3,2,5)
            plt.title('Sz')        
            plt.plot(range(N),np.real(S[n,:,2]))#,range(N))#,data[n,:,2],'x')
            #plt.ylim([-1,1])
            plt.subplot(3,2,6)
            plt.title('Tz')        
            plt.plot(range(N),np.real(T[n,:,2]))#,range(N))#,data[n,:,5],'x')        
            #plt.ylim([-1,1])
            plt.draw()
            plt.show()
            plt.pause(0.01)

        else:
            plt.figure(1,figsize=(10,8))
            plt.clf()
            plt.subplot(3,2,1)
            plt.title(r'$\sigma_x$')
            plt.plot(range(N),np.real(S[n,:,0]),range(N),np.real(Sxxz[n,:,0]),'x')
            plt.legend(['Jess','XXZ'])
            plt.ylim([-1,1])
            plt.subplot(3,2,2)
            plt.title(r'$\tau_x$')        
            plt.plot(range(N),np.real(T[n,:,0]))
            plt.legend(['Jess','XXZ'])        
            plt.ylim([-1,1])
            plt.subplot(3,2,3)
            plt.title(r'$\sigma_y$')        
            plt.plot(range(N),np.real(S[n,:,1]),range(N),np.real(S[n,:,1]),'x')
            plt.legend(['Jess','XXZ'])        
            plt.ylim([-1,1])
            plt.subplot(3,2,4)
            plt.title(r'$\tau_y$')                
            plt.plot(range(N),np.real(T[n,:,1]))
            plt.legend(['Jess','XXZ'])        
            plt.ylim([-1,1])
            plt.subplot(3,2,5)
            plt.title(r'$\sigma_z$')        
            plt.plot(range(N),np.real(S[n,:,2]),range(N),np.real(S[n,:,2]),'x')
            plt.legend(['Jess','XXZ'])        
            plt.ylim([-1,1])
            plt.subplot(3,2,6)
            plt.title(r'$\tau_z$')                        
            plt.plot(range(N),np.real(T[n,:,2]))
            plt.legend(['Jess','XXZ'])        
            plt.ylim([-1,1])
            
            plt.draw()
            plt.show()
            plt.pause(0.01)
        if it<args.switchingstep:
            tw,it=engine.doTEBD(dt=dt,numsteps=numsteps,Dmax=Dmax,tr_thresh=thresh,cnterset=it,tw=tw)
        else:
            if it==args.switchingstep:
                engine.initializeTDVP()
            it=engine.doTDVP(dt=1.0*args.msteps/args.mstepsTDVP*dt,numsteps=args.mstepsTDVP,cnterset=it,solver=args.solver)

        if args.debug:        
            #print()
            twxxz,itxxz=enginexxz.doTEBD(dt=dt,numsteps=numsteps,Dmax=Dmax,tr_thresh=thresh,cnterset=itxxz,tw=twxxz,verbose=0)
            #print()        
