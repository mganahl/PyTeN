"""
@author: Martin Ganahl
"""
import sys,os
root=os.getcwd()
os.chdir('../')
sys.path.append(os.getcwd())#add parent directory to path
os.chdir(root)

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import lib.mpslib.engines as en
import lib.mpslib.Hamiltonians as H
import argparse
import lib.mpslib.mps as mpslib
import datetime
comm=lambda x,y:np.dot(x,y)-np.dot(y,x)
anticomm=lambda x,y:np.dot(x,y)+np.dot(y,x)
herm=lambda x:np.conj(np.transpose(x))

def loadevoMPS(filename):
    a=sp.load(filename)
    times=a[:,0]
    mpss=[]
    for t in range(len(times)):
        l=[np.transpose(np.asarray(a[t][1][site][:,:,:]),(1,2,0)) for site in range(1,len(a[t][1]))]
        mpss.append(mpslib.MPS.fromList(l))
        mpss[-1].position(0)
        mpss[-1].position(len(mpss[-1]))

    N=len(mpss[0])
    sigma_x=np.asarray([[0,1],[1,0]]).astype(complex)
    sigma_y=np.asarray([[0,-1j],[1j,0]]).astype(complex)
    sigma_z=np.diag([1,-1]).astype(complex)


    sx=[np.kron(sigma_x,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    taux=[np.kron(np.eye(2),sigma_x) for n in range(N)]  #a list of local operators to be measured
    sy=[np.kron(sigma_y,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    tauy=[np.kron(np.eye(2),sigma_y) for n in range(N)]  #a list of local operators to be measured    
    sz=[np.kron(sigma_z,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    tauz=[np.kron(np.eye(2),sigma_z) for n in range(N)]  #a list of local operators to be measured    
    
    Sevo=np.zeros((len(times),len(mpss[0]),3)).astype(complex) #container for holding the measurements
    Tevo=np.zeros((len(times),len(mpss[0]),3)).astype(complex) #container for holding the measurements
    n=0
    for mps in mpss:
        Sevo[n,:,1],Sevo[n,:,0],Sevo[n,:,2]=mps.measureList(sy),mps.measureList(sx),mps.measureList(sz)
        Tevo[n,:,0],Tevo[n,:,1],Tevo[n,:,2]=mps.measureList(taux),mps.measureList(tauy),mps.measureList(tauz)        
        n+=1

    return times,Sevo,Tevo        
if __name__ == "__main__":

    parser = argparse.ArgumentParser('Branching.py: time evolution for thre Branching Hamiltonian')
    parser.add_argument('--N', help='system size (10)',type=int,default=8)
    #parser.add_argument('--switchingstep', help='step at which to switch from TEBD to TDVP (50)',type=int,default=50)        
    parser.add_argument('--D', help='MPS bond dimension (32)',type=int,default=32)
    parser.add_argument('--cp', help='check pointing steps (None)',type=int)    
    parser.add_argument('--stiffness', help='S-S interaction (3.0)',type=float,default=3.0)
    parser.add_argument('--chi', help='chi parameter (2.0)',type=float,default=2.0)
    #parser.add_argument('--solver', help='TDVP solver (2.0)',type=str,default='RK45')    
    parser.add_argument('--w', help='omega parameter (1.0)',type=float,default=1.0)
    parser.add_argument('--debug',action='store_true', help='debug (False)')
    parser.add_argument('--plot',action='store_true', help='plot intermediate results (False)')    
    parser.add_argument('--keep_cp',action='store_true',help='if given, keep the checkpoint files (False)')    
    parser.add_argument('--compare',action='store_true', help='debug (False)')    
    parser.add_argument('--theta1', help='theta1 parameter (1.0)',type=float,default=0.37619240903033213)
    parser.add_argument('--theta2', help='omega parameter (1.0)',type=float,default=2.2815584984041757)
    parser.add_argument('--phi1', help='omega parameter (1.0)',type=float,default=3.676699204438477)
    parser.add_argument('--phi2', help='omega parameter (1.0)',type=float,default=5.569523747854273)        
    parser.add_argument('--dt', help='time step for time evolution (0.01)',type=float,default=0.01)
    parser.add_argument('--thresh', help='truncation threshold (1E-8)',type=float,default=1E-8)    
    parser.add_argument('--Nsteps', help='number of timesteps (1000)',type=int,default=1000)
    parser.add_argument('--msteps', help='measurement step (every 20)',type=int,default=20)
    parser.add_argument('--mstepsTDVP', help='measurement step (every 20)',type=int,default=2)        
    parser.add_argument('--filename', help='filename for output (_Branching)',type=str,default='_Branching')
    parser.add_argument('--loadevoMPS', help='file name of an evoMPS file (None)',type=str)
    args=parser.parse_args()    
    if args.loadevoMPS and not args.compare:
        args.compare=True

    root=os.getcwd()
    date=datetime.datetime.now()
    today=str(date.year)+str(date.month)+str(date.day)
    filename=today+args.filename+'_N{0}_D{1}_stiff{2}_chi{3}_w{4}_dt{5}'.format(args.N,args.D,args.stiffness,args.chi,args.w,args.dt)
    if os.path.exists(filename):
        print('folder',filename,'exists already. Resuming will likely overwrite existing data. Hit enter to confirm')
        input()
    elif not os.path.exists(filename):
        os.mkdir(filename)
    os.chdir(filename)
    parameters=vars(args)
    with open('parameters.dat','w') as f:
        for n in parameters:
            f.write(n + ': {0}\n'.format(parameters[n]))
    f.close()
    
    N=args.N
    J=-args.N*args.stiffness    
    #initializes a product state with a specific arrangement of up and down spins
    #state is an array of length N defining which state should be put at each site;
    #values at state[n] can be in {0,1,..,d-1}
    def blochstate(theta,phi):
        return np.asarray([np.cos(theta/2.)+0j,np.sin(theta/2.)*np.exp(1j*phi)])

    #if not args.debug:
    #state=[np.kron(blochstate(args.theta1,args.phi1),blochstate(args.theta2,args.phi2))]*args.N
    state=[np.kron(blochstate(args.theta1,args.phi1),blochstate(args.theta2,args.phi2))]*args.N    
    if args.compare:
        statexxz=[blochstate(args.theta1,args.phi1)]*N
        statexxz2=[blochstate(args.theta2,args.phi2)]*N    
        mpsxxz=mpslib.MPS.productState(statexxz,obc=True,dtype=complex) #this mps for now has a maximally allowed bond dimension mps._D=1;
        mpsxxz.position(N)
        mpsxxz.position(0)
        mpoxxz=H.XXZIsing(J=J*np.ones(args.N-1),w=args.w*np.ones(N),obc=True)
        enginexxz=en.TimeEvolutionEngine(mpsxxz,mpoxxz,filename)
        Sxxz=np.zeros((args.Nsteps,N,3)).astype(complex) #container for holding the measurements
        mpsxxz2=mpslib.MPS.productState(statexxz2,obc=True,dtype=complex) #this mps for now has a maximally allowed bond dimension mps._D=1;
        mpsxxz2.position(N)
        mpsxxz2.position(0)
        enginexxz2=en.TimeEvolutionEngine(mpsxxz2,mpoxxz,filename)
        Sxxz2=np.zeros((args.Nsteps,N,3)).astype(complex) #container for holding the measurements
    
    mps=mpslib.MPS.productState(state,obc=True,dtype=complex) #this mps for now has a maximally allowed bond dimension mps._D=1;
    #normalize the state by sweeping the orthogonalizty center once back and forth through the system
    mps.position(N)
    mps.position(0)

    #initialize an MPO (MPOs are defined in lib.mpslib.Hamiltonians)
    #the MPO class in Hamiltonians implements a routine MPO.twoSiteGate(m,n,dt), which 
    #returns the exponential exp(dt*h(m,n)), where h(m,n) is the local Hamiltonian contribution 
    #acting on sites m and n

    mpo=H.BranchingHamiltonian(J=J*np.ones(args.N-1),w=np.ones(args.N)*args.w,chi=np.ones(args.N)*args.chi,obc=True)
    
    #initialize a TEBDEngine with an mps and an mpo
    #you don't have to pass an mpo here; the engine mererly assumes that
    #the object passed implements the memberfunction object.twoSiteGate(m,n,dt)
    #which should return an twosite gate
    engine=en.TimeEvolutionEngine(mps,mpo,filename)
    
    dt=-1j*args.dt/np.abs(J)
    Dmax=args.D   #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be
                 #adapted to this value in the TEBDEngine
    thresh=args.thresh  #truncation threshold
    Nmax=args.Nsteps    #number of simulations steps
    NMsteps=int(np.floor(Nmax/args.msteps)+1)
    
    S=np.zeros((NMsteps,N+1,3)).astype(complex) #container for holding the measurements
    T=np.zeros((NMsteps,N+1,3)).astype(complex) #container for holding the measurements    
    plt.ion()
    sigma_x=np.asarray([[0,1],[1,0]]).astype(complex)
    sigma_y=np.asarray([[0,-1j],[1j,0]]).astype(complex)
    sigma_z=np.diag([1,-1]).astype(complex)
    lams=np.zeros((NMsteps,args.D))
    truncWeight=np.zeros((NMsteps))
    #comm_sx_sy=[comm(np.kron(sigma_x,np.eye(2)),np.kron(sigma_y,np.eye(2))) for n in range(N)]  #a list of local operators to be measured
    #comm_sz_sx=[comm(np.kron(sigma_z,np.eye(2)),np.kron(sigma_x,np.eye(2))) for n in range(N)]  #a list of local operators to be measured    
    sx=[np.kron(sigma_x,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    taux=[np.kron(np.eye(2),sigma_x) for n in range(N)]  #a list of local operators to be measured
    sy=[np.kron(sigma_y,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    tauy=[np.kron(np.eye(2),sigma_y) for n in range(N)]  #a list of local operators to be measured    
    sz=[np.kron(sigma_z,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    tauz=[np.kron(np.eye(2),sigma_z) for n in range(N)]  #a list of local operators to be measured    

    #data=np.load('ZY_07-26_10h57m52s_complete.npy')
    #data=np.load('ZY_07-26_10h57m52s_complete.npy')
    
    #filename='Matrix_traj_07-27_12h18m59s.npy'
    if args.loadevoMPS!=None:
        #filename='Matrix_traj_07-27_12h25m40s.npy'
        tevo,Sevo,Tevo=loadevoMPS(root+'/'+args.loadevoMPS)
        for n in range(len(args.loadevoMPS)-1,-1,-1):
            if args.loadevoMPS[n]=='/':
                break
        name=args.loadevoMPS[n+1:-4]
        np.save('timeevo'+name,tevo)        
        np.save('Sevo'+name,Sevo)
        np.save('Tevo'+name,Tevo)    
    #commdata=np.load('ZY_07-26_12h27m04s_complete.npy')
    #data=np.load('magnetization-data-n8-d30.npy')
    n=0
    while engine.iteration < args.Nsteps-1:
        #do numsteps TEBD steps 
        #measure the operators 
        S[n,1:,1],S[n,1:,0],S[n,1:,2]=engine._mps.measureList(sy),engine._mps.measureList(sx),engine._mps.measureList(sz)
        T[n,1:,0],T[n,1:,1],T[n,1:,2]=engine._mps.measureList(taux),engine._mps.measureList(tauy),engine._mps.measureList(tauz)


        S[n,0,0]=engine._t0
        S[n,0,1]=engine._t0
        S[n,0,2]=engine._t0
        T[n,0,0]=engine._t0
        T[n,0,1]=engine._t0
        T[n,0,2]=engine._t0        

        schmidt=engine.mps.SchmidtSpectrum(int(N/2))        
        lams[n,0:len(schmidt)]=schmidt
        if args.compare:        
            Sxxz[n,:,1],Sxxz[n,:,0],Sxxz[n,:,2]=enginexxz._mps.measureList([sigma_y]*N),enginexxz._mps.measureList([sigma_x]*N),enginexxz._mps.measureList([sigma_z]*N)
            Sxxz2[n,:,1],Sxxz2[n,:,0],Sxxz2[n,:,2]=enginexxz2._mps.measureList([sigma_y]*N),enginexxz2._mps.measureList([sigma_x]*N),enginexxz2._mps.measureList([sigma_z]*N)        

        if args.loadevoMPS!=None:
            diff=np.abs(tevo-engine._t0)
            n2=np.argmin(diff)
            if diff[n2]<1E-6:
                plt.figure(1)
                plt.clf()
                plt.subplot(3,2,1)
                plt.title(r'$\sigma_x$')
                plt.plot(range(N),np.real(S[n,1:,0]),range(Sevo.shape[1]),Sevo[n2,:,0],'go',range(N),np.real(Sxxz[n,:,0]),'rx')
                plt.ylim([-1,1])
                plt.subplot(3,2,2)
                plt.title(r'$\tau_x$')        
                plt.plot(range(N),np.real(T[n,1:,0]),range(Tevo.shape[1]),Tevo[n2,:,0],'go',range(N),np.real(Sxxz2[n,:,0]),'rx')
                plt.ylim([-1,1])
                plt.subplot(3,2,3)
                plt.title(r'$\sigma_y$')
                plt.plot(range(N),np.real(S[n,1:,1]),range(Sevo.shape[1]),Sevo[n2,:,1],'go',range(N),np.real(Sxxz[n,:,1]),'rx')
                plt.ylim([-1,1])
                plt.subplot(3,2,4)
                plt.title(r'$\tau_y$')                        
                plt.plot(range(N),np.real(T[n,1:,1]),range(Tevo.shape[1]),Tevo[n2,:,1],'go',range(N),np.real(Sxxz2[n,:,1]),'rx')            
                plt.ylim([-1,1])
                plt.subplot(3,2,5)
                plt.title(r'$\sigma_z$')                
                plt.plot(range(N),np.real(S[n,1:,2]),range(Sevo.shape[1]),Sevo[n2,:,2],'go',range(N),np.real(Sxxz[n,:,2]),'rx')
                plt.ylim([-1,1])
                plt.subplot(3,2,6)
                plt.title(r'$\tau_z$')                                        
                plt.plot(range(N),np.real(T[n,1:,2]),range(Tevo.shape[1]),Tevo[n2,:,2],'go',range(N),np.real(Sxxz2[n,:,2]),'rx')
                plt.legend(['M-double','evo-mps','M-single'])            
                plt.ylim([-1,1])
                plt.draw()
                plt.show()
                plt.pause(0.01)
                input()
                
            else:
                plt.figure(1)
                plt.clf()
                plt.subplot(3,2,1)
                plt.title(r'$\sigma_x$')
                plt.plot(range(N),np.real(S[n,1:,0]),range(N),np.real(Sxxz[n,:,0]),'rx')            
                plt.ylim([-1,1])
                plt.subplot(3,2,2)
                plt.title(r'$\tau_x$')        
                plt.plot(range(N),np.real(T[n,1:,0]),range(N),np.real(Sxxz2[n,:,0]),'rx')
                plt.ylim([-1,1])
                plt.subplot(3,2,3)
                plt.title(r'$\sigma_y$')
                plt.plot(range(N),np.real(S[n,1:,1]),range(N),np.real(Sxxz[n,:,1]),'rx')
                plt.ylim([-1,1])
                plt.subplot(3,2,4)
                plt.title(r'$\tau_y$')                        
                plt.plot(range(N),np.real(T[n,1:,1]),range(N),np.real(Sxxz2[n,:,1]),'rx')            
                plt.ylim([-1,1])
                plt.subplot(3,2,5)
                plt.title(r'$\sigma_z$')                
                plt.plot(range(N),np.real(S[n,1:,2]),range(N),np.real(Sxxz[n,:,2]),'rx')
                plt.ylim([-1,1])
                plt.subplot(3,2,6)
                plt.title(r'$\tau_z$')                                        
                plt.plot(range(N),np.real(T[n,1:,2]),range(N),np.real(Sxxz2[n,:,2]),'rx')
                plt.legend(['M-double','M-single'])                            
                plt.ylim([-1,1])
                plt.draw()
                plt.show()
                plt.pause(0.01)

        elif args.plot:
            plt.figure(1)
            plt.clf()
            plt.subplot(3,2,1)
            plt.title(r'$\sigma_x$')
            plt.plot(range(N),np.real(S[n,1:,0]))
            plt.ylim([-1,1])
            plt.subplot(3,2,2)
            plt.title(r'$\tau_x$')        
            plt.plot(range(N),np.real(T[n,1:,0]))
            plt.ylim([-1,1])
            plt.subplot(3,2,3)
            plt.title(r'$\sigma_y$')
            plt.plot(range(N),np.real(S[n,1:,1]))
            plt.ylim([-1,1])
            plt.subplot(3,2,4)
            plt.title(r'$\tau_y$')                        
            plt.plot(range(N),np.real(T[n,1:,1]))
            plt.ylim([-1,1])
            plt.subplot(3,2,5)
            plt.title(r'$\sigma_z$')                
            plt.plot(range(N),np.real(S[n,1:,2]))
            plt.ylim([-1,1])
            plt.subplot(3,2,6)
            plt.title(r'$\tau_z$')                                        
            plt.plot(range(N),np.real(T[n,1:,2]))
            plt.ylim([-1,1])

            plt.figure(2)
            plt.clf()
            plt.semilogy(engine.mps.SchmidtSpectrum(int(N/2)),'rx')
            plt.draw()
            plt.show()
            plt.pause(0.01)
 
        #else:
        #    plt.figure(1,figsize=(10,8))
        #    plt.clf()
        #    plt.subplot(3,2,1)
        #    plt.title(r'$\sigma_x$')
        #    plt.plot(range(N),np.real(S[n,:,0]),range(N),np.real(Sxxz[n,:,0]),'x')
        #    plt.legend(['Jess','XXZ'])
        #    plt.ylim([-1,1])
        #    plt.subplot(3,2,2)
        #    plt.title(r'$\tau_x$')        
        #    plt.plot(range(N),np.real(T[n,:,0]))
        #    plt.legend(['Jess','XXZ'])        
        #    plt.ylim([-1,1])
        #    plt.subplot(3,2,3)
        #    plt.title(r'$\sigma_y$')        
        #    plt.plot(range(N),np.real(S[n,:,1]),range(N),np.real(S[n,:,1]),'x')
        #    plt.legend(['Jess','XXZ'])        
        #    plt.ylim([-1,1])
        #    plt.subplot(3,2,4)
        #    plt.title(r'$\tau_y$')                
        #    plt.plot(range(N),np.real(T[n,:,1]))
        #    plt.legend(['Jess','XXZ'])        
        #    plt.ylim([-1,1])
        #    plt.subplot(3,2,5)
        #    plt.title(r'$\sigma_z$')        
        #    plt.plot(range(N),np.real(S[n,:,2]),range(N),np.real(S[n,:,2]),'x')
        #    plt.legend(['Jess','XXZ'])        
        #    plt.ylim([-1,1])
        #    plt.subplot(3,2,6)
        #    plt.title(r'$\tau_z$')                        
        #    plt.plot(range(N),np.real(T[n,:,2]))
        #    plt.legend(['Jess','XXZ'])        
        #    plt.ylim([-1,1])
        #    
        #    plt.draw()
        #    plt.show()
        #    plt.pause(0.01)
        #if it<args.switchingstep:
        np.save('S'+filename,S)
        np.save('T'+filename,T)
        np.save('lam'+filename,lams)                
        tw,t=engine.doTEBD(dt=dt,numsteps=args.msteps,Dmax=Dmax,tr_thresh=thresh,cp=args.cp,keep_cp=args.keep_cp)
        truncWeight[n]=tw
        np.save('truncWeight'+filename,truncWeight)                        
        #else:
        #    if it==args.switchingstep:
        #        engine.initializeTDVP()
        #    it=engine.doTDVP(dt=1.0*args.msteps/args.mstepsTDVP*dt,numsteps=args.mstepsTDVP,cnterset=it,solver=args.solver)
        #if args.debug:        
        #    #print()
        if args.compare:        
            txxz,twxxz=enginexxz.doTEBD(dt=dt,numsteps=args.msteps,Dmax=Dmax,tr_thresh=thresh,verbose=0)
            txxz2,twxxz2=enginexxz2.doTEBD(dt=dt,numsteps=args.msteps,Dmax=Dmax,tr_thresh=thresh,verbose=0)        
        #    #print()        

        n+=1
    S[-1,0,0]=engine._t0
    S[-1,0,1]=engine._t0
    S[-1,0,2]=engine._t0
    T[-1,0,0]=engine._t0
    T[-1,0,1]=engine._t0
    T[-1,0,2]=engine._t0        
    S[-1,1:,1],S[-1,1:,0],S[-1,1:,2]=engine._mps.measureList(sy),engine._mps.measureList(sx),engine._mps.measureList(sz)
    T[-1,1:,0],T[-1,1:,1],T[-1,1:,2]=engine._mps.measureList(taux),engine._mps.measureList(tauy),engine._mps.measureList(tauz)
