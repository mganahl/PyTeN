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
plt.ion()

def createevoData(filename):
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

def loadevoMPS(filename,n):
    a=sp.load(filename)
    times=a[:,0]
    mpss=[]
    mps=mpslib.MPS.fromList(np.transpose(np.asarray(a[n][1][site][:,:,:]),(1,2,0)) for site in range(1,len(a[n][1])))
    return mps,times[1]
if __name__ == "__main__":

    parser = argparse.ArgumentParser('Branching.py: time evolution for a systems of coupled Heisenberg chains. Default values of all arguments are given in round brakets')
    parser.add_argument('--N', help='system size (10)',type=int,default=8)
    parser.add_argument('--D', help='maximally allowed MPS bond dimension. (32)',type=int,default=32)
    parser.add_argument('--switch',action='store_true', help='if given, simulation will switch from TEBD to TDVP once the maximal bond dimension D is reached (False)')
    parser.add_argument('--solver', help='solver type to be used in TDVP. Can be any of {RK45, RK23, LAN, SEXPMV} (LAN)',type=str,default='LAN')
    parser.add_argument('--ncv', help='number of krylov vectors to be used in the LAN solver. Larger values increase accuracy and runtimes (10)',type=int,default=10)        
    parser.add_argument('--atol', help='absolute error tolerance when using RK45 or RK23 solver. Smaller values increase accuracy and runtimes (1E-12)',type=float,default=1E-12)
    parser.add_argument('--rtol', help='relative error tolerance when using RK45 or RK23 solver. Smaller values increase accuracy and runtimes (1E-6)',type=float,default=1E-6)    
    parser.add_argument('--dt_TDVP', help='time step for TDVP time evolution; note that the internally used time-step is dt_TDVP/abs(N*stiffness) (dt)',type=float)    
    parser.add_argument('--mstepsTDVP', help='number of TDVP-time-evolution steps between measurement (msteps)',type=int)
    
    parser.add_argument('--cp', help='check pointing steps (None)',type=int)    
    parser.add_argument('--stiffness', help='stiffness (Hamiltonian parameter) (3.0)',type=float,default=3.0)
    parser.add_argument('--chi', help='chi (Hamiltonian parameter) (2.0)',type=float,default=2.0)
    parser.add_argument('--w', help='omega (Hamiltonian parameter) (1.0)',type=float,default=1.0)
    parser.add_argument('--plot',action='store_true', help='plot intermediate results (False)')    
    parser.add_argument('--keep_cp',action='store_true',help='if this flag is set, keep the checkpoint files; otherwise, only the last checkpoint will be kept (False)')    
    parser.add_argument('--theta1', help='theta1 parameter of initial state (0.3761924090303321)',type=float,default=0.37619240903033213)
    parser.add_argument('--theta2', help='theta2 parameter of initial state (2.2815584984041757)',type=float,default=2.2815584984041757)
    parser.add_argument('--phi1', help='phi1 parameter of initial state (3.676699204438477)',type=float,default=3.676699204438477)
    parser.add_argument('--phi2', help='phi2 parameter of initial state (5.569523747854273)',type=float,default=5.569523747854273)        
    parser.add_argument('--dt', help='time step for TEBD time evolution; note that the internally used time-step is dt/abs(N*stiffness) (0.01)',type=float,default=0.01)
    parser.add_argument('--truncthresh', help='truncation threshold (1E-8) for TEBD; Schmidt-values below truncthresh will be truncated',type=float,default=1E-8)    
    #parser.add_argument('--Nsteps', help='number of total timesteps to be taken; total time T=Nsteps*dt/abs(stiffness*N)(1000)',type=int,default=1000)
    parser.add_argument('--T', help='total simulation time T',type=float,default=6.0)    
    parser.add_argument('--msteps', help='number of TEBD time-evolution steps between measurements. The time-difference between measurements will be msteps*dt/abs(stiffness*N) (20)',type=int,default=20)

    parser.add_argument('--filename', help='filename-ending for output. A folder with the provided name will be generated, and results will be stored in this folder, with the provided string appended to all files generated by the simulation (_Branching)',type=str,default='_Branching')
    parser.add_argument('--loadevoMPS', help='file name of an evoMPS matrix-trajectory file. If given, a state of the mps-trajectory file will be loaded and set as initial state (see --evostate option). Note that the initial time of the simulation will be set to the time of the loaded state (None)',type=str)
    parser.add_argument('--evostate', help='the time-index of the state to be loaded from the matrix-trajectory file given in loadevoMPS (None)',type=int)    

    args=parser.parse_args()

    if args.switch:
        if (args.dt_TDVP==None):
            args.dt_TDVP=args.dt
        if (args.mstepsTDVP==None):
            args.mstepsTDVP=args.msteps
        
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

    state=[np.kron(blochstate(args.theta1,args.phi1),blochstate(args.theta2,args.phi2))]*args.N    
    mps=mpslib.MPS.productState(state,obc=True,dtype=complex) #this mps for now has a maximally allowed bond dimension mps._D=1;
    t0=0
    if args.loadevoMPS:
        if args.evostate==None:
            raise ValueError("evostate has not been specified; please provide an integer number to specify which state should be loaded from the matrix-trajectory file")
        mps,t0=loadevoMPS(root+'/'+args.loadevoMPS,args.evostate)
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
    if args.switch:
        dt_TDVP=-1j*args.dt_TDVP/np.abs(J)
    Dmax=args.D   #maximum bond dimension to be used during simulation; the maximally allowed bond dimension of the mps will be adapted to this value in the TEBDEngine
    thresh=args.truncthresh  #truncation threshold

    S=np.zeros((1,N+1,3)).astype(complex) #container for holding the measurements
    T=np.zeros((1,N+1,3)).astype(complex) #container for holding the measurements    

    sigma_x=np.asarray([[0,1],[1,0]]).astype(complex)
    sigma_y=np.asarray([[0,-1j],[1j,0]]).astype(complex)
    sigma_z=np.diag([1,-1]).astype(complex)

    sx=[np.kron(sigma_x,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    taux=[np.kron(np.eye(2),sigma_x) for n in range(N)]  #a list of local operators to be measured
    sy=[np.kron(sigma_y,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    tauy=[np.kron(np.eye(2),sigma_y) for n in range(N)]  #a list of local operators to be measured    
    sz=[np.kron(sigma_z,np.eye(2)) for n in range(N)]  #a list of local operators to be measured
    tauz=[np.kron(np.eye(2),sigma_z) for n in range(N)]  #a list of local operators to be measured    

    n=0
    engine._t0=t0
    tw=0.0
    firsttdvpstep=True


    #initial state measurements:
    S=np.expand_dims(np.asarray([engine._mps.measureList(sx),engine._mps.measureList(sy),engine._mps.measureList(sz)]).T,0)
    T=np.expand_dims(np.asarray([engine._mps.measureList(taux),engine._mps.measureList(tauy),engine._mps.measureList(tauz)]).T,0)
    schmidt=engine.mps.SchmidtSpectrum(int(N/2))
    schmidt=np.append(schmidt,np.zeros(args.D-len(schmidt)))
    lams=np.expand_dims(np.append([engine._t0],schmidt).T,0)
    truncWeight=np.expand_dims(np.asarray([engine._t0,tw]),0)
    
    while engine._t0 < args.T:
        if args.plot:
            plt.figure(1)
            plt.clf()
            plt.subplot(3,2,1)
            plt.title(r'$\sigma_x$')
            plt.plot(range(N),np.real(S[-1,:,0]))
            plt.ylim([-1,1])
            plt.subplot(3,2,2)
            plt.title(r'$\tau_x$')        
            plt.plot(range(N),np.real(T[-1,:,0]))
            plt.ylim([-1,1])
            plt.subplot(3,2,3)
            plt.title(r'$\sigma_y$')
            plt.plot(range(N),np.real(S[-1,:,1]))
            plt.ylim([-1,1])
            plt.subplot(3,2,4)
            plt.title(r'$\tau_y$')                        
            plt.plot(range(N),np.real(T[-1,:,1]))
            plt.ylim([-1,1])
            plt.subplot(3,2,5)
            plt.title(r'$\sigma_z$')                
            plt.plot(range(N),np.real(S[-1,:,2]))
            plt.ylim([-1,1])
            plt.subplot(3,2,6)
            plt.title(r'$\tau_z$')                                       
            plt.plot(range(N),np.real(T[-1,:,2]))
            plt.ylim([-1,1])

            plt.figure(2)
            plt.clf()
            plt.semilogy(lams[-1,1::],'rx')
            plt.draw()
            plt.show()
            plt.pause(0.01)

        if (not args.switch):
            tw,t=engine.doTEBD(dt=dt,numsteps=args.msteps,Dmax=Dmax,tr_thresh=thresh,cp=args.cp,keep_cp=args.keep_cp)
        elif args.switch  and (max(engine.mps.D)<args.D):
            tw,t=engine.doTEBD(dt=dt,numsteps=args.msteps,Dmax=Dmax,tr_thresh=thresh,cp=args.cp,keep_cp=args.keep_cp)
        else:
            if firsttdvpstep:
                engine.initializeTDVP()
                firsttdvpstep=False
            t=engine.doTDVP(dt=dt_TDVP,numsteps=args.mstepsTDVP,solver=args.solver.upper(),krylov_dim=args.ncv,cp=args.cp,keep_cp=args.keep_cp,rtol=args.rtol,atol=args.atol)

            
        S=np.append(S,np.expand_dims(np.asarray([engine._mps.measureList(sx),engine._mps.measureList(sy),engine._mps.measureList(sz)]).T,0),axis=0)
        T=np.append(T,np.expand_dims(np.asarray([engine._mps.measureList(taux),engine._mps.measureList(tauy),engine._mps.measureList(tauz)]).T,0),axis=0)
        schmidt=engine.mps.SchmidtSpectrum(int(N/2))
        schmidt=np.append(schmidt,np.zeros(args.D-len(schmidt)))
        lams=np.append(lams,np.expand_dims(np.append([engine._t0],schmidt).T,0),axis=0)
        truncWeight=np.append(truncWeight,np.expand_dims(np.asarray([engine._t0,tw]),0),axis=0)

        np.save('S'+filename,S)
        np.save('T'+filename,T)
        np.save('lam'+filename,lams)                
        np.save('truncWeight'+filename,truncWeight)

