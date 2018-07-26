import numpy as np
"""
flips a bit at position pos; b has to be an unsigned integer of 64 bit! (i.e. np.uint64)
"""
def flipBit(b,pos):
    mask=np.uint64(1)<<np.uint64(pos)
    if b&mask==0:#b(pos)==0
        return b|mask
    elif b&mask!=0:#b(pos)==1
        return b&(~mask)

"""
sets a bit at position pos; b has to be an unsigned integer of 64 bit! (i.e. np.uint64)
"""
def setBit(b,pos):
    mask=np.uint64(1)<<np.uint64(pos)
    return mask|b

def unsetBit(b,pos):
    mask=np.uint64(1)<<np.uint64(pos)
    return (~mask)&b

def getBit(b, pos):
    mask=np.uint64(1)<<np.uint64(pos)
    return int((b & mask)>0)


"""
returns all basis states of a system of N sites with Nup up-spins and N-Nup down spins
for a system of spin 1/2 states (or, equivalently, a system of spinless fermions)
"""
def binarybasis(N,Nup):
    if Nup==0:
        return [np.uint64(0)]
    else:
        basis = []
        for p in range(Nup-1,N):
            rest=binarybasis(p,Nup-1)
            for n in rest:
                basis.append(np.uint64(n+2**p))
        return basis


    
