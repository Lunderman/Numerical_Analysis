import numpy as np

# Conjugate transpose function 
def getH(A):
    if np.any(np.iscomplex(A)):
        return(np.conjugate(A).T)
    else:
        return(A.T)

########################################################################
########################################################################
# HW4
# Define the QR function for the REAL matrix A with full rank
def mgs(A):
    # set dimension of $A$
    m,n = A.shape
    
    # Initialize V,Q,R
    V = np.zeros((m,n))
    Q = np.zeros((m,n))
    R = np.zeros((n,n))

    for kk in range(n): 
        V[:,kk]=A[:,kk]
    for ii in range(n):
        R[ii,ii]= np.linalg.norm(V[:,ii],2)            
        Q[:,ii]=V[:,ii]/R[ii,ii]
        for jj in range(ii+1,n):
            R[ii,jj] = getH(Q[:,ii])@V[:,jj]
            V[:,jj] =V[:,jj] -R[ii,jj]*Q[:,ii]
    return(Q,R)

########################################################################
########################################################################
# HW5
    
def house(A):
    m,n = A.shape
    B = np.zeros((m,n))+A
    W = np.zeros((m,n))
    for kk in range(n):
        x = B[kk:m,kk].reshape((m-kk,1))
        e1 = np.zeros((m-kk,1))
        e1[0]=1
        v_k = np.zeros((m-kk,1))
        v_k = np.sign(x[0])*np.linalg.norm(x)*e1+x
        v_k = v_k/np.linalg.norm(v_k)
        W[kk:m,kk] = v_k.reshape((len(v_k)))
        B[kk:m,kk:n] -= 2*v_k@(getH(v_k)@B[kk:m,kk:n])
    return(W,B)

def formQ(W):
    m,n = W.shape
    Q = np.identity(m)
    for jj in range(m):
        for kk in range(n)[::-1]:
            v_k = W[kk:m,kk].reshape(((m-kk,1)))
            Q[kk:m,jj] -= 2*v_k@(getH(v_k)@Q[kk:m,jj])
    return(Q)