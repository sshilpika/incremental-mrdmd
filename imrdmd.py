import numpy as np
from numpy import dot, multiply, diag, power
from numpy import pi, exp, sin, cos
from numpy.linalg import inv, eig, pinv, solve
from scipy.linalg import svd, svdvals
from math import floor, ceil
import pandas as pd
import time

class DynamicModeDecomposition():
    '''
    Code modified from https://humaticlabs.com/blog/mrdmd-python/
    '''
    data = None
    df_sm_data = None
    augmented_data = None
    nodes = []
    
    def __init__(self, data, max_level):
        self.data = data
        self.levels = max_level
        self.df_sm_data = None
        self.augmented_data = None

    def omega_approx(self, beta):
        """Return an approximate omega value for given beta. Equation (5) from Gavish 2014."""
        return 0.56 * beta**3 - 0.95 * beta**2 + 1.82 * beta + 1.43


    def svht(self, X, sv=None, sigma=None):
        """Return the optimal singular value hard threshold (SVHT) value.
        `X` is any m-by-n matrix. `sigma` is the standard deviation of the 
        noise, if known. Optionally supply the vector of singular values `sv`
        for the matrix (only necessary when `sigma` is unknown). If `sigma`
        is unknown and `sv` is not supplied, then the method automatically
        computes the singular values."""
        # svht for sigma unknown
        try:
            m,n = sorted(X.shape) #  m <= n
        except:
            raise ValueError('invalid input matrix')
        beta = m / n # ratio between 0 and 1
        if sigma is None:
            if sv is None:
                sv = svdvals(X)
            sv = np.squeeze(sv)
            if sv.ndim != 1:
                raise ValueError('vector of singular values must be 1-dimensional')
            omega_approx = self.omega_approx(beta)
            return np.median(sv) * omega_approx
        else:
            return lambda_star(beta) * np.sqrt(n) * sigma

    
    def addblock_svd_update(self, U, S, V, B, ori=[]): 
        """incremental svd decomposition -> "An Incremental Singular Value Decomposition Approach for 
        Large-Scale Spatially Parallel & Distributed but Temporally Serial 
        Data -- Applied to Technical Flows"""
        k = U.shape[1]                                             
        b = B.shape[1]
        t = k + b
        # reduce size
        M = dot(U.conj().T, B)
        P = B - dot(U,M)
        # othogonalize P
        [QP,RP] = self.qr_modified(P)
        # othogonalize Q
        Q = np.hstack([U, QP])
    
        [QQ,RQ] = self.qr_modified(Q)
        
        SM_1 = None
        if len(S.shape) == 0:
            SM_1 = np.hstack((np.array([[S]*M.shape[0]]).T,M)) 
        elif len(S.shape) == 1:
            SM_1 = np.hstack((np.diag(S), M))
        else:
            SM_1 = np.hstack((S,M))
        
        SM_2 = np.hstack((np.zeros( M.shape ).conj().T, RP))        
        K = dot(RQ,np.vstack((SM_1, SM_2))) 
    
        tUp1,tSp1,tVp1 = svd(K, False) # SVD of input matrix
    
        rr = k + b
        tUp = tUp1[:,:rr]
        tSp = diag(tSp1)[:rr,:rr]
        tVp = tVp1.conj().T[:,:rr]
        # update S
        Sp = tSp[:t, :t]
        # update V
    
        size_v1 = len(V) if type(V) != int else 1
        size_v2 = V.shape[1] if type(V) != int else 1
        z1 = np.zeros((size_v1, b))
        z2 = np.zeros((b, size_v2))
        z3 = np.eye(b) #z3 = eye(b);
        z__ = np.hstack(([[V]], z1)) if type(V) == int else np.hstack((V, z1))
        temp = np.vstack((z__, np.hstack((z2, z3))))
        Vp = dot(temp, tVp[:t, :t]) 
        # % update U
        Up = dot(QQ,tUp[:k+b, :t])
        # end
        return (Up,Sp,Vp)


    def qr_modified(self, A):
        # general sizes
        m,n = A.shape
        # initialize everything
        Q = np.zeros((m,n))
        R = np.zeros((n,n))
        # actual algorithm
        for j in range(n):
            v = np.expand_dims(A[:, j], axis=1)
            for i in range(j):
                R[i,j] = dot(np.expand_dims(Q[:,i], axis=1).conj().T, np.expand_dims(A[:,j], axis=1))
                v = v - dot(R[i,j], np.expand_dims(Q[:, i], axis=1))
            R[j,j] = np.linalg.norm(v)
            Q[:, j] = np.squeeze(v)/R[j,j]
        return Q, R

    def dmd(self, X, Y, truncate = 0, U_O=None, S_O=None, Vh_O=None):
        if truncate == None:
            # return empty vectors
            mu = np.array([], dtype='complex')
            Phi = np.zeros([X.shape[0], 0], dtype='complex')
        else:
            
            if U_O is not None and S_O is not None and Vh_O is not None:
                U2,Sig2,Vh2 = self.addblock_svd_update(U_O, np.diag(S_O), Vh_O.conj().T, X ) 
                
                r = len(Sig2) if truncate == 0 else truncate # rank truncation
                U = U2[:,:r]
                Sig = Sig2[:r,:r]
                V = Vh2.conj().T[:,:r]
            
            else:
                U2,Sig2,Vh2 = svd(X, False) # SVD of input matrix

                r = len(Sig2) if truncate == 0 else truncate # rank truncation
                U = U2[:,:r]
                Sig = diag(Sig2)[:r,:r]
                V = Vh2.conj().T[:,:r]
             
            Atil = dot(dot(dot(U.conj().T, Y), V), inv(Sig)) # build A tilde
            mu,W = eig(Atil)
            Phi = dot(dot(dot(Y, V), inv(Sig)), W) # build DMD modes
        return mu, Phi, U2, Sig2, Vh2


    def mrdmd(self, D, level=0, bin_num=0, offset=0, max_levels=7, max_cycles=2, do_svht=True, split_points=[], inc_flag = False):
        """Compute the multi-resolution DMD on the dataset `D`, returning a list of nodes
        in the hierarchy. Each node represents a particular "time bin" (window in time) at
        a particular "level" of the recursion (time scale). The node is an object consisting
        of the various data structures generated by the DMD at its corresponding level and
        time bin. The `level`, `bin_num`, and `offset` parameters are for record keeping 
        during the recursion and should not be modified unless you know what you are doing.
        The `max_levels` parameter controls the maximum number of levels. The `max_cycles`
        parameter controls the maximum number of mode oscillations in any given time scale 
        that qualify as "slow". The `do_svht` parameter indicates whether or not to perform
        optimal singular value hard thresholding."""
    
        # 4 times nyquist limit to capture cycles
        nyq = 8 * max_cycles
    
        # time bin size
        bin_size = D.shape[1]
        if bin_size < nyq:
            return []
    
        # extract subsamples 
        step = floor(bin_size / nyq) # max step size to capture cycles
        _D = D[:,::step]
        X = _D[:,:-1]
        Y = _D[:,1:]
    
        # determine rank-reduction
        if do_svht:
            _sv = svdvals(_D)
            tau = self.svht(_D, sv=_sv)
            r = sum(_sv > tau)
        else:
            r = min(X.shape)

        # compute dmd
        mu,Phi, U, S, Vh = self.dmd(X, Y, r)
    
        # frequency cutoff (oscillations per timestep)
        rho = max_cycles / bin_size
    
        # consolidate slow eigenvalues (as boolean mask)
        slow = (np.abs(np.log(mu) / (2 * pi * step))) <= rho
        n = sum(slow) # number of slow modes
    
        # extract slow modes (perhaps empty)
        mu = mu[slow]
        Phi = Phi[:,slow]
    
        if n > 0:
    
            # vars for the objective function for D (before subsampling)
            Vand = np.vander(power(mu, 1/step), bin_size, True)
            P = multiply(dot(Phi.conj().T, Phi), np.conj(dot(Vand, Vand.conj().T)))
            q = np.conj(diag(dot(dot(Vand, D.conj().T), Phi)))
    
            # find optimal b solution
            b_opt = solve(P, q).squeeze()
    
            # time evolution
            Psi = (Vand.T * b_opt).T
    
        else:
    
            # zero time evolution
            b_opt = np.array([], dtype='complex')
            Psi = np.zeros([0, bin_size], dtype='complex')
    
        # dmd reconstruction
        D_dmd = dot(Phi, Psi)   
    
        # remove influence of slow modes
        D_curr = D.copy()
        D = D - D_dmd
    
        # record keeping
        node = type('Node', (object,), {})()
        node.level = level            # level of recursion
        node.bin_num = bin_num        # time bin number
        node.bin_size = bin_size      # time bin size
        node.start = offset           # starting index
        node.stop = offset + bin_size # stopping index
        node.step = step              # step size
        node.rho = rho                # frequency cutoff
        node.r = r                    # rank-reduction
        node.n = n                    # number of extracted modes
        node.mu = mu                  # extracted eigenvalues
        node.Phi = Phi                # extracted DMD modes
        node.Psi = Psi                # extracted time evolution
        node.b_opt = b_opt            # extracted optimal b vector
        node.split_points = split_points[level]
        node.U = U
        node.S = S
        node.Vh = Vh
        node.Y = Y
        node.D = D_curr
        if level == 0:
            node.D_SLOW = D_dmd
        self.nodes = [node]
    
        # split data along timeline and do recursion
        if level < max_levels and len(split_points[level]) !=0:
            split = split_points[level][0]
            first_split = [] if level+1 >= max_levels else [ii for ii in split_points[level+1] if ii < split]
            second_split = [] if level+1 >= max_levels else [ii-split for ii in split_points[level+1] if ii > split]
    
            first_split = []
            second_split = []

            if level+1 <= max_levels:
                first_split = split_points.copy()
                for ind, split_next in enumerate(first_split[level+1:]):
                   first_split[level+1+ind] = [iii for iii in split_next if iii <= split]
    
                second_split = split_points.copy()
                for ind, split_next in enumerate(second_split[level+1:]):
                   second_split[level+1+ind] = [iii-split for iii in split_next if iii > split]

            binN = max([n.bin_num for n in self.nodes if n.level == level]) 
            binNum = (binN + 1) if inc_flag else 2*bin_num
            
            self.nodes += self.mrdmd(
                D[:,:split],
                level=level+1,
                bin_num=binNum,
                offset=offset,
                max_levels=max_levels,
                max_cycles=max_cycles,
                do_svht=do_svht, 
                split_points=first_split,
                )
            self.nodes += self.mrdmd(
                D[:,split:],
                level=level+1,
                bin_num=binNum+1,
                offset=offset+split,
                max_levels=max_levels,
                max_cycles=max_cycles,
                do_svht=do_svht, 
                split_points=second_split,
                )
    
        return self.nodes

    def stitch(self, nodes, level):
    
        # get length of time dimension
        start = min([nd.start for nd in nodes])
        stop = max([nd.stop for nd in nodes])
        t = stop - start
    
        # extract relevant nodes
        nodes = [n for n in nodes if n.level == level]
        nodes = sorted(nodes, key=lambda n: n.bin_num)
        
        # stack DMD modes
        Phi = np.hstack([n.Phi for n in nodes])
        
        # allocate zero matrix for time evolution
        nmodes = sum([n.n for n in nodes])
        Psi = np.zeros([nmodes, t], dtype='complex')
        
        # copy over time evolution for each time bin
        i = 0
        for n in nodes:
            _nmodes = n.Psi.shape[0]
            Psi[i:i+_nmodes,n.start:n.stop] = n.Psi
            i += _nmodes
        
        return Phi,Psi

    def update_nodes(self, D):

        for level in self.levels:
            nodes = [n for n in self.nodes if n.level == level]
            nodes = sorted(nodes, key=lambda n: n.bin_num)
            

            
    # incremental mrDMD
    def inc_mrdmd(self, D, level=0, bin_num=0, offset=0, max_levels=7, max_cycles=2, do_svht=True, split_points=[], recon_threshold = 5000):

        level_nodes = [n for n in self.nodes if n.level == level and n.bin_num==bin_num] 
        
        assert(len(level_nodes) == 1)
        
        # 4 times nyquist limit to capture cycles
        nyq = 8 * max_cycles
    
        # time bin size
        bin_size = D.shape[1] + level_nodes[0].bin_size 
        if bin_size < nyq:
            return []
        
        # extract subsamples 
        step = floor(bin_size / nyq)
        _D = D[:,::step]
        X = _D[:,:-1]
        Y = _D[:,1:]
    
        # determine rank-reduction
        if do_svht:
            _sv = svdvals(_D)
            tau = self.svht(_D, sv=_sv)
            r = sum(_sv > tau)
            r = Y.shape[1]
        else:
            r = min(X.shape)
            r = Y.shape[1] 
    
        # compute dmd
        new_Y = np.hstack((level_nodes[0].Y, Y))
        new_D = np.hstack((level_nodes[0].D, D))
        mu,Phi, U, S, Vh = self.dmd(X, new_Y, r, level_nodes[0].U, level_nodes[0].S, level_nodes[0].Vh ) 
        
        # frequency cutoff (oscillations per timestep)
        rho = max_cycles / bin_size 
    
        # consolidate slow eigenvalues (as boolean mask)
        slow = (np.abs(np.log(mu) / (2 * pi * step))) <= rho
        n = sum(slow) # number of slow modes
    
        # extract slow modes (perhaps empty)
        mu = mu[slow]
        Phi = Phi[:,slow]
    
        if n > 0:
    
            DD_rec = np.hstack((level_nodes[0].D, D))
            
            Vand = np.vander(power(mu, 1/step), bin_size, True)
            P = multiply(dot(Phi.conj().T, Phi), np.conj(dot(Vand, Vand.conj().T)))
            q = np.conj(diag(dot(dot(Vand, DD_rec.conj().T), Phi)))
    
            # find optimal b solution
            b_opt = solve(P, q).squeeze()

            # time evolution
            Psi = (Vand.T * b_opt).T
    
        else:
    
            # zero time evolution
            b_opt = np.array([], dtype='complex')
            Psi = np.zeros([0, bin_size], dtype='complex')
    
        # dmd reconstruction
        D_dmd = dot(Phi, Psi)  
    
        # remove influence of slow modes
        D = D - D_dmd[:,-D.shape[1]:]

        offset_next = level_nodes[0].D.shape[1]
        recon_err = np.linalg.norm(level_nodes[0].D_SLOW - D_dmd[:,:-D.shape[1]])

# if reconstruction error is over threshold either recompute older results (in steps) or ignore slower modes (for precomputed results) at level 0
        if recon_err > recon_threshold:
            Psi, Phi = np.zeros_like(Psi),np.zeros_like(Phi)
            
        for nn in self.nodes: 
            nn.level += 1   
    
        # record keeping
        node = type('Node', (object,), {})()
        node.level = level            # level of recursion
        node.bin_num = bin_num        # time bin number
        node.bin_size = bin_size      # time bin size
        node.start = offset           # starting index
        node.stop = offset + bin_size # stopping index
        node.step = step              # step size
        node.rho = rho                # frequency cutoff
        node.r = r                    # rank-reduction
        node.n = n                    # number of extracted modes
        node.mu = mu                  # extracted eigenvalues
        node.Phi = Phi                # extracted DMD modes
        node.Psi = Psi                # extracted time evolution
        node.b_opt = b_opt            # extracted optimal b vector
        node.split_points = split_points[level]
        node.U = U
        node.S = S
        node.Vh = Vh
        node.Y = new_Y
        node.D = new_D
        if level == 0:
            node.D_SLOW = D_dmd
        self.nodes = [node] if level != 0 else self.nodes+[node] 
    
        # split data into two and do recursion
        if level < max_levels and len(split_points[level]) !=0:
            split = split_points[level][0]
            first_split = [] if level+1 >= max_levels else [ii for ii in split_points[level+1] if ii < split]
            second_split = [] if level+1 >= max_levels else [ii-split for ii in split_points[level+1] if ii > split]
    
            first_split = []
            second_split = []
            if level+1 <= max_levels:
                first_split = split_points.copy()
                for ind, split_next in enumerate(first_split[level+1:]):
                   first_split[level+1+ind] = [iii for iii in split_next if iii <= split]
    
                second_split = split_points.copy()
                for ind, split_next in enumerate(second_split[level+1:]):
                   second_split[level+1+ind] = [iii-split for iii in split_next if iii > split]

            if level == 0:

                split_points.insert(0,[])
                self.nodes += self.mrdmd(
                    D,
                    level=level+1,
                    bin_num=1,
                    offset=offset_next,#D.shape[1],
                    max_levels=max_levels,
                    max_cycles=max_cycles,
                    do_svht=do_svht, 
                    split_points=split_points,
                    inc_flag = True
                    )
                   
            else:
                binN = max([n.bin_num for n in self.nodes if n.level == level]) 
                
                self.nodes += self.mrdmd(
                    D[:,:split],
                    level=level+1,
                    bin_num = binN+1,
                    offset=offset,
                    max_levels=max_levels,
                    max_cycles=max_cycles,
                    do_svht=do_svht, 
                    split_points=first_split,
                    )
                self.nodes += self.mrdmd(
                    D[:,split:],
                    level=level+1,
                    bin_num=binN+2,
                    offset=offset+split,
                    max_levels=max_levels,
                    max_cycles=max_cycles,
                    do_svht=do_svht, 
                    split_points=second_split,
                    )
    
        return self.nodes
