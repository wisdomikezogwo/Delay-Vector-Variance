import numpy as np


# Delay Vector Variance method for real and complex signals


# USAGE: C = dvv (X, m, Nsub, nd, Ntv)
#	X       original real-valued or complex time series
#	m       delay embedding dimension
#	Ntv     number of points on horizontal axes
#	Nsub	number of reference DVs to consider
#	nd      Span over which to perform DVV

def dvv(X=None, m=None, Nsub=None, nd=None, Ntv=None, *args, **kwargs):
    # Default parameters
    if X.all() == None:
        print('Not enough Input arguments')
    if m == None:
        m = 4
    if Nsub == None:
        Nsub = 200
    if nd == None:
        nd = 2
    if Ntv == None:
        Ntv = 25 * nd

    N = X.shape[0]
    tau = 1
    d = np.zeros(((N - (m * tau)), Nsub))
    y = np.zeros((Ntv, 1))
    count = 0
    acc = 0

    # Makes input vector X a column vector
    # Redundant since I'm only gonna be passing column vectors
    # if (X.shape[0] > 1):
    #    X=X.T

    # Generate Nsub subset from existing DV's, randomly
    temp = np.random.permutation(N - m * tau)
    # print(type(Nsub), Nsub)
    ref = temp[np.arange(Nsub)] + (m * tau)

    # Computes the pair wise distances b/w reference DV's and all DV's  for i in range(1,6): X(j-m*tau:tau:j-tau));
    for i in range(Nsub):
        for j in range(((m * tau) + 1), N):
            d[(j - (m * tau)), i] = np.linalg.norm(
                X[np.arange((ref[i] - (m * tau)), (ref[i] - tau), tau)] - X[np.arange((j - (m * tau)), (j - tau), tau)])
            # Added +1 cause of arange
            if (ref[i] != j):
                acc = acc + d[j - (m * tau), i]
                count = count + 1

    # Mean and std variation calculation of input data
    avg = acc / count
    count = 0
    acc = 0

    for i in range(Nsub):
        for j in range((m * tau) + 1, N):
            if (ref[i] != j):
                acc = acc + (d[j - (m * tau), i] - avg) ** 2
                count = count + 1

    variance = np.sqrt(acc / (count - 1))
    # Calculates the range vector consisting of Ntv equally spaced regions
    n = (np.arange(Ntv)) - 1
    rd = avg - (nd * variance) + (((2 * nd) * variance) * n) / (Ntv - 1)

    # Creates sets of DV's, for each ref element of subset and value rd, which have np.linalg.norms closer than distance rd to ref
    for n in range(rd.shape[0]):
        if (rd[n] > 0):
            tot = 0
            count = 0

            for k in range(Nsub):
                IND = (d[:, k - 1] <= rd[n]).compress((d[:, k - 1] <= rd[n]).flat) + (m * tau)
                # (a>5.5).nonzero()   a.compress((a>5.5).flat)
                IND = IND[IND != k]
                # sets have atleast 30 DVs
                if (IND.shape[0] >= 30):
                    tot = tot + np.var(X[IND])
                    count = count + 1

            if (np.logical_not(count)):
                y[n] = np.nan

            else:
                y[n] = tot / (count * np.var(X))

        else:
            y[n] = np.nan

    # Horizontal axis
    T = (rd.T - avg) / variance
    T = T.reshape((50, 1))

    # DVV Output
    data = np.hstack((T, y))

    return data
