import scipy.integrate as sci
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
#importing modules 
N = 31  #int(input('N = '))  # dimension of matrix is N*N

@jit(nopython=True)
def G(qs, qi, e):  # collected JTMA function
    p = 1.1  # sigma_p
    c = 12.  # sigma_c
    s = 2.5 * c  # sigma_s
    cs = abs(np.sqrt(1 / (1 - e))) #cs=ci
    return (50./c)* np.exp(-(abs(qs)**2+abs(qi)**2)/(2*c**2))*np.exp(-(abs(qs+qi)**2)/(2*p**2))*np.sinc(((2+e)*0.5*(cs*abs(qs - qi))**2+e*0.5*(cs*abs(qs + qi))**2)/(s**2))#combining and simplifying collected JTMA
    
for e in np.arange(0, 1, 0.1):  # loop starts for different e value
    Pr1 = np.identity(N)  # initialising G matrix
    for i in range(N):  # loop for different a_s value
        for j in range(N):  # loop for different a_i value
            a_s = i - (N - 1) / 2
            a_i = j - (N - 1) / 2
            Pr1[i][j] = G(a_s, a_i, e)
            print(i, j, Pr1[i][j])

    plt.contourf(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), Pr1)
    plt.colorbar()
    plt.title('JTMA $\epsilon$ = %.2f' % e)
    plt.show()  # visualising G matrix which is collected JTMA

    Pr = np.identity(N)  # initialisng for 2Dpi measurement matrix.
    for i in range(N):
        for j in range(i,N):
            a_i = i - (N - 1) / 2
            a_s = j - (N - 1) / 2
            I1=sci.dblquad(G,-np.inf,a_s,-np.inf,a_i,args=[e])[0]#G--
            I2=sci.dblquad(G,a_s,np.inf,a_i, np.inf, args=[e])[0]#G++
            I3=sci.dblquad(G,-np.inf,a_s,a_i,np.inf, args=[e])[0]#G-+
            I4=sci.dblquad(G,a_s,np.inf,-np.inf,a_i, args=[e])[0]#G+-
            if i == j:
                Pr[i][j] = 0.5 * abs(I1 + I2 - I3 - I4) ** 2
            else:
                Pr[i][j] = abs(I1 + I2 - I3 - I4) ** 2
            print(i, j, Pr[i][j])

    Pr = Pr + Pr.transpose()  # for symmetric Pr
    plt.contourf(np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), np.arange(-(N - 1) / 2, 1 + (N - 1) / 2), Pr)
    plt.colorbar()
    plt.xlabel('$a_s$')
    plt.ylabel('$a_i$')
    plt.title('Pr($a_s$, $a_i$) $\epsilon$ = %.2f' % e)
    plt.show()
