# 2Dpi_data_fitting

This code finds the best fit for the function Pr(a,a) and Pr(a,-a). The input file "2Dpi.mat" consists of the data in the collumn named 'JXACW', we are importing the data and finding the required best fit. The output of the code contains, value of sigma_c, sigma_p and graph for fitted function and original data.

Required Packages: 
1) time
2) from numba import jit
3) numpy
4) scipy
5) matplotlib

Note: The scaling factor is calculated to be sf=4.34 for the particular case and needed to be changed while running for any arbitrary data.
