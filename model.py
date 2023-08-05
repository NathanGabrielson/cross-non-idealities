from urllib.request import urlopen
from scipy import linalg
import urllib.parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import scipy.sparse as sparse

size = [4]

def crossbar(resis_map, BLres, WLres, resBitOne, resWordOne, resBitTwo, resWordTwo, \
             VoltWLone, VoltBLone, VoltWLtwo, VoltBLtwo, request ):
    
    finalCrossbar = []
    n = len(resis_map[0])
    m = len(resis_map)


    #### Matrix A   

    matrixA = np.zeros((m * n,m * n))

    for count in range (0, m):

        tempA = np.zeros((n,n))

        for i in range(0, n):

            for j in range(0, n):

                if(i==0 and j==0):
                    tempA[i,j] = (1/resWordOne[count]) + (1/resis_map[count, 1]) + (1/WLres)
                elif(i==(n-1) and j==(n-1)):
                    tempA[i,j] = (1/resWordTwo[count]) + (1/resis_map[count, n-1]) + (1/WLres)
                elif(abs(i-j)==1):
                    tempA[i,j] = (-1/WLres)
                elif(i==j):
                    tempA[i,j] = (1/resis_map[count,2]) + (2/WLres)

        offset = count * n

        matrixA[offset:offset + n, offset:offset+n] = tempA
    
    #### Matrix B

    matrixB = np.zeros((m * n,m * n))

    for count in range (0, m):


        for i in range(0, n):

            tempB = np.zeros((n,n))

            for j in range(0,n):

                if (i==j):
                    tempB[i,j] = -1/resis_map[count, i]

        offset = count * n

        matrixB[offset:offset+n, offset:offset+n] = tempB


    #### Matrix C

    matrixC = np.zeros((m*n, m*n))

    for count in range (0, n):
        
        tempC = np.zeros((m, m*n))

        for i in range(0, m):
            tempC[i, n * (i-1) + count] = 1/resis_map[i, count]

        offset = count * n

        matrixC[offset:offset+n,:] = tempC

    #### Matrix D

    matrixD = np.zeros((m*n, m*n))

    for count in range (0, n):

        tempD = np.zeros((m, m*n))

        for i in range(0, m):
            
            if (i == 0):
                tempD[i, count] = ((-1/resBitOne[count]) + (-1/BLres) + (-1/resis_map[i,count]))
                tempD[i, (n * i) + count] = 1/BLres
            elif (1 <= i and i <= (m-2)):
                tempD[i, n*(i-1) + count] = 1/BLres
                tempD[i, n*(i) + count] = ((-1/BLres) + (-1/resis_map[i,count]) + (-1/BLres))
                tempD[i, n*(i+1) + count] = 1/BLres
            elif (i == m-1):
                tempD[i, n*(i-1) + count] = 1/BLres
                tempD[i,n*(i) + count] = ((-1/resBitTwo[count]) + (-1/resis_map[i,j]) + (-1/BLres))
        
        offset = count * n

        matrixD[offset:offset+n, :] = tempD

    #### Matrix E 

    matrixE = np.zeros(m**2 + n**2)
    
    Ew = np.zeros(m**2)
    Eb = np.zeros(n**2)

    for i in range (0, m):

        tempEw = np.zeros((m))
        offset = m * i

        tempEw[0] = VoltWLone[i]/resWordOne[i]
        tempEw[m-1] = VoltWLtwo[i]/resWordTwo[i]


        Ew[offset: offset + m] = tempEw
            
    for i in range (0, n):
        tempEb = np.zeros((n))
        offset = n * i

        tempEb[0] = VoltBLone[i]/resBitOne[i]
        tempEb[n-1] = VoltBLtwo[i]/resBitTwo[i]

        Eb[offset: offset + n] = tempEb

    matrixE[0:m**2] = Ew
    matrixE[m**2:]  = Eb

    ### End Matrices

    upper = np.hstack((matrixA, matrixB))
    lower = np.hstack((matrixC, matrixD))

    combin = np.vstack((upper,lower))

    unknownVolt = np.linalg.solve(combin, matrixE)

    WLvolt, BLvolt = np.array_split(unknownVolt, 2)

    WLvoltShaped = WLvolt.reshape(n,m)
    
    BLvoltShaped = BLvolt.reshape(n,m)


    if (request == 'wl'):
        return WLvoltShaped
    elif (request == 'bl'):
        return BLvoltShaped
    else:
        return 1
    
    




########

size = 4

resis_map = np.full((size,size), 2)

BLres = 500
WLres = 500
resBitOne = [1,2,3,4] #rbiasl in his code
resWordOne = [1,2,3,4] #wllbiasl
resBitTwo = [1,2,3,4] #rbiasr
resWordTwo = [1,2,3,4] #wllbiasr
VoltWLone = np.full(size, 1)
VoltBLone = np.full(size, 1)
VoltWLtwo = np.full(size, 1)
VoltBLtwo = np.full(size, 1)
request = 'wl'
secRequest = 'bl'

result = crossbar(resis_map=resis_map, BLres=BLres, WLres=WLres, resWordOne=resWordOne,\
                   resBitOne=resBitOne, resWordTwo=resWordTwo, resBitTwo=resBitTwo, VoltWLone=VoltWLone,\
                      VoltBLone=VoltBLone, VoltWLtwo=VoltWLtwo, VoltBLtwo=VoltBLtwo, request=request)
secResult = crossbar(resis_map=resis_map, BLres=BLres, WLres=WLres, resWordOne=resWordOne,\
                   resBitOne=resBitOne, resWordTwo=resWordTwo, resBitTwo=resBitTwo, VoltWLone=VoltWLone, \
                    VoltBLone=VoltBLone, VoltWLtwo=VoltWLtwo, VoltBLtwo=VoltBLtwo, request=secRequest)


print('Wordline: \n', result)

print('\nBitline:\n', secResult)



def contour(figloc, ax, axloc, data, title):
  """
  Plots a contour given array data


  Parameters: 
    figloc: figure object of subplot
    ax: axis object of subplot
    axloc: number of subplot, starting from 0, and counting 
      left to right, top to bottom
    data: array to be plotted
    title: title of plot


  Returns:
    None
  """
  size = len(data[0])
  x = np.arange(1,size+1,1)
  y = np.arange(size,0,-1)
  cmap = ax.flat[axloc].pcolormesh(x, y, data, shading = 'auto', 
                                   edgecolors = 'black', cmap='jet')
  #, norm=colors.LogNorm(1e-7, 1e-5))
  figloc.colorbar(cmap, ax = ax.flat[axloc])
  ax.flat[axloc].set_title(title)
  ax.flat[axloc].set_yticks(np.arange(1,size+1,5))
  ax.flat[axloc].set_yticklabels(x[::-5])

fig1, ax1 = plt.subplots(1,2, figsize=(12,6)) 


contour(fig1, ax1, 0, result, 'WL dist, 1V left')
conto