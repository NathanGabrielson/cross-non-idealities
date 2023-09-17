from urllib.request import urlopen
from scipy import linalg
import urllib.parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import scipy.sparse as sparse
import sys

np.set_printoptions(threshold=sys.maxsize)

def crossbar(resis_map, BLres, WLres, resBitOne, resWordOne, resBitTwo, resWordTwo, \
             VoltWLone, VoltBLone, VoltWLtwo, VoltBLtwo, request ):
    
    n = len(resis_map[0])
    m = len(resis_map)

    #### Matrix A   

    matrixA = np.zeros((m * n,m * n), dtype='float32')

    for count in range (0, m):

        tempA = np.zeros((n,n), dtype='float32')

        offset = count * n

        for i in range(0, n):

            for j in range(0, n):

                if(i==0 and j==0):
                    tempA[i,j] = (1/resWordOne[count]) + (1/resis_map[count, 0]) + (1/WLres)
                elif(abs(i-j)==1):
                    tempA[i,j] = (-1/WLres)
                elif(i==(n-1) and j==(n-1)):
                    tempA[i,j] = (1/resWordTwo[count]) + (1/resis_map[count, n-1]) + (1/WLres)
                elif(i==j):
                    tempA[i,j] = (1/resis_map[count,j]) + (2/WLres)
        
        matrixA[offset:offset + n, offset:offset+n] = tempA

    #### Matrix B

    matrixB = np.zeros((m * n,m * n), dtype='float32')

    for count in range (0, m):

        tempB = np.zeros((n,n), dtype='float32')

        
        for i in range(0, n):

            

            for j in range(0,n):

                if (i==j):
                    tempB[i,j] = -1/resis_map[count, i]

        offset = count * n

        
        matrixB[offset:offset+n, offset:offset+n] = tempB


    #### Matrix C

    matrixC = np.zeros((m*n, m*n), dtype='float32')

    for count in range(0,n):
        tempC = np.zeros((m, m*n), 
                                dtype='float32')
        

        offset = m*count   

        for i in range(0, m):
            tempC[i, n*(i) + count] = \
              1/resis_map[i,count]

        matrixC[offset : offset+n, :] = tempC
    
    #### Matrix D

    matrixD = np.zeros((m*n, m*n), dtype='float32')

    for count in range (0, n):

        tempD = np.zeros((m, m*n), dtype='float32')

        for i in range(0, m):
            
            if (i == 0):
                tempD[i, count] = (-1/resBitOne[count]) + (-1/BLres) + (-1/resis_map[i,count])
                tempD[i, n + count] = 1/BLres

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

    
    matrixE = np.zeros((2 * (m * n)), dtype='float32')
    
    Ew = np.zeros((m**2), dtype='float32')
    Eb = np.zeros((n**2),dtype='float32')

    for i in range (0, m):

        tempEw = np.zeros((m), dtype='float32')
        offset = m * i
        tempEw[0] = VoltWLone[i]/resWordOne[i]
        tempEw[m-1] = VoltWLtwo[i]/resWordTwo[i]


        Ew[offset: offset + m] = tempEw
            
    for i in range (0, n):
        tempEb = np.zeros((n), dtype='float32')
        offset = n * i

        tempEb[0] = -(VoltBLone[i]/resBitOne[i])
        tempEb[n-1] = -(VoltBLtwo[i]/resBitTwo[i])

        #Eb[offset: offset + n] = tempEb 
        # Commented out as it seems to produce better results
        # I'm not quite sure why this is the case, however

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


size = 20
sizes=[size]
high=10e20
low=10e-20
ron = 33333
roff = 333333

wlbias_zeros = np.array([0 for i in range(0,sizes[0])]) 
wlbias_one_v = wlbias_zeros.copy()
wlbias_one_v[:] = 1


resWordOne = wlbias_one_v.astype('float32')
resWordOne[resWordOne == 0] = high
resWordOne[resWordOne!=high] = low


resWordTwo = wlbias_zeros.astype('float32') 
resWordTwo[resWordTwo==0] = high
resWordTwo[resWordTwo!=high] = low


BLres = 500
WLres = 500


resBitOne = np.full(size, high, dtype='float32') 
resBitTwo = np.full(size, low, dtype='float32') 


VoltWLone = np.full(size, 1, dtype='float32')
VoltWLtwo = np.full(size, 0, dtype='float32')
VoltBLone = np.full(size, 1, dtype='float32')
VoltBLtwo = np.full(size, 1, dtype='float32')

resis_map = np.random.choice([ron,roff], size**2).reshape(size,size)

request = 'wl'

WLoneVolt = crossbar(resis_map=resis_map, BLres=BLres, WLres=WLres, resWordOne=resWordOne,\
                   resBitOne=resBitOne, resWordTwo=resWordTwo, resBitTwo=resBitTwo, VoltWLone=VoltWLone,\
                      VoltBLone=VoltBLone, VoltWLtwo=VoltWLtwo, VoltBLtwo=VoltBLtwo, request=request)


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
Ron = (100/3)*1e3
Roff = (1000/3)*1e3

contour(fig1, ax1, 0, WLoneVolt, 'WL dist, 1V left')



fig2, ax2 = plt.subplots(1,3, figsize=(12,4))
sizes=[16,32,64]


for i in range(0,3):
    for j in range(0,3):
        
        array_size = sizes[j]
        
        x = np.arange(1,array_size+1,1)
        
        all_on = np.full((array_size, array_size), Ron)
        all_off = np.full((array_size, array_size), Roff)
        all_random = \
          np.random.choice([Roff,Ron], 
                           array_size**2).reshape(array_size, array_size)
        
        layouts = [all_on, all_off, all_random]
        
        right = np.zeros(array_size)
        left = right.copy()
        left[0] = 1
        

        resWordOne = left.astype('float32')
        resWordOne[resWordOne == 0] = 10e20
        resWordOne[resWordOne!=10e20] = 10e-20

        resWordTwo = right.astype('float32')
        resWordTwo[resWordTwo == 0] = 10e20
        resWordTwo[resWordTwo != 10e20] = 10e-20

        resBitOne = np.full(sizes[j], high, dtype='float32') 
        resBitTwo = np.full(sizes[j], low, dtype='float32') 

        answer = crossbar(resis_map=layouts[i] ,BLres=1, WLres=1, resBitOne=resBitOne,\
                           resBitTwo=resBitTwo,request='wl', VoltWLone=left, VoltWLtwo=left, \
                           VoltBLone=right, VoltBLtwo=right, resWordOne= resWordOne, \
                            resWordTwo = resWordTwo)

        wls_only = answer[0:int(sizes[j]**2)].reshape(sizes[j],sizes[j])
        ax2[i].plot(x, wls_only[0],label = str(sizes[j])+' x '+ str(sizes[j]))
        ax2[i].set_title('Average R=' + str(np.average(layouts[i])))
        ax2[i].set_ylim(0,1)
        ax2[i].set_xlim(0,sizes[len(sizes)-1])
        
    ax2[i].legend(loc="lower right")

