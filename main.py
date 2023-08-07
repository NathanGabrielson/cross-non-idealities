from urllib.request import urlopen
from scipy import linalg
import urllib.parse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import scipy.sparse as sparse

# All Functions
def crossbar_time(Ri, layout_array, Runselhi, Runsellow, plot_code, 
                  wlbiasl, wlbiasr, t_start, t_end, very_high):
  """
  Calculates crossbar arrays over a time interval based on odd and 
  even columns alternating high R values.


  Notes:
    The assumption is that the upper side of BLs are floating, and 
    lower side are grounded.


  Parameters: 
    Ri: interconnect R (assumed to be same for WL, BL)
    layout_array: resistance map in array form
    Runselhi: R_sense that simulates floating (very high resistance)
    Runsellow: R_sense that simulates wire (very low resistance)
    plot_code: choose value to plot
      'wl': WL voltage distribution
      'bl': BL voltage distribution
      'wl-bl': (WL-BL) voltage distribution
      'current-act': current distribution for WL bias applied 
        all at once
      'current-obs': current distribution for WL bias applied 
        line by line
    wllbiasl: List of WL bias values for the left side
    wllbiasr: List of WL bias values for the right side
    t_start, t_end: start and end values
    very_high: R value that changes based on select


  Returns:
    List of list of currents
      E.g. [[3,5,6],[20,30,50]]
        [3,5,6] means that I_1 changes from 3 to 5 to 6
    List of crossbar arrays for desired value (e.g. current), each 
      corresponding to a time point
    List of Rmap arrays, each corresponding to a time point


  """
  copy = layout_array.copy()


  currents_per_time = []
  array_per_time = []
  layout_per_time = []


  for time in range(t_start, t_end):
    if (time % 2 != 0):
      layout_array[:,::2] = very_high
      layout_array[:,1::2] = copy[:,1::2]
    else:
      layout_array[:,1::2] = very_high
      layout_array[:,::2] = copy[:,::2]
    
    current = crossbar(Ri, layout_array, Runselhi, Runsellow, plot_code, 
                       wlbiasl, wlbiasr)

    array_per_time.append(current)
    layout_per_time.append(layout_array.copy())


    current_sums = np.sum(current, axis = 0)


    num_outputs = int(len(current_sums)/2)
    
    shorted_sums = np.zeros(num_outputs)


    for i in range(0,num_outputs):
      shorted_sums[i] = current_sums[i] + current_sums[i+1]


    currents_per_time.append(shorted_sums)
  
  # asterisk unzips the currents
  return list(zip(*currents_per_time)), array_per_time, layout_per_time 


def crossbar(Ri, layout_array, Runselhi, Runsellow, plot_code, 
             wlbiasl, wlbiasr):


    """
    Calculates a single crossbar


    Notes:
      The assumption is that the upper side of BLs are floating, 
      and lower side are grounded.


    Parameters: 
      Ri: interconnect R (assumed to be same for WL, BL)
      layout_array: resistance map in array form
      Runselhi: R_sense that simulates floating (very high resistance)
      Runsellow: R_sense that simulates wire (very low resistance)
      plot_code: choose value to plot
        'wl': WL voltage distribution
        'bl': BL voltage distribution
        'wl-bl': (WL-BL) voltage distribution
        'current-act': current distribution for WL bias 
          applied all at once.
        'current-obs': current distribution for WL bias 
          applied line by line.
      wllbiasl: List of WL bias values for the left side
      wllbiasr: List of WL bias values for the right side


    Returns:
      Single crossbar array
    """
    if(plot_code == 'current_obs'):
      array_size = len(layout_array[0])
      
      obs_array = np.zeros((array_size,array_size))


      for i in range(0,array_size):
        wlbiasl_temp = np.zeros(array_size)
        wlbiasr_temp = np.zeros(array_size)


        wlbiasl_temp[i] = wlbiasl[i]
        wlbiasr_temp[i] = wlbiasr[i]


        result = my_function(Ri, layout_array, Runselhi, 
                             Runsellow, 'current_act', 
                             wlbiasl_temp, wlbiasr_temp)
        current_sum = np.sum(result, axis=0)
        obs_array[i,:] = current_sum


      return obs_array
    else:
      return my_function(Ri, layout_array, Runselhi, Runsellow, 
                         plot_code, wlbiasl, wlbiasr)

def my_function(Ri, layout_array, Runselhi, Runsellow, 
                plot_code, wlbiasl, wlbiasr):
    """
    Calculates a single crossbar, except for applying 
    line-by-line bias.


    Notes:
      The assumption is that the upper side of BLs are 
      floating, and lower side are grounded.


    Parameters: 
      Ri: interconnect R (assumed to be same for WL, BL)
      layout_array: resistance map in array form
      Runselhi: R_sense that simulates floating (very high resistance)
      Runsellow: R_sense that simulates wire (very low resistance)
      plot_code: choose value to plot
        'wl': WL voltage distribution
        'bl': BL voltage distribution
        'wl-bl': (WL-BL) voltage distribution
        'current-act': current distribution for WL bias 
          applied all at once.
      wllbiasl: List of WL bias values for the left side
      wllbiasr: List of WL bias values for the right side


    Returns:
      Single crossbar array
    """
    array_size = len(layout_array[0])


    Rbiasl = wlbiasl
    Rbiasl = Rbiasl.astype('float32')

    Rbiasl[Rbiasl == 0] = Runselhi
    Rbiasl[Rbiasl != Runselhi] = Runsellow



    Rbiasr = wlbiasr
    Rbiasr = Rbiasr.astype('float32')
    Rbiasr[Rbiasr == 0] = Runselhi
    Rbiasr[Rbiasr != Runselhi] = Runsellow


    matrix_a = np.zeros((array_size**2, array_size**2), 
                        dtype='float32')
    
  
    for main_count in range(0,array_size):
        sub_matrix_a = np.zeros((array_size, array_size), 
                                dtype='float32')
        
        a = array_size*main_count


        for i in range(0, array_size):
            for j in range(0, array_size):
  
                if ((i == 0) & (j == 0)):
                    # the other wordlines
                    sub_matrix_a[i, j] = (1/Rbiasl[main_count]) + \
                      (1/layout_array[main_count, j]) + (1/Ri) 
                elif (abs(i-j) == 1):
                    sub_matrix_a[i, j] = (-1/Ri)
                elif ((i == j) & (i == array_size - 1)):
                    sub_matrix_a[i, j] = (1/Rbiasr[main_count]) + \
                     (1/layout_array[main_count, j]) + (1/Ri)
                elif (i == j):
                    sub_matrix_a[i, j] = \
                     (1/layout_array[main_count, j]) + (2/Ri)
                    
        

        matrix_a[a:a+array_size, a:a+array_size] = sub_matrix_a


    ################################
    
    matrix_b = np.zeros((array_size**2, array_size**2), dtype='float32')


    for main_count in range(0,array_size):
        sub_matrix_b = np.zeros((array_size, array_size), dtype='float32')


        b = array_size*main_count


        for i in range(0, array_size):
            for j in range(0, array_size):
                if (i == j):
                    sub_matrix_b[i, j] = (-1/layout_array[main_count,j])


        matrix_b[b:b+array_size, b:b+array_size] = sub_matrix_b
        
    ##################################
    
    matrix_c = np.zeros((array_size**2, array_size**2), 
                        dtype='float32')

    for main_count in range(0,array_size):
        sub_matrix_c = np.zeros((array_size, array_size**2), 
                                dtype='float32')


        c = array_size*main_count


        for i in range(0, array_size):
            sub_matrix_c[i, array_size*(i) + main_count] = \
              1/layout_array[i,main_count]

        matrix_c[c:c+array_size, :] = sub_matrix_c
        
    ######################################
    
    matrix_d = np.zeros((array_size**2, array_size**2), dtype='float32')


    for main_count in range(0,array_size):
        sub_matrix_d = np.zeros((array_size, array_size**2), dtype='float32')


        d = array_size*main_count


        for i in range(0, array_size):
            if (i == 0):
                sub_matrix_d[i, main_count] = (-1/Runselhi) + (-1/Ri) + \
                 (-1/layout_array[i, main_count]) # to make top bitlines floating
                sub_matrix_d[i, array_size*(i+1) + main_count] = (1/Ri)
            elif (i >= 1) & (i <= (array_size - 2)):
                sub_matrix_d[i, array_size*(i-1) + main_count] = (1/Ri)
                sub_matrix_d[i, array_size*(i) + main_count] = (-2/Ri) + \
                 (-1/layout_array[i, main_count])
                sub_matrix_d[i, array_size*(i+1) + main_count] = (1/Ri)
            elif (i == (array_size - 1)) & (main_count == 0):
                sub_matrix_d[i, array_size*(i-1) + main_count] = (1/Ri)
                sub_matrix_d[i, array_size*(i) + main_count] = (-1/Ri) + \
                 (-1/layout_array[i, main_count]) + (-1/Runsellow)
            elif (i == (array_size - 1)):
                sub_matrix_d[i, array_size*(i-1) + main_count] = (1/Ri)
                sub_matrix_d[i, array_size*(i) + main_count] = (-1/Ri) + \
                 (-1/layout_array[i, main_count]) + (-1/Runsellow)


        matrix_d[d:d+array_size, :] = sub_matrix_d
    
    ########################################
    
    matrix_e = np.zeros((2*array_size**2, 1), dtype='float32')


    matrix_e[:int(array_size**2):array_size, 0] = wlbiasl / Rbiasl
    matrix_e[array_size-1:int(array_size**2):array_size, 0] = \
      wlbiasr / Rbiasr


    ################################
    
    top_half = np.hstack((matrix_a, matrix_b))
    bottom_half = np.hstack((matrix_c, matrix_d))


    abcd = np.vstack((top_half, bottom_half))

    voltage_dist = np.linalg.solve(abcd, matrix_e)

    wl_volt_dist, bl_volt_dist = np.array_split(voltage_dist, 2)

    wl_volt_dist_array = wl_volt_dist.reshape(array_size, array_size)
    bl_volt_dist_array = bl_volt_dist.reshape(array_size, array_size)

    wl_minus_bl_array = wl_volt_dist_array - bl_volt_dist_array
    
    if (plot_code == 'wl'):
      return wl_volt_dist_array
    elif (plot_code == 'bl'):
      return bl_volt_dist_array
    elif (plot_code == 'wl-bl'):
      return wl_minus_bl_array
    elif (plot_code == 'current_act'):
      return wl_minus_bl_array / layout_array
    
def rmap(Roff, Ron, mode, size):
  """
  Calculates a resistance array map


  Parameters: 
    Roff: R value when memory device is off
    Ron: R value when memory device is on
    mode: type of resistance map
      -'random': random arrangement of R_on, R_off
      -'on': all R_on
      -'off': all R_off
      -'checker': checkerboard of R_on, R_off


  Returns:
    Single Rmap array
  """
  if(mode == 'random'):
    return np.random.choice([Roff,Ron], size**2).reshape(size, size)
  elif(mode == 'on'):
    return np.full((size, size), Ron)
  elif(mode == 'off'):
    return np.full((size, size), Roff)
  elif(mode == 'checker'):
    zero_ones = np.indices((size,size)).sum(axis=0) % 2
    zero_ones[zero_ones == 0] = Ron
    zero_ones[zero_ones != Ron] = Roff
    return zero_ones


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


def singleplot(figloc, ax, axloc, data_x, data_y, label, 
               marker, title):
  """
  Plots a scatter given x and y data


  Parameters: 
    figloc: figure object of subplot
    ax: axis object of subplot
    axloc: number of subplot, starting from 0, and going 
      left to right, top to bottom
    data_x: list of x-axis data to be plotted
    data_y: list of y_axis data to be plotted
      Each element corresponds to a series 
        e.g. [[1,5,6], [16,17,17]]
        [1,5,6] is for I1, [16,17,17] is for I2
    label: series label to be shown in legend
    marker: appearance of data points 
      (e.g. '.', 'o', '-', '--')
    title: title of plot


  Returns:
    None
  """
  for i in range(0,len(data_y)):
    ax.flat[axloc].plot(np.arange(t_start, t_end, 1), 
                        data_y[i], marker, label='I_' + str(i+1))
  ax.flat[axloc].set_title(title)
  ax.flat[axloc].legend(loc='lower right')

# I_CV Plots


sizes = [4] # 1-element array with desired n-value for n x n crossbar
Ri = 500
Ron = 33333
Roff = 333333
Runselhi = 10e20
Runsellow = 10e-20
plot_code = 'current_act'


# 4-element list for left WL bias, alternating between 1 and 0.5
wlbiasl = np.resize([1,0.5], sizes[0]) 


# 4-element list for right WL bias, is all zeros
wlbiasr = np.array([0 for i in range(0,sizes[0])])


t_start = 1
t_end = 6
very_high = np.array(1e15, dtype=np.int64)


# creates 3-row, 4-column subplot that is 12 wide and 9 high, 
# and stores into figure and axis objects
figcurrent2, ax2 = plt.subplots(3,(t_end-t_start),figsize=(12,6))


# calculates a checkerboard Rmap array and 
# assigns to variable layout_array
layout_array = rmap(Roff=Roff, Ron=Ron, mode='checker', size=sizes[0])


# calculates current distribution array for a crossbar 
# and assigns to variable current
current = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                   Runsellow=Runsellow, plot_code=plot_code, 
                   wlbiasl=wlbiasl, wlbiasr=wlbiasr)


# calculates WL voltage distribution array for a crossbar 
# and assigns to variable volt
volt = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                Runsellow=Runsellow, plot_code='wl', 
                wlbiasl=wlbiasl, wlbiasr=wlbiasr)


# calculates current distribution for all-at-once WL bias, for defined time interval
# variables
# # list_outputs: list of list of currents
# # array_outputs: list of current distribution arrays
# # layouts: list of Rmap arrays
list_outputs, array_outputs, layouts = \
  crossbar_time(Ri, layout_array, Runselhi, Runsellow, 
                plot_code, wlbiasl, wlbiasr, t_start, t_end, very_high)


# iterate over number of Rmap arrays
for i in range(0,len(layouts)):
  # contour plot Rmap on subplots 0 to 3, for t = 1 to t = 4
  contour(figcurrent2, ax2, i, layouts[i], 'Rmap, t = ' + str(i+1))


  # contour plot current distribution on 
  # subplots 4 to 7, for t = 1 to t = 4
  contour(figcurrent2, ax2, i+(t_end-t_start), 
          array_outputs[i], 'currents, t = ' + str(i+1))


# plots scatterplot of current values for times 1,2,3,4, on 
# subplot 8, with line marker type and 
# label of 'I_' for each current on legend
singleplot(figcurrent2, ax2, 2*(t_end-t_start), [1,2,3,4], 
           list_outputs, 'I_', '-', 'currents')


# plots voltage distribution for WL on subplot 9
contour(figcurrent2, ax2, 2*(t_end-t_start)+1, volt, 'WL dist')


figx, axx = plt.subplots(1,2,figsize=(12,3))
singleplot(figx, axx, 0, [1,2,3,4], list_outputs, 'I_', '-', 'currents')


# makes all the plots fit
plt.tight_layout()

# Wordline Voltage Plot


fig3, ax = plt.subplots(1,3,figsize=(12, 4))


Ron = (100/3)*1e3
Roff = (1000/3)*1e3


sizes = [16,32,64]


Rwl = 1
Rbl = 1
Runselwl=10e20
Runselbl=10e-20
Vapp=1
Rapp=1e-10
Rground=11
plot_code = 'wl_volt_dist'


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
        
        answer = my_function(Ri=1, layout_array=layouts[i], Runselhi=10e20, 
                             Runsellow=10e-20, plot_code='wl', 
                             wlbiasl=left, wlbiasr=right)
        wls_only = answer[0:int(sizes[j]**2)].reshape(sizes[j],sizes[j])
        ax[i].plot(x, wls_only[0],label = str(sizes[j])+' x '+ str(sizes[j]))
        ax[i].set_title('Average R=' + str(np.average(layouts[i])))
        ax[i].set_ylim(0,1)
        ax[i].set_xlim(0,sizes[len(sizes)-1])
        
    ax[i].legend(loc="lower right")

# Voltage Plots


sizes = [20] # 1-element array with desired n-value for n x n crossbar
Ri = 500
Ron = 33333
Roff = 333333
Runselhi = 10e20
Runsellow = 10e-20
plot_code = 'current_act'


# 4-element list for right WL bias, is all zeros
wlbias_zeros = np.array([0 for i in range(0,sizes[0])]) 
wlbias_one_v = wlbias_zeros.copy()
wlbias_one_v[:] = 1
t_start = 1
t_end = 5
very_high = 1e15


fig3, ax3 = plt.subplots(2,4,figsize=(12,6))


layout_array = rmap(Roff=Roff, Ron=Ron, mode='random', size=sizes[0])


wl_1side = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                    Runsellow=Runsellow, plot_code='wl', 
                    wlbiasl=wlbias_one_v, wlbiasr=wlbias_zeros)
bl_1side = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                    Runsellow=Runsellow, plot_code='bl', 
                    wlbiasl=wlbias_one_v, wlbiasr=wlbias_zeros)
wl_bl_1side = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                       Runsellow=Runsellow, plot_code='wl-bl', 
                       wlbiasl=wlbias_one_v, wlbiasr=wlbias_zeros)
current_1side = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                         Runsellow=Runsellow, plot_code='current_act', 
                         wlbiasl=wlbias_one_v, wlbiasr=wlbias_zeros)


wl_2side = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                    Runsellow=Runsellow, plot_code='wl', 
                    wlbiasl=wlbias_one_v, wlbiasr=wlbias_one_v)
bl_2side = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                    Runsellow=Runsellow, plot_code='bl', 
                    wlbiasl=wlbias_one_v, wlbiasr=wlbias_one_v)
wl_bl_2side = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                       Runsellow=Runsellow, plot_code='wl-bl', 
                       wlbiasl=wlbias_one_v, wlbiasr=wlbias_one_v)
current_2side = crossbar(Ri=Ri, layout_array=layout_array, Runselhi=Runselhi, 
                         Runsellow=Runsellow, plot_code='current_act', 
                         wlbiasl=wlbias_one_v, wlbiasr=wlbias_one_v)


contour(fig3, ax3, 0, wl_1side, 'WL dist, 1V left')
contour(fig3, ax3, 1, bl_1side, 'BL dist, 1V left')
#contour(fig3, ax3, 2, wl_bl_1side, 'WL-BL dist, 1V left')
#contour(fig3, ax3, 3, current_1side, 'current-actual dist, 1V left')
contour(fig3, ax3, 4, wl_2side, 'WL dist, 1V left-right')
contour(fig3, ax3, 5, bl_2side, 'BL dist, 1V left-right')
#contour(fig3, ax3, 6, wl_bl_2side, 'WL-BL dist, 1V left-right')
#contour(fig3, ax3, 7, current_2side, 'current-actual dist, 1V left-right')


# makes all the plots fit
plt.tight_layout()

# Current Plots


sizes = [20] # 1-element array with desired n-value for n x n crossbar
Ri = 500
Ron = 33333
Roff = 333333
Runselhi = 10e20
Runsellow = 10e-20
plot_code = 'current_act'


# 4-element list for right WL bias, is all zeros
wlbias_zeros = np.array([0 for i in range(0,sizes[0])]) 
wlbias_one_v = wlbias_zeros.copy()
wlbias_one_v[:] = 1
t_start = 1
t_end = 5
very_high = 1e15


fig3, ax3 = plt.subplots(1,4,figsize=(12,3))


layout_array = rmap(Roff=Roff, Ron=Ron, mode='random', size=sizes[0])


current_1side = crossbar(Ri=Ri, layout_array=layout_array, 
                         Runselhi=Runselhi, Runsellow=Runsellow, 
                         plot_code='current_act', wlbiasl=wlbias_one_v, 
                         wlbiasr=wlbias_zeros)
current_1side_obs = crossbar(Ri=Ri, layout_array=layout_array, 
                             Runselhi=Runselhi, Runsellow=Runsellow, 
                             plot_code='current_obs', wlbiasl=wlbias_one_v, 
                             wlbiasr=wlbias_zeros)


current_2side = crossbar(Ri=Ri, layout_array=layout_array, 
                         Runselhi=Runselhi, Runsellow=Runsellow, 
                         plot_code='current_act', wlbiasl=wlbias_one_v,
                         wlbiasr=wlbias_one_v)
current_2side_obs = crossbar(Ri=Ri, layout_array=layout_array, 
                             Runselhi=Runselhi, Runsellow=Runsellow, 
                             plot_code='current_obs', wlbiasl=wlbias_one_v, 
                             wlbiasr=wlbias_one_v)


contour(fig3, ax3, 0, current_1side, 'current-actual, 1V left')
contour(fig3, ax3, 1, current_1side_obs, 'current-observed, 1V left')
contour(fig3, ax3, 2, current_2side, 'current-actual, 1V left-right')
contour(fig3, ax3, 3, current_2side_obs, 'current-observed, 1V left-right')


# makes all the plots fit
plt.tight_layout()
