# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:02:19 2024

@author: Bram Verreussel
"""

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from colorsys import hsv_to_rgb
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import sys
import warnings
if 'cv2' in sys.modules:
    import cv2
    CV2_IMPORTED = True
else:
    warnings.warn('cv2 not installed. Using fallback')
    CV2_IMPORTED = False


#%%
def colorize(z):
    r = np.abs(z)
    arg = np.angle(z) 
    
    
    r = np.clip(r, 0, 1)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    v = r
    s = 1
    
    v = np.clip(v, 0, 1)
    
    c = np.vectorize(hsv_to_rgb) (h,s,v) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose(1,2,0) 
    return c



def complex_imshow(img: np.array, ax=None, plt=None, display_function=colorize,
                   norm=None, abs_colorbar=True, phase_colorbar=True, 
                   **imshow_args
                   ):
    if ax is None:
        if plt is None:
            raise ValueError('Provide either the `ax` or the `plt` keyword')
        else:
            fig, ax = plt.subplots()

    # `normalize` is function that maps [vmin, vmax] to [0, 1]
    vmin, vmax = np.min(np.abs(img)), np.max(np.abs(img))
    normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    # display image using `display_function`
    img_data = display_function(normalize(img))
    ax_img = ax.imshow(img_data)

    if abs_colorbar or phase_colorbar:
        # perform some magic so colorbar have same height as image
        ax_divider = make_axes_locatable(ax)

    # we want 6% padding for first colorbar and 15% for second
    padding = "6%"

    if abs_colorbar:
        # define array of colors to show in colorbar
        abs_colors = display_function( [ np.linspace( 0, 1, 200 ) ] )[0]
        
        # convert array to colormaps
        abs_cmap = ListedColormap(abs_colors)
        
        abs_colorbar_ax = ax_divider.append_axes('right', size='5%', pad=padding)
        padding = "15%"
        
        ax_img.figure.colorbar(
                        matplotlib.cm.ScalarMappable(norm=normalize, cmap=abs_cmap),
                        cax=abs_colorbar_ax,
                        label=''
                        )
         
    if phase_colorbar:
        phase_colors = display_function( [ np.exp(1j*np.linspace( 0, 2*np.pi, 200 )) ] )[0]
        phase_cmap = ListedColormap(phase_colors)
        phase_colorbar_ax = ax_divider.append_axes('right', size='5%', pad=padding)
        normalize_phase = matplotlib.colors.Normalize(vmin=0, vmax=2*np.pi, clip=False)
        cbar = ax_img.figure.colorbar(
                        matplotlib.cm.ScalarMappable(
                                            norm=normalize_phase, 
                                            cmap=phase_cmap
                                            ),
                        cax=phase_colorbar_ax,
                        label=''
                        )
        
        # for phase use fancy ticks that show multiples of pi
        cbar.set_ticks(np.arange(4 + 1)*np.pi/2)
        cbar.set_ticklabels(['$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])


# shorthand for complex_imshow:
cimshow = complex_imshow

#%%

X, Y = np.meshgrid(np.arange(100) - 50, np.arange(100) - 50)
img = X + 1j*Y

# simplest use case
plt.title('simple use case')
cimshow(img, plt=plt)
plt.show()
# If you do plt.title after cimshow, the title shows above the colorbars

# use axes instead of plt, low dpi to show difference
fig, ax = plt.subplots(dpi=25)
cimshow(-img, ax=ax)
ax.set_title('use ax instead of plt')
plt.show()

# turn off colorbars independently
fig, ax = plt.subplots(dpi=300)
plt.title('no colorbars')
cimshow(img, ax=ax, abs_colorbar=False, phase_colorbar=False)
plt.show()

# show off (weird) custom display function
def colorize2(z):
    # must be normalized!
    z /= np.max(np.abs(z) ) + .00001
    
    
    real = np.real(z)
    result = np.array( [np.clip(real, 0, 1), np.abs(z)*np.abs(np.imag(z)), np.clip( - real, 0, 1)] )
    result = np.transpose(result, axes=(1, 2, 0))
    return result

fig, ax = plt.subplots(dpi=300)
plt.title('custom display function')
cimshow(img, ax=ax, display_function=colorize2)
plt.show()

#%%
import cv2
import numpy as np
import matplotlib.pyplot as plt
import hsluv
import numexpr
def colorize_cv(z):
    # x = np.arange(256, dtype='float')
    # y = np.arange(256, dtype='float')
    # X, Y = np.meshgrid(x, y)
    # zero = X*0
    # one = zero + 1
    # # rgb = np.array([X, Y, zero])
    # # rgb = rgb.transpose((1, 2, 0))
    # # bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    # # lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    # # lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    r = np.abs(z)
    x = np.real(z)
    y = np.imag(z)
    h = np.angle(z)
    h = h % ( 2*np.pi )
    h_deg = h*180/np.pi
    
    s = np.ones(h.shape)*100
    l = r*60
    hsl = np.array([h_deg,s,l])
    c = np.apply_along_axis(hsluv.hsluv_to_rgb, 0, hsl)# --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose(1,2,0) 
    return c
    L = r - 1
    a = x
    b = y
    norm = np.sqrt(L**2 + a**2 + b**2)
    L = L*128/norm + 128
    a = a*128/norm + 128
    b = b*128/norm + 128
    # L = X
    # a = (np.cos(Y/256*2*np.pi))*128*X/256 + 128
    # b = (np.sin(Y/256*2*np.pi))*128*X/256 + 128
    img = np.array([L, a, b]).astype('uint8')
    img = img.transpose((1, 2, 0))
    bgr = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cv2.imshow('image', bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return rgb, lab

n = 100
x = np.linspace(-1, 1, num=n, dtype='float')
y = np.linspace(-1, 1, num=n, dtype='float')
X, Y = np.meshgrid(x, y)
Z = X + 1j*Y
Z[np.where(np.abs(Z)>1.)] = 1.
c = colorize_cv(Z)
plt.imshow(c)
# plt.imshow(rgb)
# plt.show()
# plt.imshow(lab[:,:,0])
# plt.colorbar()
# plt.show()
# plt.imshow(lab[:,:,1])
# plt.colorbar()
# plt.show()
# plt.imshow(lab[:,:,2])
# plt.colorbar()
# plt.show()
# %%
h = np.arange(10)
s = np.arange(10)
l = np.arange(10)
import numexpr as ne

ne.evaluate('hsluv.hsluv_to_rgb([h, s, l])')
# %%
from numba import njit
import numba
hsluv_jit = numba.jit(nopython=True)(hsluv.hsluv_to_rgb)
# %%
hsluv_jit([h, s, l])
# %%
