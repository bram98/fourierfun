# -*- coding: utf-8 -*-
"""
Created on Tue Jun 18 12:02:19 2024

@author: Bram Verreussel
"""

#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
from colorsys import hsv_to_rgb
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import oklab
from dataclasses import dataclass
from numpy.typing import ArrayLike

__all__ = ['cimshow', 'complex_imshow', 'colorize', 'colorize_hsv', 
           'colorize_oklab', 'CustomColorbar']

#%%
def colorize_hsv(z):
    '''
    https://stackoverflow.com/a/20958684/3502079
    '''
    r = np.abs(z)
    arg = np.angle(z) 
    
    oklab.oklch_to_rgb
    r = np.clip(r, 0, 1)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    v = r
    s = 1
    
    v = np.clip(v, 0, 1)
    
    c = np.vectorize(hsv_to_rgb) (h,s,v) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose(1,2,0) 
    return c

def colorize_oklab(z):
    r = np.abs(z)
    arg = np.angle(z)
    
    r = np.clip(r, 0, 1)
    
    l = r*0.78
    c = r*0.11
    h = arg/(2*np.pi)
    
    img_lch = oklab.list_to_img([l, c, h])
    img_rgb = oklab.oklch_to_rgb(img_lch, clip=True)
    
    return img_rgb

colorize = colorize_oklab

def z_to_cmap(z, display_function, scalar_output, cmap):
    colors = display_function( [ z ] )[0]
    
    # convert array to colormaps
    if scalar_output:
        return colormaps[cmap]
    else:
        return ListedColormap(colors)

def make_colorbar(cmap, ax_img, ax_divider, padding, vmin=0, vmax=1, 
                  ticks=None, ticklabels=None, label=''):
    # # define array of colors to show in colorbar
    # abs_colors = display_function( [ np.linspace( 0, 1, 200 ) ] )[0]
    
    # # convert array to colormaps
    # if scalar_output:
    #     abs_cmap = colormaps[cmap]
    # else:
    #     abs_cmap = ListedColormap(abs_colors)
    
    cbar_ax = ax_divider.append_axes('right', size='5%', pad=padding)
    padding = "15%"
    
    
    normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    
    cbar = ax_img.figure.colorbar(
                    matplotlib.cm.ScalarMappable(norm=normalize, cmap=cmap),
                    cax=cbar_ax,
                    label='')
    cbar_ax.set_title(label)
    if not ticks is None:
        cbar.set_ticks(ticks)
    if not ticklabels is None:
        cbar.set_ticklabels(ticklabels)

@dataclass
class CustomColorbar:
    z: ArrayLike
    label: str = ''
    ticks: ArrayLike = None
    ticklabels: ArrayLike = None
    vmin: float = None
    vmax: float = None
    

re_cbar = CustomColorbar(
    z=np.linspace(0, 1, num=200),
    label='re'
    )
im_cbar = CustomColorbar(
    z=np.linspace(0, 1j, num=200),
    label='im'
    )

def complex_imshow(img: np.array, ax=None, plt=None, display_function=colorize,
                   norm=None, abs_colorbar=False, phase_colorbar=False, 
                   cmap='viridis', vmin=None, vmax=None, custom_cbars=[],
                   **imshow_args
                   ):
    if ax is None:
        if plt is None:
            raise ValueError('Provide either the `ax` or the `plt` keyword')
        else:
            fig, ax = plt.subplots()

    # `normalize` is function that maps [vmin, vmax] to [0, 1]
    if vmin is None or vmax is None:
        abs_img = np.abs(img)
        
        if vmin is None:
            vmin = np.min(abs_img)
            
        if vmax is None:
            vmax = np.max(abs_img)
    normalize = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax, clip=False)

    # display image using `display_function`
    img_data = display_function(normalize(img))
    scalar_output = len(img_data.shape) == 2
    ax_img = ax.imshow(img_data, cmap=cmap)

    

    # we want 6% padding for first colorbar and 15% for second
    padding = "6%"
    
    cbars = []
    
    if abs_colorbar:
        abs_cbar = CustomColorbar(
                z=np.linspace( 0, 1, 200 ),
                label='abs',
                vmin=None,
                vmax=None
            )
        cbars.append(abs_cbar)
    
    if phase_colorbar:
        phase_cbar = CustomColorbar(
                z=np.exp(1j*np.linspace( 0, 2*np.pi, 200 )),
                label='arg',
                ticks=np.arange(4 + 1)*np.pi/2,
                ticklabels=['$0$', r'$\pi/2$', r'$\pi$',
                            r'$3\pi/2$', r'$2\pi$'],
                vmin=0,
                vmax=2*np.pi
            )
        
        cbars.append(phase_cbar)
        
    cbars += custom_cbars
    
    if len(cbars) > 0:
        # perform some magic so colorbars have same height as image
        ax_divider = make_axes_locatable(ax)
    
    print(len(cbars))
    
    padding = "6%"
    for cbar in cbars:
        
    
        custom_cmap = z_to_cmap(cbar.z, display_function, scalar_output, cmap)
        
        if scalar_output and cbar.vmin is None and cbar.vmax is None:
            cbar.vmin = np.min(display_function(img))
            cbar.vmax = np.max(display_function(img))
            print(cbar.vmin, cbar.vmax)

        make_colorbar(cmap=custom_cmap, ax_img=ax_img, ax_divider=ax_divider, 
                      padding=padding, vmin=cbar.vmin, vmax=cbar.vmax,
                      ticks=cbar.ticks, ticklabels=cbar.ticklabels, 
                      label=cbar.label)
        padding = "15%"
    # make_colorbar(cmap, ax_img, ax_divider, padding, vmin=0, vmax=1, 
    #                   ticks=None, ticklabels=None, label=''):
    if not plt is None:
        plt.sca(ax)


# shorthand for complex_imshow:
cimshow = complex_imshow




#%%
if __name__ == '__main__':
    # simplest use case
    X, Y = np.meshgrid(np.arange(100) - 50, np.arange(100) - 50)
    img = X + 1j*Y
    
    cimshow(img, plt=plt)
    plt.title('simple use case')
    plt.show()
    # If you do plt.title after cimshow, the title shows above the colorbars
    
    # use axes instead of plt for more complicated figures
    fig, axes = plt.subplots(ncols=2)
    cimshow(img, ax=axes[0])
    axes[0].set_title('use ax instead of plt')
    cimshow(img, ax=axes[1], display_function=np.abs, abs_colorbar=True)
    axes[1].set_title('use ax instead of plt')
    plt.tight_layout()
    plt.show()
    
    # Default colorbars
    fig, ax = plt.subplots()
    plt.title('default colorbars')
    cimshow(img, ax=ax, abs_colorbar=True, phase_colorbar=True)
    plt.tight_layout()
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
    
    # Custom colorbar
    fig, axes = plt.subplots(ncols=2)
    cimshow(img, ax=axes[0], display_function=np.real, custom_cbars=[re_cbar])
    axes[0].set_title('Custom colorbar')
    
    im_cbar = CustomColorbar(
        z = np.linspace(0, 1j, num=200),
        label = 'im'
        )
    cimshow(img, ax=axes[1], display_function=np.imag, custom_cbars=[im_cbar])
    axes[1].set_title('')
    plt.tight_layout()
    plt.show()

