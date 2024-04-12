#%%
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from colorsys import hls_to_rgb
from colorsys import hsv_to_rgb
from matplotlib.colors import ListedColormap
from dataclasses import dataclass






def gaussian(X, Y, sigma=5):
    return np.exp( -(X**2 + Y**2)/(2*sigma**2))

def LG10(X, Y, w0):
    R2 = X**2 + Y**2
    R = np.sqrt(R2)
    phi = np.arctan2(Y, X)
    return R*2/(w0*np.sqrt(np.pi))*np.exp( -R2/w0**2 - 1j*phi )
    # return phi

def smiley_phase(X, Y, diameter=.45):
    
    # ymax, xmax = X.shape
    xmax = X[0,-1] - X[0,0]
    ymax = Y[-1,0] - Y[0,0]
    D = min(xmax, ymax)*diameter
    # Head
    head = ( X  )**2 + ( Y  )**2 < (.45*D)**2
    img = head.astype(complex)

    # Mouth
    mouth = ( X  )**2 + ( Y  )**2 < (.35*D)**2
    np.logical_and(mouth,  Y > 0, out=mouth)
    mouth = mouth.astype(complex)
    img += mouth*(-1 -1)

    # Eyes
    left = np.logical_and(
        np.abs(X  + D*.17) < .031*D,
        np.abs(Y  + D*.18) < .13*D
        )
    left = left.astype(complex)
    img += left*(-1 - 1j)

    right = np.logical_and(
        np.abs(X  - D*.17) < .031*D,
        np.abs(Y  + D*.18) < .13*D
        )
    right = right.astype(complex)
    img += right*(-1 + 1j)
    return img



def add_colorbar(plt, axes_image: matplotlib.image.AxesImage, img_shape: tuple[int, int], pad=.04, cmap='viridis'):
    aspect = img_shape[0]/img_shape[1]
    print(cmap)
    plt.colorbar(axes_image, ax=axes_image.axes, fraction=0.046*aspect, pad=pad, cmap=cmap)

def plot_complex(img: np.ndarray, ax=None, display_function=np.real, 
                 display_fft_function=None, colorbar=True, dpi=200, **plotargs):
    if ax is None:
        fig, ax = plt.subplots(dpi=dpi)
    img_display = display_function(img)
    print(ax)
    ax_img = ax.imshow(img_display, **plotargs)
    # phase_c = 
    abs_colors = display_function( [ np.linspace( np.min(img_display), np.max(img_display) ) ] )[0]
    phase_colors = display_function( [ np.exp(1j*np.linspace(0, 2*np.pi)) ] )[0]
    display_phase = True
    if len(phase_colors.shape)==1:

        abs_cmap = 'viridis'
        # print(display_function, phase_colors)
        # phase_colors -= np.min(phase_colors)
        # if not isclose(np.max(phase_colors), 0):
        #     phase_colors/= np.max(phase_colors)
        # phase_colors = matplotlib.colormaps['viridis'](phase_colors)
        # phase_cmap = ListedColormap(phase_colors)
        display_phase = False
    elif 3 <= phase_colors.shape[1] <= 4:
        abs_cmap = ListedColormap(abs_colors)
        phase_cmap = ListedColormap(phase_colors)
    else:
        pass #IDK what will happen here, probably error
    if colorbar:
        # add_colorbar(plt, ax_img, img.shape, pad=.1, cmap=phase_cmap)
        # add_colorbar(plt, ax_img, img.shape)
        # cmap = matplotlib.cm.cool
        abs_norm = matplotlib.colors.Normalize(vmin=np.min(img_display), vmax=np.max(img_display))
        if display_phase: 
            phase_norm = matplotlib.colors.Normalize(vmin=0, vmax=2*np.pi)
            plt.colorbar(matplotlib.cm.ScalarMappable(norm=phase_norm, cmap=phase_cmap),
                  ax=ax,
                  label='phase'
                  )

        plt.colorbar(matplotlib.cm.ScalarMappable(norm=abs_norm, cmap=abs_cmap),
              ax=ax,
              label='norm'
              )

    
    plt.tight_layout()
    plt.show()

def plot_with_fft(img: np.ndarray, display_function=np.real, fft=False, 
                 display_fft_function=None, colorbar=True, dpi=200, **plotargs):
    fig, (ax_real, ax_fft) = plt.subplots(ncols=2, dpi=dpi)
        
    ax_img_real = ax_real.imshow(display_function(img), **plotargs)
    # ax_img_real = plot_complex(img, ax=ax_real, colorbar=False)

    img_fft = np.fft.fftshift(img)
    img_fft = np.fft.fft2(img_fft)
    img_fft = np.fft.fftshift(img_fft)
    if 'extent' in plotargs:
        extent = plotargs['extent']
        Lx = extent[1] - extent[0]
        Ly = extent[3] - extent[2]
        dx = Lx/(img.shape[1] - 1)
        dy = Ly/(img.shape[0] - 1)
        # dkx = 1/Lx
        kxmax = 1/dx
        kymax = 1/dy
        print(Lx, kxmax/2, img.shape[1])
        plotargs['extent'] = [-kxmax/2, kxmax/2, -kymax/2, kymax/2]
    
    if display_fft_function is None:
        display_fft_function = display_function
    ax_img_fft = ax_fft.imshow(display_fft_function(img_fft), **plotargs)
    # ax_img_fft = plot_complex(img_fft, ax=ax_fft, display_function=display_function)

def plot_complex_(img: np.ndarray, display_function=np.real, fft=False, 
                 display_fft_function=None, colorbar=True, dpi=200, **plotargs):
    if fft:
        fig, (ax_real, ax_fft) = plt.subplots(ncols=2, dpi=200)
    else:
        fig, ax_real = plt.subplots(dpi=200)
        
    ax_img_real = ax_real.imshow(display_function(img), **plotargs)
    
    
    if fft:
        img_fft = np.fft.fftshift(img)
        img_fft = np.fft.fft2(img_fft)
        img_fft = np.fft.fftshift(img_fft)
        if 'extent' in plotargs:
            extent = plotargs['extent']
            Lx = extent[1] - extent[0]
            Ly = extent[3] - extent[2]
            dx = Lx/(img.shape[1] - 1)
            dy = Ly/(img.shape[0] - 1)
            # dkx = 1/Lx
            kxmax = 1/dx
            kymax = 1/dy
            print(Lx, kxmax/2, img.shape[1])
            plotargs['extent'] = [-kxmax/2, kxmax/2, -kymax/2, kymax/2]
        
        if display_fft_function is None:
            display_fft_function = display_function
            
        ax_img_fft = ax_fft.imshow(display_fft_function(img_fft), **plotargs)
    
    if colorbar:
        add_colorbar(plt, ax_img_real, img.shape)
        if fft:
            add_colorbar(plt, ax_img_fft, img_fft.shape)
    
    plt.tight_layout()
    plt.show()

def normalize_arr(v, vmin=0, vmax=1, const_value=1):
    vmin0, vmax0 = np.min(v), np.max(v)
    if isclose(vmin0, vmax0):
        v += const_value - vmin0
    else:
        v -= vmin0
        v *= (vmax - vmin)/np.max(v)
        v += vmin
    return v

def colorize(z, normalize=False):
    r = np.abs(z)
    arg = np.angle(z) 
    
    
    if normalize:
        r = normalize_arr(r, vmin=0, vmax=1.5, const_value=1)
    else:
        r = np.clip(r, 0, 1)

    h = (arg + np.pi)  / (2 * np.pi) + 0.5
    v = r
    # s = 1
    s = np.min([np.ones_like(r), 2 - r], axis=0)
    
    v = np.clip(v, 0, 1)
    
    c = np.vectorize(hsv_to_rgb) (h,s,v) # --> tuple
    c = np.array(c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    c = c.transpose(1,2,0) 
    return c


    


