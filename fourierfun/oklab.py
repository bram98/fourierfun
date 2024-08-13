'''
https://bottosson.github.io/posts/oklab/
'''
#%%
import numpy as np
import matplotlib.pyplot as plt

__all__ = ['matmul', 'list_to_img', 'img_to_list', 'float_to_u8', 
           'u8_to_float', 'linrgb_to_rgb', 'rgb_to_linrgb', 'rgb_to_oklab',
           'oklab_to_rgb', 'oklab_to_oklch', 'oklch_to_oklab', 'oklch_to_rgb']
# %%
M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
               [0.2119034982, 0.6806995451, 0.1073969566],
               [0.0883024619, 0.2817188376, 0.6299787005]])

M2 = np.array([[0.2104542553, +0.7936177850, -0.0040720468],
               [1.9779984951, -2.4285922050, +0.4505937099],
               [0.0259040371, +0.7827717662, -0.8086757660]])

M2inv = np.array([[1., +0.3963377774, +0.2158037573],
                  [1., -0.1055613458, -0.0638541728],
                  [1., -0.0894841775, -1.2914855480]])

M1inv = np.array([[+4.0767416621, -3.3077115913, +0.2309699292],
                  [-1.2684380046, +2.6097574011, -0.3413193965],
                  [-0.0041960863, -0.7034186147, +1.7076147010]])

def matmul(M, v):
    '''
    multiplies NxN matrix by array of vectors of shape (:, :, ... , N)
    '''
    v_ = np.expand_dims(v, axis=-1)
    return np.matmul(M, v_).squeeze(-1) 

def list_to_img(col_list):
    col_list = np.array(col_list)
    axes = np.arange(len(col_list.shape))
    return np.transpose(np.array(col_list), (*axes[1:], 0))

def img_to_list(img):
    axes = np.arange(len(img.shape))
    return np.transpose(np.array(img), (-1, *axes[:-1]))

def float_to_u8(f, clip=False):
    if clip:
        f = np.clip(f, 0, 1)
    return np.round(f*255).astype(np.uint8)

def u8_to_float(f):
    return f.astype(float)/255.

def linrgb_to_rgb(rgb, clip=False):
    small_values = rgb < 0.0031308
    rgb = (1 - small_values)*( (1.055) * rgb**(1.0/2.4) - 0.055 ) + \
            (small_values)*rgb*12.92
    if clip:
        rgb = np.clip(rgb, 0, 1)
    return rgb

def rgb_to_linrgb(rgb, clip=False):
    small_values = rgb < 0.04045
    rgb = (1 - small_values)*(((rgb + 0.055)/(1 + 0.055))**2.4 ) + \
            (small_values)*rgb/12.92
    return rgb

def rgb_to_oklab(rgb, clip=False):
    linrgb = rgb_to_linrgb(rgb, clip=False)
    # linrgb = linrgb_to_rgb(rgb, clip=False)
    # matrix multiply over last axis
    lms = matmul(M1, linrgb)
    lms_ = np.cbrt(lms)
    oklab = matmul(M2, lms_)
    if clip:
        oklab = np.clip(oklab, 0, 1)
    return oklab

def oklab_to_rgb(lab, clip=False):
    lms_ = matmul(M2inv, lab)
    lms = np.power(lms_, 3)
    linrgb = matmul(M1inv, lms)
    rgb = linrgb_to_rgb(linrgb, clip=False)
    # rgb = rgb_to_linrgb(linrgb, clip=False)
    if clip:
        rgb = np.clip(rgb, 0, 1)
    return rgb

# def rgb_to_oklab2(rgb, clip=False):
#     lms = M1@rgb
#     lms_ = np.cbrt(lms)
#     oklab = M2@lms_
#     if clip:
#         oklab = np.clip(oklab, 0, 1)
#     return oklab

def oklab_to_oklch(lab, clip=False):
    L = lab[..., 0]
    a = lab[..., 1]
    b = lab[..., 2]
    C = np.sqrt(a**2 + b**2)
    h = np.arctan2(b, a)
    LCh = np.array([L, C, h])
    LCh = list_to_img(LCh)
    return LCh

def oklch_to_oklab(lch, clip=False):
    L = lch[..., 0]
    C = lch[..., 1]
    h = lch[..., 2]
    h *= 2*np.pi
    # print(h.shape)
    a = C*np.cos(h)
    b = C*np.sin(h)
    Lab = np.array([L, a, b])
    Lab = list_to_img(Lab)
    return Lab

def oklch_to_rgb(lch, clip=False):
    return oklab_to_rgb(oklch_to_oklab(lch), clip=clip)


