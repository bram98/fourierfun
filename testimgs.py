import numpy as np

def gaussian(X, Y, sigma=5):
    return np.exp( -(X**2 + Y**2)/(2*sigma**2))

def gaussian_vortex(X, Y, w0):
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

def smiley_amplitude(X, Y, diameter=.45):
    
    # ymax, xmax = X.shape
    xmax = X[0,-1] - X[0,0]
    ymax = Y[-1,0] - Y[0,0]
    D = min(xmax, ymax)*diameter
    # Head
    head = ( X  )**2 + ( Y  )**2 < (.45*D)**2
    img = head.astype(float)

    # Mouth
    mouth = ( X  )**2 + ( Y  )**2 < (.35*D)**2
    np.logical_and(mouth,  Y > 0, out=mouth)
    mouth = mouth.astype(float)
    img -= mouth

    # Eyes
    left = np.logical_and(
        np.abs(X  + D*.17) < .031*D,
        np.abs(Y  + D*.18) < .13*D
        )
    left = left.astype(float)
    img -= left

    right = np.logical_and(
        np.abs(X  - D*.17) < .031*D,
        np.abs(Y  + D*.18) < .13*D
        )
    right = right.astype(float)
    img -= right
    return img