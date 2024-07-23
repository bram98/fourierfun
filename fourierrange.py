import numpy as np
from math import isclose

__all__ = ['fft2', 'ifft2', 'XY', 'XY_c', 'FourierRange', 'FourierRange2d']

def fft2(img, input_c=True, output_c=True):
    if input_c:
        img_fft = np.fft.fftshift(img)
        
    img_fft = np.fft.fft2(img_fft)
    
    if output_c:
        img_fft = np.fft.fftshift(img_fft)
        
    return img_fft

def ifft2(img, input_c=True, output_c=True):
    if input_c:
        img_fft = np.fft.fftshift(img)
        
    img_fft = np.fft.ifft2(img_fft)
    
    if output_c:
        img_fft = np.fft.fftshift(img_fft)
        
    return img_fft

def XY(nx, ny, Lx=None, Ly=None):
    if Lx is None:
        Lx = nx
    if Ly is None:
        Ly = ny

    return FourierRange2d(nx, ny, xmax=Lx, ymax=Ly).XY()

def XY_c(nx, ny, Lx=None, Ly=None):
    if Lx is None:
        Lx = nx
    if Ly is None:
        Ly = ny
        
    return FourierRange2d(nx, ny, xmax=Lx, ymax=Ly).XY_c()

class FourierRange2d:
    def __init__(self, nx, ny, xmax=None, ymax=None, k_in_radians=False):
        self.k_in_radians = k_in_radians
        self.f_rangex = FourierRange(nx, xmax=xmax, k_in_radians=k_in_radians)
        self.f_rangey = FourierRange(ny, xmax=ymax, k_in_radians=k_in_radians)
        
    def from_kw(nx=None, ny=None, xmax=None, ymax=None, dx=None, dy=None, kxmax=None, kymax=None, dkx=None, dky=None, k_in_radians=False):
        frequency_unit = 2*np.pi if k_in_radians else 1
        
        xargs = {'n': nx, 'xmax': xmax, 'dx': dx, 'kmax': kxmax, 'dk': dkx}
        yargs = {'n': ny, 'xmax': ymax, 'dx': dy, 'kmax': kymax, 'dk': dky}
        
        xargs = {key:value for (key, value) in xargs.items() if not value is None}
        yargs = {key:value for (key, value) in yargs.items() if not value is None}
        
        xresult = FourierRange._get_xargs(xargs, frequency_unit=frequency_unit)
        yresult = FourierRange._get_xargs(yargs, frequency_unit=frequency_unit)
        return FourierRange2d(xresult['n'], yresult['n'], xmax=xresult['xmax'], ymax=yresult['xmax'])
    def from_ranges(rangex, rangey=None):
        if rangey is None:
            rangey = rangex
        return FourierRange2d(rangex.nx, rangey.nx, xmax=rangex.xmax, ymax=rangey.xmax)
    @property
    def nx(self):
        return self.f_rangex.n
    @property
    def ny(self):
        return self.f_rangey.n
    @property
    def xmax(self):
        return self.f_rangex.xmax
    @property
    def ymax(self):
        return self.f_rangey.xmax
    @property
    def dx(self):
        return self.f_rangex.dx
    @property
    def dy(self):
        return self.f_rangey.xmax
    @property
    def dkx(self):
        return self.f_rangex.dk
    @property
    def dky(self):
        return self.f_rangey.dk
    @property
    def kxmax(self):
        return self.f_rangex.kmax
    @property
    def kymax(self):
        return self.f_rangey.kmax
    
    def xrange(self):
        return self.f_rangex.xrange()
    
    def yrange(self):
        return self.f_rangey.xrange()
    
    def xrange_c(self):
        return self.f_rangex.xrange_c()
    
    def yrange_c(self):
        return self.f_rangey.xrange_c()
    
    def kxrange(self):
        return self.f_rangex.krange()
    
    def kyrange(self):
        return self.f_rangey.krange()
    
    def XY(self):
        return np.meshgrid(self.f_rangex.xrange(), self.f_rangey.xrange())
    
    def XY_c(self):
        return np.meshgrid(self.f_rangex.xrange_c(), self.f_rangey.xrange_c())
    
    def KxKy(self):
        return np.meshgrid(self.f_rangex.krange(), self.f_rangey.krange())
    
    def KxKy_c(self):
        return np.meshgrid(self.f_rangex.krange_c(), self.f_rangey.krange_c())
    
    def plot_extent(self):
        '''
        Helper function. Returns [0, xmax, 0, ymax - dx]. Is designed to be
        used with the FourierRange.XY() method and can be fed directly into the 
        extent argument of plt.imshow()
        '''
        return [*self.f_rangex.plot_extent(), *self.f_rangey.plot_extent()]
    
    def plot_extent_c(self):
        '''
        Helper function. Returns the equivalent of [X.min(), X.max(), Y.min(), Y.max()] 
        where X, Y are the result of FourierRange.XY_c() method. It can be fed directly into the 
        extent argument of plt.imshow()
        '''
        return [*self.f_rangex.plot_extent_c(), *self.f_rangey.plot_extent_c()]
    
    def plot_extent_fft(self):
        '''
        Helper function. Returns [0, xmax, 0, ymax - dx]. Is designed to be
        used with the FourierRange.XY() method and can be fed directly into the 
        extent argument of plt.imshow()
        '''
        return [*self.f_rangex.plot_extent_fft(), *self.f_rangey.plot_extent_fft()]
    
    def plot_extent_fft_c(self):
        '''
        Helper function. Returns the equivalent of [X.min(), X.max(), Y.min(), Y.max()] 
        where X, Y are the result of FourierRange.XY_c() method. It can be fed directly into the 
        extent argument of plt.imshow()
        '''
        return [*self.f_rangex.plot_extent_fft_c(), *self.f_rangey.plot_extent_fft_c()]
    
    def __repr__(self):
        return f'FourierRange2d({self.nx}, {self.ny}, xmax={self.xmax}, ymax={self.ymax}, k_in_radians={self.k_in_radians})'


class FourierRange:
    warnings = True
    
    def __init__(self, n: int, xmax=None, k_in_radians=False):
        n = FourierRange._round(n)
        self._n = n
        
        if xmax is None:
            xmax = n
        
        self._xmax = xmax
        
        self._dx = xmax/n
        
        self.k_in_radians = k_in_radians
        self._frequency_unit = 2*np.pi if k_in_radians else 1
        
        self._dk = self._frequency_unit/( xmax )
        
        self._kmax = self.dk*self.n
    
    def _round(n_f):
        n = int(np.round(n_f))
        if FourierRange.warnings and not isclose(n, n_f):
            print('Warning: n provided was not an integer and is therefore rounded')
        return n
    def _get_xargs(xargs: dict, frequency_unit=1.):
        if 'n' in xargs:
            n = FourierRange._round( xargs['n'] )
            if 'xmax' in xargs:
                return xargs
            elif 'dx' in xargs:
                dx = xargs['dx']
                return {'n': n, 'xmax': n*dx}
            elif 'kmax' in xargs:
                kmax = xargs['kmax']
                return {'n': n, 'xmax': frequency_unit*n/kmax}
            elif 'dk' in xargs:
                dk = xargs['dk']
                return {'n': n, 'xmax': frequency_unit/dk}
            else:
                raise Exception(f'Error while constructing FourierRange from keywords. Not enough values: {xargs.items()}')
        
        elif 'xmax' in xargs:
            xmax = xargs['xmax']
            if 'dx' in xargs:
                dx = xargs['dx']
                n = FourierRange._round( xmax/dx )
                return {'n': n, 'xmax': xmax}
            elif 'kmax' in xargs:
                kmax = xargs['kmax']
                n = FourierRange._round( frequency_unit*xmax*kmax )
                return {'n': n, 'xmax': frequency_unit*n/kmax}
            elif 'dk' in xargs:
                raise Exception('Error while constructing from keywords. xmax and dk are inconsistent variables')
            else:
                raise Exception(f'Error while constructing FourierRange from keywords. Not enough values: {xargs.values()}')
        elif 'dx' in xargs:
            dx = xargs['dx']
            if 'kmax' in xargs:
                raise Exception('Error while constructing from keywords. kmax and dx are inconsistent variables')
            elif 'dk' in xargs:
                dk = xargs['dk']
                n = FourierRange._round( frequency_unit/( dx*dk ) )
                return {'n': n, 'xmax': n*dx}
            else:
                raise Exception(f'Error while constructing FourierRange from keywords. Not enough values: {xargs.items()}')
        elif 'kmax' in xargs:
            kmax = xargs['kmax']
            if 'dk' in  xargs:
                dk = xargs['dk']
                n = FourierRange._round(kmax/dk)
                return {'n': n, 'xmax': frequency_unit/dk}
            else:
                raise Exception(f'Error while constructing FourierRange from keywords. Not enough values: {xargs.items()}')
        else:
            raise Exception(f'Error while constructing FourierRange from keywords. Not enough values: {xargs.items()}')
            
    def from_kw(n=None, xmax=None, dx=None, kmax=None, dk=None, k_in_radians=False):
        frequency_unit = 2*np.pi if k_in_radians else 1
        
        xargs = {'n': n, 'xmax': xmax, 'dx': dx, 'kmax': kmax, 'dk': dk}
        xargs = {key:value for (key, value) in xargs.items() if not value is None}
        
        d = FourierRange._get_xargs(xargs, frequency_unit)
        return FourierRange(d['n'], xmax=d['xmax'], k_in_radians=k_in_radians)
        
    @property
    def n(self):
        return self._n
    
    @property
    def xmax(self):
        return self._xmax
    
    @property
    def dx(self):
        return self._dx
    
    @property
    def dk(self):
        return self._dk
    
    @property
    def kmax(self):
        return self._kmax
    
    def xrange(self):
        return np.linspace(0, self.xmax, endpoint=False, num=self.n)
    
    def xrange_c(self):
        if self.n%2 == 0:
            # even
            return np.linspace( -self.xmax/2, self.xmax/2, endpoint=False, num=self.n)
        else:
            # odd
            return np.linspace( -self.xmax/2 + self.dx/2, self.xmax/2 + self.dx/2, endpoint=False, num=self.n)
        
    def krange(self):
        return np.linspace(0, self.kmax, endpoint=False, num=self.n)
    
    def krange_c(self):
        if self.n%2 == 0:
            # even
            return np.linspace( -self.kmax/2, self.kmax/2, endpoint=False, num=self.n)
        else:
            # odd
            return np.linspace( -self.kmax/2 + self.dk/2, self.kmax/2 + self.dk/2, endpoint=False, num=self.n)
    
    def plot_extent(self):
        '''
        Helper function. Returns first and last item of .xrange()
        '''
        return [0, self.xmax - self.dx]
    
    def plot_extent_c(self):
        '''
        Helper function. Returns first and last item of .xrange_c()
        '''
        if self.n%2 == 0:
            # even
            return [ -self.xmax/2, self.xmax/2 - self.dx]
        else:
            # odd
            return [ -self.xmax/2 + self.dx/2, self.xmax/2 - self.dx/2]
    
    
    def plot_extent_fft(self):
        '''
        Helper function. Returns first and last item of .xrange()
        '''
        return [0, self.kmax - self.dk]
    
    def plot_extent_fft_c(self):
        '''
        Helper function. Returns first and last item of .xrange_c()
        '''
        if self.n%2 == 0:
            # even
            return [ -self.kmax/2, self.kmax/2 - self.dk]
        else:
            # odd
            return [ -self.kmax/2 + self.dk/2, self.kmax/2 - self.dk/2]
    
    def __repr__(self):
        return f'FourierRange({self.n}, xmax={self.xmax}, k_in_radians={self.k_in_radians})'