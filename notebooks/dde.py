# https://zulko.wordpress.com/2013/03/01/delay-differential-equations-easy-with-python/
# REQUIRES PACKAGES Numpy AND Scipy INSTALLED
import numpy as np
import scipy.integrate
import scipy.interpolate
  
class ddeVar:
    """ special function-like variables for the integration of DDEs """
     
     
    def __init__(self,g,tc=0):
        """ g(t) = expression of Y(t) for t<tc """
         
        self.g = g
        self.tc = tc
        # We must fill the interpolator with 2 points minimum
        self.itpr = scipy.interpolate.interp1d(
            np.array([tc-1,tc]), # X
            np.array([self.g(tc),self.g(tc)]).T, # Y
            kind='linear', bounds_error=False,
            fill_value = self.g(tc))
             
             
    def update(self,t,Y):
        """ Add one new (ti,yi) to the interpolator """
         
        self.itpr.x = np.hstack([self.itpr.x, [t]])
        Y2 = Y if (Y.size==1) else np.array([Y]).T
        self.itpr.y = np.hstack([self.itpr.y, Y2])
        self.itpr.fill_value = Y
         
         
    def __call__(self,t=0):
        """ Y(t) will return the instance's value at time t """
         
        return (self.g(t) if (t<=self.tc) else self.itpr(t))
  
 
 
class dde(scipy.integrate.ode):
    """ Overwrites a few functions of scipy.integrate.ode"""
     
     
    def __init__(self,f,jac=None):
        def f2(t,y,args):
            return f(self.Y,t,*args)
        scipy.integrate.ode.__init__(self,f2,jac)
        self.set_f_params(None)
         
  
    def integrate(self, t, step=0, relax=0):
        scipy.integrate.ode.integrate(self,t,step,relax)
        self.Y.update(self.t,self.y)
        return self.y
         
  
    def set_initial_value(self,Y):
        self.Y = Y #!!! Y will be modified during integration
        scipy.integrate.ode.set_initial_value(self, Y(Y.tc), Y.tc)
  
  
  
def ddeint(func, g, tt, fargs=None):
    """ similar to scipy.integrate.odeint. Solves the DDE system
        defined by func at the times tt with 'history function' g
        and potential additional arguments for the model, fargs
    """
     
    dde_ = dde(func)
    dde_.set_initial_value(ddeVar(g, tt[0]))
    dde_.set_f_params(fargs if fargs else [])
    # v = [g(tt[0])] + [
    v = [ 
        dde_.integrate(dde_.t + dt).ravel()
        for dt in np.diff(tt)
    ]
    # return np.array(v)
    return np.array([g(tt[0])] + [x[0] for x in v])