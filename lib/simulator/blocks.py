import numpy as np
#import control
from scipy.integrate import ode
from scipy.signal import tf2ss

def integrator(Ti=1.0, x0=0.0):
    def deriv(t, x, u):
        return [u(0)/Ti]
    def out(t, x, u):
        return x[0]

    return (deriv, out, [x0])

def suma(signs):
    k_signs = [-1 if sign=='-' else 1 for sign in signs]
    
    def out(t, x, u):
        res = 0
        
        for i, sign in enumerate(signs):
            res += k_signs[i]*u(i)

        return res
    

    return (None, out, [])

def saturate(u_min, u_max):
    def out(t, x, u):
        u0 = u(0)
        return u_min if u0 < u_min else u_max if u0 > u_max else u0

    return (None, out, [])

def gain(K=1.0):
    def out(t, x, u):
        return K*u(0)
        #return [K*uk for uk in u]

    return (None, out, [])

def step(y0, y1, t_step):
    def out(t, x, u):
        if t<t_step:
            return y0
        else:
            return y1

    return (None, out, [])

def stairs(v_t, v_y):
    def out(t, x, u):
        i = next(x for x, val in enumerate(v_t) if val > t) 
        return v_y[i-1] # TODO: check limit values
        
    return (None, out, [])


def sine(A, w, fi):
    def out(t, x, u):
        return A*math.sin(w*t+fi)

    return (None, out, [])

def sec_order(wn, zeta, x0):
    def deriv(t, x, u):
        dx = [x[1], u(0)*wn**2 - 2*wn*zeta*x[1] - wn**2*x[0]]
        return dx
    def out(t, x, u):
        return x[0]
    return (deriv, out, x0)
    
def tf(wn, zeta, x0):
    def deriv(t, x, u):
        dx = [x[1], u(0)*wn**2 - 2*wn*zeta*x[1] - wn**2*x[0]]
        return dx
    def out(t, x, u):
        return x[0]
    return (deriv, out, x0)    

def ss(A, B, C, D, x0):
    A = np.matrix(A)
    B = np.matrix(B)
    C = np.matrix(C)
    D = np.matrix(D)
    
    n_u = B.shape[1]
    u_mask = D.any(axis=0).A1
    
    #print(u_mask)
    
    
    def deriv(t, x, u):
        uv = [u(i) for i in range(n_u)]
        uvt = np.array(uv).transpose()
        
        #print("der---")
        #print(A)
        #print(x)
        #print(B)
        #print(uvt)
        
        dx = A@x + B@uvt
        return dx.A1
    def out(t, x, u):
        uv = [u(i) if u_mask[i] else 0.0 for i in range(n_u)]
        uvt = np.array(uv).transpose()
        res = C@x+D@uvt
        #print("out----")
        #print(C)
        #print(x)
        #print(D)
        #print(uvt)
        #print(res.A1)
        
        return np.asscalar(res)
    return (deriv, out, x0)

def tf(num, den):
    A, B, C, D = tf2ss(num, den)
    
    return ss(A, B, C, D, np.zeros(A.shape[0]))
