import matplotlib.pyplot as plt
from scipy.integrate import ode
import numpy as np

def resp_plot(t, y, label='', legend=[]):
    fig = plt.figure()
    fig.set_label(label)

    plt.plot(t, y)
    plt.grid()
    plt.legend(legend)
    return plt
    #plt.show()
    
def f_ode(d_model, x0, u, t_final):
    def deriv_u(t, x):
        return d_model(t, u(t), *x)
    
    def deriv_u_const(t, x):
        return d_model(t, u, *x)
    
    t_arr = np.array([])
    y_arr = np.matrix([])
    
    def solout(t, y):
        nonlocal t_arr, y_arr
        t_arr = np.hstack((t_arr, t))
        y_arr = np.vstack([y_arr, y]) if y_arr.size else y   
    
    if u is None:
        u = 0.0
    if callable(u):
        der = deriv_u
    else:
        der = deriv_u_const
    
    r = ode(der)
    r.set_integrator('dopri5', rtol=1e-9, atol=1e-12, nsteps=5000)
    r.set_initial_value(x0)
    r.set_solout(solout)
    r.integrate(t_final)
    return (t_arr, y_arr)

def step_signal(t_step, y0, y1):
    def f_u(t):
        if t<t_step:
            return y0
        else:
            return y1
    return f_u

def sine_signal(A, w, fi):
    def f_u(t):
        return A*math.sin(w*t+fi)
    return f_u

def mux(signals):
    def f_u(t):
        return [signal(t) for signal in signals]
    return f_u
