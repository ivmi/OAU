import numpy as np
#import control
from scipy.integrate import ode
#from scipy.signal import tf2ss

def simulate(blocks, connections, outputs, t_final):
    def f_u(iblock, t, x):
        x_split = np.array_split(x, x_inds)
        def u(i=0):
            isrcblock = conns.get((iblock, i))
            if isrcblock:
                ub = f_u(isrcblock[0], t, x)
                
                return blocks[isrcblock[0]][1](t, x_split[isrcblock[0]], ub)
            else:
                return 0.0
        
        return u
    
    def deriv(t, x):
        dx = []
        
        x_split = np.array_split(x, x_inds)
        
        for ib, block in enumerate(blocks):
            if block[0] is None:
                continue
            
            xb = x_split[ib]
            
            u = f_u(ib, t, x)
            dxb = block[0](t, xb, u)
            dx.extend(dxb)
            
        return dx

    def solout(t, x):
        nonlocal t_arr, y_arr
        t_arr = np.hstack((t_arr, t))
        x_split = np.array_split(x, x_inds)
        
        yv = []
        for ib in outputs:
            u = f_u(ib, t, x)
            y = blocks[ib][1](t, x_split[ib], u)
            yv.append(y)
        
        y_arr = np.vstack([y_arr, yv]) if y_arr.size else np.array(yv)

    t_arr = np.array([])
    y_arr = np.matrix([])
        
    conns = {c[1]: c[0] for c in connections}
    r = ode(deriv)
    r.set_integrator('dopri5', rtol=1e-9, atol=1e-12, nsteps=5000)
         

    x_inds = []
    acc=0
    for b in blocks:
        x_inds.append(acc+len(b[2]))
        acc+=len(b[2])
    x_inds = x_inds[0:-1]
        
    x0 = [x0 for block in blocks for x0 in block[2]]
    #print(x0)
    #print([b[2] for b in blocks])
    #print(np.array_split(x0, x_inds))
    r.set_initial_value(x0)
    r.set_solout(solout)
    r.integrate(t_final)
    return (t_arr, y_arr)    
