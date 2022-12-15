

# Model dynamics z' = F(z)
def rk4(z, n_state, n_ctrl, rhs, dt=1):
    """ Rung-Kutta 4 method """
    # assert len(z) == n_state + n_ctrl
    x, u = z[:n_state], z[-n_ctrl:]
    f1 = rhs(0, x, u)
    f2 = rhs(0, x + f1 * dt / 2, u)
    f3 = rhs(0, x + f2 * dt / 2, u)
    f4 = rhs(0, x + f3 * dt, u)
    return x + (f1 + 2 * f2 + 2 * f3 + f4) * dt / 6