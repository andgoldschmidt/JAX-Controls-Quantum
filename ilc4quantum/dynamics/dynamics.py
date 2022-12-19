

def rk4(x, u, rhs, dt):
    """
    Model dynamics z' = F(x, u) using the Rung-Kutta 4 method.

    Notes:
        1.  Assumes no explicit time-dependence (TODO: Why do we have rhs(t,...)?)

    :param x: Initial state
    :param u: Initial control
    :param rhs: Function with signature (t: time, x: state, u: control) --> x_dot: state
    :param dt: Discrete time step
    :return: x[t+dt]
    """
    f1 = rhs(0, x, u)
    f2 = rhs(0, x + f1 * dt / 2, u)
    f3 = rhs(0, x + f2 * dt / 2, u)
    f4 = rhs(0, x + f3 * dt, u)
    return x + (f1 + 2 * f2 + 2 * f3 + f4) * dt / 6


def lift_model(discrete_model, n_state, n_ctrl):
    """
    Model dynamics z' = F(z) using `discrete_model`.
    """
    def f(z):
        x, u = z[:n_state], z[-n_ctrl:]
        return discrete_model(x, u)

    return f