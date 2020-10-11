"""
Michael Lam
ASTP-720, Fall 2020

Set of functions to perform ODE solving
The assumption is that either one or two variables can be solved
for simultaneously. Therefore, rather than making a generic solver
like scipy's odeint(), I have intentionally constructed these to
take a set of functions, dy_A/dt and and y_{A,0}, and also allow
for optional arguments dy_B/dt and y_{B,0}.

By design, many of these functions are similar internally. However,
I will write them out as separate functions explicitly, rather
than trying to combine them all into a single function with
internal logic.
"""

import numpy as np


def euler(ts, dyAdt, yA0, dyBdt=None, yB0=None):
    """
    Forward Euler method for solving ODEs of the form
    y' = f(t, y)
    dyBdt and yB0 are given, then it will solve a set
    of coupled ODEs of the form
    y_A' = f_A(t, y_A, y_B)
    y_B' = f_B(t, y_A, y_B)
    as denoted in class.

    Parameters
    ----------
    ts : list, np.ndarray
        Times to solve the functions y(t) at
    dyAdt : function
        Derivative function of y_A. It must take two arguments, t and y_A,
        unless the y_B options are given below, in which case it must
        take three: t, y_A, and y_B, in that order.
    yA0 : float
        Initial value for y_A(0) = y_{A,0}
    dyBdt (optional) : function
        Derivative function of y_B. It must take three arguments:
        t, y_A, and y_B, in that order.
    yB0 (optional) : float
        Initial value for y_B(0) = y_{B,0}


    Returns
    -------
    yAs : np.ndarray
        Timeseries of y_A(t)
    yBs : np.ndarray
        If dyBdt and yB0 are not None, then this will also return
        the timeseries of y_b(t)
    """

    # Set up return array(s)
    yAs = np.zeros_like(ts)
    yAs[0] = yA0
    two_eqns = False #easier to read flag for whether we have two equations or not
    if dyBdt is not None and yB0 is not None:
        two_eqns = True
        yBs = np.zeros_like(ts)
        yBs[0] = yB0

    for i, t in enumerate(ts[:-1]): #do not go to the last timestep
        h = ts[i+1] - t
        if two_eqns:
            yAs[i+1] = yAs[i] + h*dyAdt(t, yAs[i], yBs[i])
            yBs[i+1] = yBs[i] + h*dyBdt(t, yAs[i], yBs[i])
        else:
            yAs[i+1] = yAs[i] + h*dyAdt(t, yAs[i])

    if two_eqns:
        return yAs, yBs
    else:
        return yAs


def heun(ts, dyAdt, yA0, dyBdt=None, yB0=None, niter=10):
    """
    Heun's method for solving ODEs of the form
    y' = f(t, y)
    dyBdt and yB0 are given, then it will solve a set
    of coupled ODEs of the form
    y_A' = f_A(t, y_A, y_B)
    y_B' = f_B(t, y_A, y_B)
    as denoted in class.

    The Picard iterations work on a fixed number, rather
    than some threshold.

    Parameters
    ----------
    ts : list, np.ndarray
        Times to solve the functions y(t) at
    dyAdt : function
        Derivative function of y_A. It must take two arguments, t and y_A,
        unless the y_B options are given below, in which case it must
        take three: t, y_A, and y_B, in that order.
    yA0 : float
        Initial value for y_A(0) = y_{A,0}
    dyBdt (optional) : function
        Derivative function of y_B. It must take three arguments:
        t, y_A, and y_B, in that order.
    yB0 (optional) : float
        Initial value for y_B(0) = y_{B,0}
    niter (optional) : int
        Number of Picard iterations/corrector steps to perform.


    Returns
    -------
    yAs : np.ndarray
        Timeseries of y_A(t)
    yBs : np.ndarray
        If dyBdt and yB0 are not None, then this will also return
        the timeseries of y_b(t)
    """

    # Set up return array(s)
    yAs = np.zeros_like(ts)
    yAs[0] = yA0
    two_eqns = False #easier to read flag for whether we have two equations or not
    if dyBdt is not None and yB0 is not None:
        two_eqns = True
        yBs = np.zeros_like(ts)
        yBs[0] = yB0

    for i, t in enumerate(ts[:-1]): #do not go to the last timestep
        h = ts[i+1] - t
        if two_eqns:
            f_A_i = dyAdt(t, yAs[i], yBs[i])
            f_B_i = dyBdt(t, yAs[i], yBs[i])
            yAs[i+1] = yAs[i] + h*f_A_i
            yBs[i+1] = yBs[i] + h*f_B_i
            # Now perform Picard iteration
            for k in range(niter):
                # At the i+1 time step, replace the next k+1
                # iteration with the previous.
                yAs[i+1] = yAs[i] + 0.5*h*(f_A_i + dyAdt(t, yAs[i+1], yBs[i+1]))
                yBs[i+1] = yBs[i] + 0.5*h*(f_B_i + dyBdt(t, yAs[i+1], yBs[i+1]))
        else:
            f_i = dyAdt(t, yAs[i])
            yAs[i+1] = yAs[i] + h*f_i
            # Now perform Picard iteration
            for k in range(niter):
                # At the i+1 time step, replace the next k+1
                # iteration with the previous.
                yAs[i+1] = yAs[i] + 0.5*h*(f_i + dyAdt(t, yAs[i+1]))

    if two_eqns:
        return yAs, yBs
    else:
        return yAs


def RK4(ts, dyAdt, yA0, dyBdt=None, yB0=None):
    """
    Classical Runge-Kutta method for solving ODEs of the form
    y' = f(t, y)
    dyBdt and yB0 are given, then it will solve a set
    of coupled ODEs of the form
    y_A' = f_A(t, y_A, y_B)
    y_B' = f_B(t, y_A, y_B)
    as denoted in class.

    Parameters
    ----------
    ts : list, np.ndarray
        Times to solve the functions y(t) at
    dyAdt : function
        Derivative function of y_A. It must take two arguments, t and y_A,
        unless the y_B options are given below, in which case it must
        take three: t, y_A, and y_B, in that order.
    yA0 : float
        Initial value for y_A(0) = y_{A,0}
    dyBdt (optional) : function
        Derivative function of y_B. It must take three arguments:
        t, y_A, and y_B, in that order.
    yB0 (optional) : float
        Initial value for y_B(0) = y_{B,0}


    Returns
    -------
    yAs : np.ndarray
        Timeseries of y_A(t)
    yBs : np.ndarray
        If dyBdt and yB0 are not None, then this will also return
        the timeseries of y_b(t)
    """

    # Set up return array(s)
    yAs = np.zeros_like(ts)
    yAs[0] = yA0
    two_eqns = False #easier to read flag for whether we have two equations or not
    if dyBdt is not None and yB0 is not None:
        two_eqns = True
        yBs = np.zeros_like(ts)
        yBs[0] = yB0

    for i, t in enumerate(ts[:-1]): #do not go to the last timestep
        h = ts[i+1] - t
        if two_eqns:
            # Evaluate each set of k values one after another
            # for both y_A and y_B
            kA1 = h*dyAdt(t, yAs[i], yBs[i])
            kB1 = h*dyBdt(t, yAs[i], yBs[i])
            kA2 = h*dyAdt(t + h/2.0, yAs[i] + kA1/2.0, yBs[i] + kB1/2.0)
            kB2 = h*dyBdt(t + h/2.0, yAs[i] + kA1/2.0, yBs[i] + kB1/2.0)
            kA3 = h*dyAdt(t + h/2.0, yAs[i] + kA2/2.0, yBs[i] + kB2/2.0)
            kB3 = h*dyBdt(t + h/2.0, yAs[i] + kA2/2.0, yBs[i] + kB2/2.0)
            kA4 = h*dyAdt(t + h, yAs[i] + kA3, yBs[i] + kB3)
            kB4 = h*dyBdt(t + h, yAs[i] + kA3, yBs[i] + kB3)
            yAs[i+1] = yAs[i] + (kA1 + 2*kA2 + 2*kA3 + kA4)/6.0
            yBs[i+1] = yBs[i] + (kB1 + 2*kB2 + 2*kB3 + kB4)/6.0

        else:
            k1 = h*dyAdt(t, yAs[i])
            k2 = h*dyAdt(t + h/2.0, yAs[i] + k1/2.0)
            k3 = h*dyAdt(t + h/2.0, yAs[i] + k2/2.0)
            k4 = h*dyAdt(t + h, yAs[i] + k3)
            yAs[i+1] = yAs[i] + (k1 + 2*k2 + 2*k3 + k4)/6.0

    if two_eqns:
        return yAs, yBs
    else:
        return yAs
