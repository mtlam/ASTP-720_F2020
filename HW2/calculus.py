'''
Michael Lam
ASTP-720, Spring 2020

Set of functions to perform numerical calculus

There were multiple ways to answer these given
the intentional ambiguity in the problem.
I simply chose one way.

For example, one could have looked at sets of points (x, y) 
and interpolation as necessary. 

'''

def derivative(func, x, h):
    """
    Evaluate the derivative of a function
    at point x, with step size h
    Uses the symmetric derivative

    Parameters
    ----------
    func : function
        Function to evaluate
    x : float
        Point to evaluate derivative
    h : float 
        Step size


    Returns
    -------
    retval : float
        Derivative

    """
    return (func(x+h) - func(x-h))/(2*h)
    




def integrate(func, x_init, x_final, slices=100, mode="simpson"):
    """
    Integration via multiple methods

    Parameters
    ----------
    func : function
        Function to evaluate
    x_init : float
        Point to begin integration
    x_final : float 
        Point to end integration
    slices : int
        Number of iterations/slices to perform integration

    Returns
    -------
    integral : float
        Solution of the integral
    """

    h = (x_final - x_init)/float(slices)

    integral = 0.0
    if mode == "midpoint":
        for i in range(slices):
            x0 = x_init + i*h
            x1 = x0 + i*h
            xmid = (x0 + x1)/2.0
            integral += func(xmid)*h

    if mode == "trapezoid":
        for i in range(slices):
            x0 = x_init + i*h
            x1 = x0 + h
            y0 = func(x0)
            y1 = func(x1)
            integral += (y0 + y1)/2.0 * h
    if mode == "simpson":
        #enforce even number of slices out of laziness
        for i in range(0, slices, 2): 
            x0 = x_init + i*h
            x1 = x0 + h
            x2 = x1 + h
            y0 = func(x0)
            y1 = func(x1)
            y2 = func(x2)
            integral += (y0 + 4*y1 + y2) * (h/3)

    return integral



