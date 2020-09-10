"""
Michael Lam
ASTP-720, Spring 2020

Set of functions to perform root finding
"""

# Default threshold
THRESHOLD = 0.0000001





def bisect(f, a, b, threshold=THRESHOLD, full_output=False):
    """
    Bisection method for root finding

    Parameters
    ----------
    f : function
        Mathematical function to find root
    a : float
        Lower bound of search range
    b : float
        Upper bound of search range
    threshold: float, optional
        Threshold to stop iterating
    full_output: bool, optional
        Non-zero to return all optional outputs

    Returns
    -------
    c : float
        Value of root
    numiter : int
        Number of iterations, if full_output == True


    Raises
    ------
    ValueError
        If `a` equals `b`
        If f(a) and f(b) have the same sign

    """

    # Initial function checking
    if a == b:
        raise ValueError("a == b, search range equals zero")
    if f(a)*f(b) > 0:
        raise ValueError("f(a) and f(b) have the same sign")
    if a > b:
        a, b = b, a

    numiter = 0

    # Already at the root
    if abs(f(a)) <= threshold:
        if full_output:
            return a, numiter
        return a
    if abs(f(b)) <= threshold:
        if full_output:
            return b, numiter
        return b

    while True:
        c = (b+a)/2.0
        if f(c) == 0 or abs(f(c)) <= threshold:
            if full_output:
                return c, numiter
            return c
        elif f(a)*f(c) < 0:
            b = c
        else: #f(c)*f(b) < 0
            a = c
        numiter += 1


def newton(f, fprime, x_0, threshold=THRESHOLD, full_output=False):
    """
    Newton's method for root finding

    Parameters
    ----------
    f : function
        Mathematical function to find root
    fprime: function
        Derivative of f
    x_0 : float
        Initial guess
    threshold: float, optional
        Threshold to stop iterating
    full_output: bool, optional
        Non-zero to return all optional outputs

    Returns
    -------
    x_n : float
        Value of root
    numiter : int
        Number of iterations, if full_output == True

    Raises
    ------
    ValueError
        If f'(x_n) == 0

    """
    numiter = 0
    x_n = x_0
    while True:
        y = f(x_n)
        if abs(y) <= threshold:
            if full_output:
                return x_n, numiter
            return x_n
        yprime = fprime(x_n)
        if yprime == 0:
            raise ValueError("Value at derivative is zero")
        x_n = x_n - y/yprime
        numiter += 1


def secant(f, x_0, x_1, threshold=THRESHOLD, full_output=False):
    """
    Secant method for root finding

    Parameters
    ----------
    f : function
        Mathematical function to find root
    x_0 : float
        First initial guess
    x_1 : float
        Second initial guess
    threshold: float, optional
        Threshold to stop iterating
    full_output: bool, optional
        Non-zero to return all optional outputs

    Returns
    -------
    x_n : float
        Value of root
    numiter : int
        Number of iterations, if full_output == True
    """

    numiter = 0
    if abs(f(x_0)) <= threshold:
        if full_output:
            return x_0, numiter
        return x_0

    x_nm1 = x_0 #x_n-1
    x_n = x_1
    while True:
        y_nm1 = f(x_nm1)
        y_n = f(x_n)
        if abs(y_n) <= threshold:
            if full_output:
                return x_n, numiter
            return x_n
        dx = x_n - x_nm1
        dy = y_n - y_nm1

        x_n = x_n - y_n * dx / dy #technically x_np1
        numiter += 1
