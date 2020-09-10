'''
Michael Lam
ASTP-720, Spring 2020

Set of functions to perform interpolation

Should only do linear interpolation for now 
as we haven't discussed matrices yet, sorry
'''



def linear_interpolator(xs, ys):
    """
    Linear interpolation of points (x, y)

    Parameters
    ----------
    xs : list, np.ndarray
        List of x values
    ys : list, np.ndarray
        List of x values

    Returns
    -------
    return_function : float
        Interpolating function

    """

    def return_function(xp):
        """ xp is the x value to find the interpolation """
        for i in range(len(xs)-1):
            # Define interpolating range for this iteration
            x0 = xs[i]
            x1 = xs[i+1]
            if not (x0 <= xp <= x1):
                continue
            y0 = ys[i]
            y1 = ys[i+1]
            xd = (xp - x0)/float(x1 - x0)
            yp = y0*(1-xd) + y1*xd
            return yp
        return None
    return return_function
