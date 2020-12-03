'''
Michael Lam
ASTP-720, Fall 2020

Cooley-Tukey Fast Fourier Transform Algorithm
'''

import numpy as np


def fft(h_n, threshold=16):
    """
    Cooley-Tukey Fast Fourier Transform Algorithm

    Parameters
    ----------
    h_n : list, np.ndarray
        Array to perform FFT, h_n
    threshold (optional) : int
        Threshold below which one does the classic DFT calculation

    Returns
    -------
    H_k : np.ndarray
        Discrete Fourier Transform, H_k
    """

    N = len(h_n)
    range_arr = np.arange(N)
    w = np.exp(-2*np.pi*1j/N)


    if N <= threshold:
        # W_kn matrix, see Lecture 19 notes
        # First, define the the range [0, 1, ... N-1]
        # Then, use meshgrid and multiply the results to get the powers
        # of w, i.e., w^(kn) power.
        n_grid, k_grid = np.meshgrid(range_arr, range_arr)
        W_kn = w**(k_grid*n_grid)
        # Now dot with h, H = W@h
        H_k = W_kn @ h_n
    else:
        # Use slice indexing to get the even and odd parts of h_n
        h_n_even = h_n[::2]
        h_n_odd = h_n[1::2]
        # Recursively call fft()
        E_k = fft(h_n_even)
        O_k = fft(h_n_odd)
        # Now combine, with w^k, make this an array for ease
        w_pow_k = w**range_arr
        H_k = np.zeros(N, dtype=np.complex)
        H_k[:N//2] = E_k + O_k*w_pow_k[:N//2] #first half-interval of H_k
        H_k[N//2:] = E_k + O_k*w_pow_k[N//2:] #second half-interval of H_k

    return H_k
