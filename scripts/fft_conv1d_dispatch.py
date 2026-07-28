"""Shared, dependency-free mirror of the Host algorithm dispatch."""

FFT_MIN_K = 64
FFT_MAX_NFFT_UB = 1024
FFT_MAX_NFFT_GM = 4096


def select_algorithm(L, K):
    """Return (algorithm name, power-of-four FFT length) for a shape."""
    need = max(2, L + K - 1)
    nfft = 4
    while nfft < need:
        nfft *= 4

    if K < FFT_MIN_K or need > FFT_MAX_NFFT_GM:
        algo = "DIRECT"
    elif need <= FFT_MAX_NFFT_UB:
        algo = "FFT-UB"
    else:
        algo = "FFT-GM"
    return algo, nfft
