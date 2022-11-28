import numpy as np


def quantise_audio(in_signal, N, up_limit, down_limit, NSFlag, dither_flag):
    """

    :param in_signal: Numpy array holding the input signal
    :param N: Integer. Quantization bits
    :param up_limit:
    :param down_limit:
    :param NSFlag: Integer
                    0 No noise-shaping
                    2 2nd order noise-shaping
                    3 3rd order noise-shaping
    :param dither_flag: Integer
                        0   No dither
                        1	RPDF dither
                        2	TPDF dither
                        3   HighPass TPDF dither
    :return:
    """
    out_signal = []
    dither = []

    # Length of samples for the input signal
    npoints = np.size(in_signal)

    # Calculate the PCM quantization step
    LSB = (up_limit - down_limit) / (2 ** N - 1)

    # Calculate dither sequence if desired
    if dither_flag == 0:
        dither = np.zeros(np.size(in_signal))
        in_signal = in_signal + dither
    else:
        dither = create_dither(dither_flag, LSB, npoints)
        in_signal = in_signal + dither


    # Perform quantization process without NoiseShaping
    if NSFlag == 0:
        out_signal = LSB * np.floor((in_signal / LSB) + 1 / 2)
        # Instead of using find, usage of numpy.where

        # extreme cases where s exceeds up_limit
        out_signal = np.where(out_signal > up_limit,up_limit, out_signal)

        # extreme case where s goes lower than down_limit
        out_signal = np.where(out_signal < down_limit, down_limit, out_signal)

    # Perform quantization process 2nd Order NoiseShaping
    if NSFlag == 2:
        f1 = 0
        f2 = 0
        out_signal = np.zeros(npoints)
        for i in range(0,npoints):
            w = 2 * f1 - f2
            out_signal[i] = LSB * np.floor(((in_signal[i] - w) / LSB) + 1 / 2)

            if out_signal[i] > up_limit:
                out_signal[i] = up_limit

            if out_signal[i] < down_limit:
                out_signal[i] = down_limit

            f = out_signal[i] - in_signal[i] + w

            f2 = f1
            f1 = f

    # Perform quantization process 3rd Order NoiseShaping.
    if NSFlag == 3:
        f1 = 0
        f2 = 0
        f3 = 0
        for i in range(0, npoints):
            w = 3 * f1 - 3 * f2 + f3
            out_signal[i] = LSB * np.floor(((in_signal[i] - w) / LSB) + 1 / 2)

            if out_signal[i] > up_limit:
                out_signal[i] = up_limit

            if out_signal[i] < down_limit:
                out_signal[i] = down_limit

            f = out_signal[i] - in_signal[i] + w
            f3 = f2
            f2 = f1
            f1 = f

    return out_signal


def create_dither(dithtype, LSB, npoints):
    """

    :param dithtype:
        1 : RPDF in the range [-LSB/2 LSB/2]
        2 : TPDF in the range [-LSB LSB]
        3 : HighPass RPDF in the range [-LSB LSB]
    :param LSB: amplitude value that corresponds to the Least Significant Bit of the quantizer
    :param npoints: length of signal
    :return:
    """
    assert (dithtype in [1, 2, 3])
    # Initialization
    # a1 = np.random.randint(1e4)
    a1 = 3453
    # a2 = np.random.randint(1e4)
    a2 = 2945
    # m = 2 ** bits  # power of 2 - is it related to quantization bits?
    m = 2 ** 16
    coeff = 3453
    c1 = 1
    c2 = 1
    dithran1 = 1531
    dithran2 = 18531
    dithtemp1 = 0
    dithtemp2 = 0
    dithtemp = 0

    bits = 16

    dith = np.zeros(npoints)
    if dithtype == 1:
        # "RPDF"
        for i in range(0, npoints):
            dithran1 = np.mod((dithran1 * a1) + c1, m)
            dithtemp1 = (LSB * (dithran1 - (2 ** (bits - 1) - 1))) / (2 ** bits)
            dith[i] = dithtemp1
    elif dithtype == 2:
        # "TPDF"
        for i in range(0, npoints):
            dithran1 = np.mod((dithran1 * a1) + c1, m)
            dithran2 = np.mod((dithran1 * a2) + c2, m)
            dithtemp1 = (LSB * (dithran1 - (2 ** (bits - 1) - 1))) / (2 ** bits)
            dithtemp2 = (LSB * (dithran2 - (2 ** (bits - 1) - 1))) / (2 ** bits)
            dithtemp = np.divide((dithtemp1 + dithtemp2), 2)
            dith[i] = dithtemp
    else:
        for i in range(0, npoints):
            # "HP-TPDF"
            dithran1 = np.mod((dithran1 * a1) + c1, m)
            dithtemp1 = (LSB * (dithran1 - (2 ** (bits - 1) - 1))) / (2 ** bits)
            dithtemp = (dithtemp1 - dithtemp2) / 2
            dithtemp2 = dithtemp
            dith[i] = dithtemp
    return dith


def plotSpectrum(in_signal, fs, db=True, logx=False, ax=None):
    """
    
    :param in_signal: 
    :param fs: 
    :param db: 
    :param logx: 
    :param ax: 
    :return: 
    """
    l = np.size(in_signal)
    # Different methods to generate fvec
    # timestep = 1 / rate
    # fvec = np.fft.fftfreq(S.shape[0], d=timestep)
    # fvec = np.linspace(0, fs, fs)
    fvec = [x for x in np.arange(0,(fs-(1/l)), fs/l)]

    spec = np.fft.fft(in_signal)
    magSpec = np.abs(spec)
    magSpec_dB = 20*np.log10(magSpec/np.max(magSpec))

    if ax is not None:
        if logx:
            pl = ax.semilogx(fvec[:l//2 + 1], magSpec_dB[:l // 2 + 1] if db else magSpec[:l // 2 + 1],  linewidth=1)
        else:
            pl = ax.plot(fvec[:l // 2 + 1], magSpec_dB[:l // 2 + 1] if db else magSpec[:l // 2 + 1],  linewidth=1)
    else:
        raise Exception('No active axe object passed to function')

    return ax


def Interpolate_zeros(input_signal, R):
    """
    This function adds R-1 zeros between the original InputSignal
    digital samples. It can be used for performing xR oversampling
    (prior to FIR filtering and requantization)
    :param input_signal:
    :param R:
    :return:
    """
    size = input_signal.shape
    if len(size) > 1:
        rows = size[0]
        cols = size[1]
        if cols > rows:
            input_signal = input_signal.transpose()

    padding_matrix = np.zeros(R)
    padding_matrix[0] = 1

    """
     Now calculate the kronecker product of the input signal
    with the calculated zero-padding_matrix
    """

    OSS_signal_with_zeros = np.kron(input_signal, padding_matrix)
    return OSS_signal_with_zeros


def calc_thdn(in_spectrum, freq, c):
    """
    This function calculates the Total Harmonic Doistortion+Noise (THD+N)
    for a sinewave signal, using the equation:

                    spurious harmonic power
       THD_NOISE = -------------------------
                    total power

    The input argument c represents the index of the spectral coefficient
    that corresponds to the input sinewave frequency.

    :param in_spectrum:
    :param freq:
    :param c: SineFrequency
    :return:
    """
    audible_freq_mask = np.where(freq < 20000)  #Audible frequency region
    # audible_freqs = freq[audible_freq_mask]

    in_spectrum = in_spectrum[audible_freq_mask]

    in_spectrum = np.power(in_spectrum, 2)

    # Calculate the total spectral power
    den = np.sum(in_spectrum) - in_spectrum[1]

    # Calculate the spurious spectral power
    num = den - in_spectrum[c] - in_spectrum[c - 1] - in_spectrum[c + 1]

    # Final calculation of THD_NOISE parameter
    thd_noise = (num / den)

    return thd_noise




