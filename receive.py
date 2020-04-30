from constants import subcarriers, frame_headers
import matplotlib.pyplot as plt
import numpy as np
import random
import commpy

MODE = "23"
Tu = 24.00    # OFDM symbol payload length (ms)
Td = 2.66     # OFDM symbol GI length (ms)
Ts = Tu + Td  # full OFDM symbol length (ms)
Ns = 15       # number of symbols in packet
FFT_SIZE = 256
GI_NUM = int(FFT_SIZE*Td/Tu)  # number of samplas in Guard interval
SNRDB = 25


def send_symbol(symbol, mode):

    if len(symbol) != int(mode):
        raise ValueError

    # fill symbol with data in frequency domain
    f_data = np.zeros(FFT_SIZE, complex)
    f_data[subcarriers[mode]] = symbol

    # translate to time domain
    t_data = np.fft.ifft(f_data)

    # add cyclic prefix (guard interval)
    return np.hstack([t_data[-GI_NUM:], t_data])


tx_out = send_symbol(frame_headers[MODE], MODE)
print(len(tx_out))


def add_delay(signal):
    """Adds random time delay"""
    delay_len = random.randrange(0, len(signal))
    delay = np.zeros(delay_len, complex)
    # delay = [rand() + rand()*1j for _ in range(delay_len)]
    return np.hstack([delay, signal])


delayed = add_delay(tx_out)
# print(delayed)


def add_awgn(signal, snr_db):
    """add AWGN"""
    signal_power = np.mean(abs(signal**2))
    sigma2 = signal_power * 10**(-snr_db/10)  # calculate noise power based on signal power and SNR

    print("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))

    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*signal.shape)+1j*np.random.randn(*signal.shape))
    return signal + noise


rx_in = add_awgn(delayed, SNRDB)
# print(rx_in)

plt.subplot(1, 2, 1)
plt.plot([x.real for x in delayed], label='TX CLEAR')
plt.plot([x.real for x in rx_in], label='TX AWGN')
plt.subplot(1, 2, 2)
plt.plot([x.imag for x in delayed], label='TX CLEAR')
plt.plot([x.imag for x in rx_in], label='TX AWGN')
plt.show()