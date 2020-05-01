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
print("Guard interval size is {}".format(GI_NUM))
SNRDB = 10


def f_symbol(data, mode):

    if len(data) != int(mode):
        raise ValueError

    # fill symbol with data in frequency domain
    f_data = np.zeros(FFT_SIZE, complex)
    f_data[subcarriers[mode]] = data

    return f_data


def send_symbol(f_data):
    # translate to time domain
    t_data = np.fft.ifft(f_data)

    # add cyclic prefix (guard interval)
    return np.hstack([t_data[-GI_NUM:], t_data])


tx_f_data = f_symbol(frame_headers[MODE], MODE)
tx_out = send_symbol(tx_f_data)
# add one more symbol for better picture
tx_out = np.hstack([tx_out, send_symbol(tx_f_data)])


def add_delay(signal, delay_len=None):
    """Adds random time delay"""
    if not delay_len:
        delay_len = random.randrange(0, FFT_SIZE+GI_NUM)
    print("Add {} delay cycles".format(delay_len))
    delay = np.zeros(delay_len, complex)
    # delay = [rand() + rand()*1j for _ in range(delay_len)]
    return np.hstack([delay, signal]), delay_len


delayed, DELAY = add_delay(tx_out)
# print(delayed)


def add_awgn(signal, snr_db):
    """add AWGN"""
    signal_power = np.mean(abs(signal**2))
    sigma2 = signal_power * 10**(-snr_db/10)  # calculate noise power based on signal power and SNR

    print("RX Signal power: %.4f. Noise power: %.4f" % (signal_power, sigma2))

    # Generate complex noise with given variance
    noise = np.sqrt(sigma2/2) * (np.random.randn(*signal.shape) +
    1j*np.random.randn(*signal.shape))

    return signal + noise


rx_in = add_awgn(delayed, SNRDB)
# rx_in = delayed

# plt.figure()
# plt.subplot(1, 2, 1)
# plt.plot([x.real for x in delayed], label='TX CLEAR')
# plt.plot([x.real for x in rx_in], label='TX AWGN')
# plt.subplot(1, 2, 2)
# plt.plot([x.imag for x in delayed], label='TX CLEAR')
# plt.plot([x.imag for x in rx_in], label='TX AWGN')


# coarse timing offset estimation
def delay_n_correlate(samples):
    """Delay & correlate method"""

    def sample(i):
        return 0 if i < 0 else samples[i]

    F = []

    for m in range(len(samples)):
        conv = 0
        for r in range(0, GI_NUM):
            conv += sample(m-r) * sample(m-r-FFT_SIZE).conjugate()
        F.append(abs(conv))

    return F


def mmse(samples):

    def sample(i):
        return 0 if i < 0 else samples[i]

    F = []

    for m in range(len(samples)):
        sq = 0
        sq_del = 0
        conv = 0
        for r in range(0, GI_NUM):
            sq += abs(sample(m-r))**2
            sq_del += abs(sample(m-r-FFT_SIZE))**2
            conv += sample(m-r) * sample(m-r-FFT_SIZE).conjugate()
        F.append(sq + sq_del - 2*abs(conv))

    return F


F1 = delay_n_correlate(rx_in)
F2 = mmse(rx_in)

F1_max = F1.index(max(F1))
F2_min = FFT_SIZE + GI_NUM + F2[FFT_SIZE+GI_NUM:].index(min(F2[FFT_SIZE+GI_NUM:]))
print("Delay & correlate maximum at {}, packet starts at {}".format(
      F1_max, F1_max-FFT_SIZE-GI_NUM))
print("MMSE minimum at {}, packet starts at {}".format(
      F2_min, F2_min-FFT_SIZE-GI_NUM))

detected_symbol = rx_in[F2_min-FFT_SIZE-GI_NUM:F2_min]
detected_fft_window = rx_in[F2_min-FFT_SIZE:F2_min]

plt.figure()
plt.plot(F1, label='Delay & Correlation')
plt.plot(F2, label='MMSE')
# plt.axvline(x=F1.index(max(F1)), label="max", color='red')
plt.axvline(x=DELAY, label="symbol start", color='green', ls='--')
# plt.axvline(x=DELAY+GI_NUM, label="GI end", color='blue', ls='--')
plt.axvline(x=DELAY+GI_NUM+FFT_SIZE, color='green', ls='--')
plt.axvline(x=DELAY+2*(GI_NUM+FFT_SIZE), color='green', ls='--')
plt.legend()

# ---------------------------------------------------
# rx_f_data = np.fft.fft(detected_fft_window)
rx_f_data = [round(x.real, 0) for x in np.fft.fft(detected_fft_window)]

# x = np.linspace(0, FFT_SIZE, FFT_SIZE)
plt.figure()
plt.title("Frequency domain data (real part)")
plt.plot([y.real for y in tx_f_data], label='TX')
plt.plot([y.real for y in rx_f_data], label='RX')
plt.legend()


errors = sum([rx_f_data[i].real-tx_f_data[i].real != 0 for i in range(FFT_SIZE)])
print("Receive errors: {}".format(errors))

plt.show()
