import numpy as np
import pandas as pd
import scipy
import os

from obspy.signal.trigger import recursive_sta_lta
from scipy import signal, stats

def zeropad_trigger_lab(sta, lta, tau, signal_org, nstep=0):
    cft = recursive_sta_lta(signal_org, sta, lta)
    t0 = np.min(np.where(cft[int(1.5 * lta)::] >= tau))  # skip the first 1.5*lta window for the triggering
    t0 = t0 + int(1.5 * lta)

    # Zero-padding, if needed, to taper with Blackman-Harris window
    if nstep != 0:
        if t0 < nstep - 1:  # Need to zero-pad, signal not long enough
            signal_pad = signal_org[:(t0 + nstep - 1)]
            signal_pad = np.append(np.zeros(nstep - 1 - t0, 1), signal_pad)
        else:
            signal_pad = signal_org[(t0 - nstep + 1):(t0 + nstep - 1)]  # No need to zero-pad
    else:
        if t0 < len(signal_org) / 2:  # Need to zero-pad, signal not long enough
            signal_pad = np.append(np.zeros([len(signal_org[t0:]) - t0, 1]), signal_org)
        else:
            signal_pad = signal_org[t0:]

    # plt.figure(999)
    # plt.subplot(211)
    # plt.plot(np.arange(0,len(cft)),cft)
    # plt.plot(np.arange(0,len(signal_org)),signal_org/np.max(abs(signal_org))+1)
    # plt.vlines(t0,ymin=0,ymax=tau,color='r')
    #
    # plt.subplot(212)
    # #tpad=np.arange(0,t0*2,1)
    # #tsig=np.arange()
    # plt.plot(signal_pad[::-1])
    # plt.plot(signal_org[::-1])
    # plt.vlines(len(signal_pad)/2, ymin=min(signal_pad), ymax=max(signal_pad),color='r')
    # plt.show()

    return t0, signal_pad
def zeropad_theory(npoint, signal):
    # Zero-pads before first arrival
    if len(signal) > npoint:
        raise ValueError("Cannot zero pad because data is too short.")

    return np.pad(signal, (npoint - len(signal), 0), mode='constant')
def stack_timeseries(data,dt):

    # Stacks aligned traces

    if np.all(dt[0]*np.ones((1,len(dt)),dtype='int32') == dt)==1:

        data_stacked = np.mean(data, axis=1)
        data_std = np.std(data, axis=1)

        # plt.figure(99)
        # plt.subplot(311)
        # plt.plot(data)
        # plt.plot(data_stacked,'r')
        # plt.subplot(312)
        # plt.plot(data_stacked,'r')
        # plt.plot(data_stacked+data_std)
        # plt.plot(data_stacked-data_std)
        # plt.subplot(313)
        # plt.hist(data_std, bins='auto')
        # plt.show()

    else:
        print('Sampling rate not equal across all traces. Cannot average.')
        exit()

    return data_stacked,data_std
def load_one_mat_idx(dir,basename,file_nmb,idx1,idx2,ch):

    # Loads a single mat ball and snips data waveform using index 1 (idx1) and index 2 (idx2)

    [df,dt,skip_trace] = mat_io(dir, basename, file_nmb, ch)

    if idx1 > idx2:
        print('Cannot load mat as idx1 > idx2')
        exit()

    if idx1 < df['{:s}'.format(ch)].size and idx2 <= df['{:s}'.format(ch)].size:

        shape = df['{:s}'.format(ch)].shape
        if shape[1] == 1:  # For older version of Picoscope 6 software
            data = df['{:s}'.format(ch)][idx1:idx2]
        elif shape[0] == 1:  # For new version of Picoscope 7 software
            data = df['{:s}'.format(ch)][0, idx1:idx2]
            data = data.reshape(-1, 1)
        data

    return data,dt,skip_trace
def load_one_mat(dir, basename, file_nmb, t_tot, ch, skp):
    # Load single .mat file and extract waveform snippet

    # Load data from the .mat file
    df, dt, skip_trace = mat_io(dir, basename, file_nmb, ch)

    # Calculate start and end indices based on time length `t_tot` and sampling interval `dt`
    idx_offset = int(skp / dt)
    idx1 = int(df['Length'][0, 0] / 2 - t_tot / (2 * dt) + idx_offset)
    idx2 = idx1 + int(t_tot / dt)

    # Extract data based on the shape and reshape if necessary
    data_key = df[ch]
    if data_key.shape[1] == 1:  # Picoscope 6 format
        data = data_key[idx1:idx2]
    else:  # Picoscope 7 format
        data = data_key[0, idx1:idx2].reshape(-1, 1)

    # Retrieve relevant metadata
    voltage_range = df['voltage_range'][0, 0]
    ball_diameter = df['ball_diameter'][0, 0]
    height = df['height'][0, 0]

    return data, dt, idx1, idx2, voltage_range, skip_trace, ball_diameter, height
def mat_io(dir, basename, file_nmb, ch):

    # Load .mat file
    file_name = f"{dir}{basename}-{file_nmb:04d}.mat"
    print(f"Loading {file_name}", end=' ')
    df = scipy.io.loadmat(file_name)

    # Calculate and round sampling interval
    dt = 1 / round(1 / df['Tinterval'][0, 0])

    # Access channel data and reshape if necessary
    data = df[ch]
    if data.shape[1] == 1:  # Picoscope 6 format
        data = data[:, 0]
    else:  # Picoscope 7 format
        data = data[0, :].reshape(-1, 1)

    print("Done")

    # Remove DC offset
    data -= np.average(data)
    # print("DC offset subtracted")

    # Check for NaN or Inf values
    skip_trace = np.isinf(data).any() or np.isnan(data).any()
    if skip_trace:
        print("Data contains NaN or Inf.")

    return df, dt, skip_trace
def load_mat(dir, basename, nb_files, t_tot, ch, skp):

    # Loads .mat files (MATLAB) for sensor data
    # Snips data waveform from center to a +/- length of t_tot (s)
    #
    # Input:
    # dir: directory path to ball (string, e.g. '/home/user/')
    # basename: basename of ball to load (e.g. 'chA.mat')
    # nb_files: number of files N (e.g. '3')
    # t_tot: total data length to load in seconds (e.g. '0.25')
    # ch: variable name of data vector in .mat file (e.g. 'A')
    # skp: total +/- event time to skip from trigger in seconds (e.g. 0.02sec)
    #
    # Output:
    # data_all: data matrix
    # dt_all: time step dt used in acquisition
    # idx: index of data loaded
    # voltage_range: voltage range used in acquisition
    # ball_diameter_all: ball diameters for particular data
    # height_all: drop height for particular data

    # Convert `skp` to an array to handle skips more effectively
    skp_array = np.full(nb_files, skp if isinstance(skp, int) else skp)

    data_all = []
    dt_all = []
    ball_diameter_all = []
    height_all = []
    filenmb = []
    idx = []

    for i in range(1, nb_files + 1):
        # Load data for each file
        data, dt, idx1, idx2, voltage_range, skip_trace, ball_diameter, height = (
            load_one_mat(dir, basename, i, t_tot, ch, skp_array[i - 1]))

        if not skip_trace:
            data_all.append(data)
            dt_all.append(dt)
            ball_diameter_all.append(ball_diameter)
            height_all.append(height)
            idx.append([idx1, idx2])
            filenmb.append(i)
        else:
            print(f'Skipping ball impact for file number {i}')

    # Convert lists to numpy arrays for further processing
    data_all = np.column_stack(data_all)
    dt_all = np.array(dt_all)
    ball_diameter_all = np.array(ball_diameter_all)
    height_all = np.array(height_all)
    idx = np.array(idx)
    idx = np.transpose(idx)

    # Check for consistency in acquisition parameters
    if not np.allclose(dt_all, dt_all[0]):
        raise ValueError("Sampling rates (1/dt) should be consistent across all experiments.")
    if not np.allclose(ball_diameter_all, ball_diameter_all[0]):
        raise ValueError("Ball diameters should be consistent across all experiments.")
    if not np.allclose(height_all, height_all[0]):
        raise ValueError("Drop heights should be consistent across all experiments.")

    return data_all, dt_all, idx, voltage_range, filenmb, ball_diameter_all, height_all
def corr_align_mat(data, dt, t_tot, idx, t_tot_align, dir, basename, ch, filenmb):

    # Aligns all waveforms based on the offset of the first input data trace (data[:,0])
    # Alignment buffer is a np.nan value

    if data.shape[1] == 1:
        print('Data is only 1 trace. No need to corr_align.')
        exit()
    if t_tot_align == 0:
        print('Cannot align if total_tot_align = 0s')
        exit()

    if np.all(dt[0]*np.ones((1,len(dt)),dtype='int32') == dt)==0:
        print('Make sure sampling rates (1/dt) are equal across all balldrops.')
        exit()

    if t_tot_align > t_tot:
        print('Error the total event time for alignment exceeds total event time.')
        print('t_tot_align > t_tot')
        exit()

    for i in range(data.shape[1]):

        # Index for correlation
        idx1_corr = int((t_tot - t_tot_align)/dt[i])
        idx2_corr = int((t_tot + t_tot_align)/dt[i])

        if idx2_corr > data.shape[0]:
            idx1_corr = 0
            idx2_corr = data.shape[0]
            print('Setting t_tot_align = t_tot')

        if i == 0:
            corr = scipy.signal.correlate(data[idx1_corr:idx2_corr,0],data[idx1_corr:idx2_corr,i])
        else:
            corr = np.column_stack((corr,scipy.signal.correlate(data[idx1_corr:idx2_corr, 0], data[idx1_corr:idx2_corr, i])))

        # plt.figure(-99)
        # plt.subplot(211)
        # plt.plot(np.arange(0, len(data[idx1_corr:idx2_corr, 0])), data[idx1_corr:idx2_corr, 0])
        # plt.plot(np.arange(0, len(data[idx1_corr:idx2_corr, i])), data[idx1_corr:idx2_corr, i])
        # plt.subplot(212)
        # if i == 0:
        #     plt.plot(np.arange(0, len(corr[:])), corr[:])
        #     plt.plot(corr.argmax(), np.max(corr), 'r*')
        # else:
        #     plt.plot(np.arange(0, len(corr[:, i])), corr[:, i])
        #     plt.plot(corr[:, i].argmax(), np.max(corr[:, i]), 'r*')
        # plt.show()


    for i in range(corr.shape[1]):

        if i==0:
            corr_shift = corr[:,i].argmax()
        else:
            corr_shift = np.column_stack((corr_shift,(corr[:, i].argmax())))

    corr_shift = np.ndarray.flatten(corr_shift - corr_shift[0,0])

    data_aligned = np.zeros(data.shape)
    idx_corr = np.zeros((2, len(corr_shift)), dtype='int')

    for i in range(len(corr_shift)):
        [data_snip,dt_snip,skip_trace] = load_one_mat_idx(dir, basename, filenmb[i], idx[0,i] - corr_shift[i], idx[1,i] - corr_shift[i],ch)

        data_aligned[:, i] = np.ndarray.flatten(data_snip)

        idx_corr[0,i] = idx[0,i] - corr_shift[i]
        idx_corr[1,i] = idx[1,i] - corr_shift[i]

    return data_aligned,corr,corr_shift,idx_corr
def load_mat_idx(dir,basename,nb_files,idx,ch):

    # Aligns all waveforms based on the index of the trace
    # Alignment buffer is a np.nan value

    # Loads .mat files (MATLAB)
    #
    # Input:
    # dir: directory path to ball (string, e.g. '/home/user/')
    # basename: basename of ball to load (e.g. 'ChA.mat')
    # Loads mat ball and snips data waveform from center to a +/- length of t_tot (s)

    j = 1
    filenmb = []
    for i in range(1, nb_files + 1):

        idx1 = idx[0,i-1]
        idx2 = idx[1,i-1]
        [data,dt,skip_trace] = load_one_mat_idx(dir, basename, i, idx1, idx2, ch)

        if skip_trace == False:

            if j == 1:
                data_all = data
                dt_all = np.array(dt)
                j = 0
            else:
                data_all = np.column_stack((data_all, data))
                dt_all = np.append(dt_all, dt)

            filenmb.append(i)

        else:
            print('Skipping ball')



    if np.all(dt_all[0] * np.ones((1, len(dt_all)), dtype='int32') == dt_all) == 0:
        print('Make sure sampling rates (1/dt) are equal across all experiments.')
        exit()

    return data_all,dt_all,idx,filenmb
def corr_align_diff(wav1, wav2, sta, lta, trig):
    """
    Align two waveforms by calculating the shift needed using STA/LTA and correlation.

    Parameters:
        wav1 (ndarray): Reference waveform.
        wav2 (ndarray): Waveform to align.
        sta (int): Short-term average window.
        lta (int): Long-term average window.
        trig (float): Initial STA/LTA trigger threshold.

    Returns:
        tuple: Correlation array and alignment lag (lag_corr).
    """

    # Check STA/LTA windows
    if sta + lta > len(wav2):
        raise ValueError("STA/LTA windows too long: sta + lta > len(wav2)")

    # Flatten waveforms
    wav1 = wav1.flatten()
    wav2 = wav2.flatten()

    # Compute STA/LTA function and trigger
    cft2 = recursive_sta_lta(wav2, sta, lta)
    idx = np.argmax(cft2 > trig) if np.any(cft2 > trig) else None

    # Adjust trigger threshold if needed
    while idx is None:
        trig -= 0.1
        print(f'No STA/LTA detection. Adjusting trigger to {trig:.1f}')
        idx = np.argmax(cft2 > trig) if np.any(cft2 > trig) else None

    # Calculate alignment lag based on STA/LTA and correlation

    lag_sta_lta = len(wav2) // 2 - idx
    wav2_cut = wav2[int(idx - len(wav1)/2): int(idx + len(wav1)/2)]

    # Perform cross-correlation
    corr = scipy.signal.correlate(wav1, wav2_cut, mode='full')
    lags = scipy.signal.correlation_lags(len(wav1), len(wav2_cut), mode='full')

    # Find maximum correlation within a window centered on STA/LTA lag
    corr_box = corr[int((len(corr) - sta) / 2):int((len(corr) + sta) / 2)]
    lags_box = lags[int((len(corr) - sta) / 2):int((len(corr) + sta) / 2)]

    idx2 = lags_box[np.argmax(corr_box)]  # Peak lag in the correlation box
    lag_corr = idx2 + lag_sta_lta

    # print(idx)
    # print(idx+idx2)
    # print(-lag_sta_lta)
    # print(-lag_corr)
    #
    # plt.close(-99)
    # plt.figure(-99)
    # plt.subplot(611)
    # plt.plot(wav2)
    # plt.axvline(idx, color = 'r', label = f'sta {idx}')
    # plt.axvline(idx + idx2, color = 'purple', label = f'corr {idx + idx2}')
    # plt.xlim([len(wav2)/2-len(wav1)/2, len(wav2)/2+len(wav1)/2])
    # plt.legend()
    # plt.subplot(612)
    # plt.plot(wav2_cut)
    # plt.plot(np.roll(wav2_cut,-lag_sta_lta), label = f'sta {-lag_sta_lta}')
    # plt.plot(np.roll(wav2_cut,-lag_corr), label = f'corr {-lag_corr}')
    # plt.legend()
    # plt.subplot(613)
    # plt.plot(wav1)
    # plt.subplot(614)
    # plt.plot(wav2_cut)
    # plt.subplot(615)
    # plt.plot(cft2)
    # plt.axvline(idx, color = 'r')
    # plt.subplot(616)
    # plt.plot(lags, corr)
    # plt.plot(lags_box, corr_box)
    # plt.axvline(idx2, color = 'purple')
    # plt.show()

    return corr, lag_corr
def spatial_distance_align(data, dt, idx, dir, basename, ch, filenmb):

    if data.shape[1] == 1:
        print('Data is only 1 trace. No need to align.')
        exit()

    if np.all(dt[0]*np.ones((1,len(dt)),dtype='int32') == dt)==0:
        print('Make sure sampling rates (1/dt) are equal across all balldrops.')
        exit()

    distance_sum = np.zeros(data.shape)
    roll = np.zeros(data.shape[1], dtype=int)
    for i in range(data.shape[1]):
        count = 0
        for j in np.arange(-int(data.shape[0]/2),-int(data.shape[0]/2)+data.shape[0]):
            data_org = np.ndarray.flatten(data[:,0])
            data_shift = np.roll(data[:,i],j)
            distance = scipy.spatial.distance.euclidean(data_org, data_shift)
            distance_sum[count,i] = distance
            count = count + 1

        roll[i] = distance_sum[:,i].argmin() - int(data.shape[0]/2)

        min_index = distance_sum[:, i].argmin()

        # plt.figure(-99)
        # plt.subplot(311)
        # plt.title('roll = {:d}'.format(roll[i]))
        # plt.plot(np.arange(-int(data.shape[0]/2),-int(data.shape[0]/2)+data.shape[0]),distance_sum[:,i])
        # plt.plot(roll[i],distance_sum[min_index, i],'r*')
        # plt.xlabel('Roll')
        # plt.subplot(312)
        # plt.plot(data[:,0])
        # plt.plot(data[:,i])
        # plt.subplot(313)
        # plt.plot(data[:,0])
        # plt.plot(np.roll(data[:,i],roll[i]))
        # plt.show()

    data_aligned = np.zeros(data.shape)
    idx_coor = np.zeros((2,len(roll)),dtype='int')
    for i in range(len(roll)):

        idx1 = idx[0,i] - roll[i]
        idx2 = idx[1,i] - roll[i]
        [data_snip, dt_snip, skip_trace] = load_one_mat_idx(dir, basename, filenmb[i], idx1,
                                                            idx2, ch)
        data_aligned[:, i] = np.ndarray.flatten(data_snip)

        idx_coor[0,i] = idx[0,i] - roll[i]
        idx_coor[1,i] = idx[1,i] - roll[i]

    return data_aligned,distance_sum,roll,idx_coor
def specfem_read(filepath_fe, filename_fe, nrow=None):
    # Load the file, handling both full and limited rows with default nrows=None
    filepath = os.path.join(filepath_fe, filename_fe)

    print(f"Loading {filepath}", end=' ')
    df = pd.read_table(filepath, header=None, sep=' ', skipinitialspace=True, engine='python', nrows=nrow)
    print("Done")

    # Extract dt and amp columns
    dt, amp = df[0].values, df[1].values

    return dt, amp
def ball_impact(sf, R1, h1, mu1, E1, rho1, mu2, E2, v_ini=None, sf_min=None):
    # Ball impact / Hertz impact theory
    # Equations from Wu and McLaskey 2018, Gregory and Glaser 2012
    #
    # Inputs:
    #
    # sf: sampling frequency (Hz)
    # sf_min: minimum frequency (Hz) or zero padding
    #
    # R1: radius of ball (m)
    # h1: drop height of the ball (m)
    # v_ini: incoming velocity (m/s)
    #
    # mu1: Poisson ratio ball ()
    # E1: Young's modulus ball (Pa)
    # rho1: Density of ball (kg/m3)
    #
    # mu2: Poisson ratio platen ()
    # E2: Young's modulus platen (Pa)
    #
    # Outputs:
    #
    # ft: force deformation due to ball impact (N)
    #
    # t: rise time (s)
    #
    # tc: contact time (s)

    # Set default incoming velocity and minimum frequency if not provided
    v0 = np.sqrt(2 * 9.81 * h1) if v_ini is None else v_ini
    sf_min = sf if sf_min is None else sf_min

    # Precompute constants for Hertz impact theory
    delta = (1 - mu1**2) / (np.pi * E1) + (1 - mu2**2) / (np.pi * E2)
    tc = 4.53 * (4 * rho1 * np.pi * delta / 3)**0.4 * R1 * v0**-0.2  # Contact time (s)
    fmax = 1.917 * rho1**0.6 * delta**-0.4 * R1**2 * v0**1.2  # Maximum force upon impact (N)

    # Time array and force function during contact
    t1 = np.arange(0, tc, 1 / sf)
    ft1 = fmax * np.sin(np.pi * t1 / tc)**1.5  # Forcing function (N)

    # Apply zero padding if needed
    if (1 / sf_min - 1 / sf) > tc:

        t = np.arange(0, (1/sf_min-1/sf), 1 / sf)
        ft = np.zeros(len(t))  # zero padding

        t[:len(t1)] = t1
        ft[:len(ft1)] = ft1

    else:
        t, ft = t1, ft1

    return ft, t, tc
def bin_fft(freq, amp, freq_bin, amp_std = None, bin_min = None):

    if bin_min is None:
        bins = np.arange(np.min(freq), np.max(freq), freq_bin)
        bin_means, bin_edges, binnumber = stats.binned_statistic(freq, amp, statistic='mean',
                                                                 bins=bins)  # Sampling by bin means (smoothing)
    else:
        bins = np.arange(bin_min, np.max(freq), freq_bin)
        bin_means, bin_edges, binnumber = stats.binned_statistic(freq, amp, statistic='mean',
                                                                 bins=bins,range=(bin_min,np.max(freq)))  # Sampling by bin means (smoothing)

    if amp_std is not None:

        bin_std_means, bin_edges_std, binnumber_std = stats.binned_statistic(freq, amp_std**2, statistic='sum',
                                                                 bins=bins)  # Sampling by bin means (smoothing)

        bin_std_count, bin_edges_std, binnumber_std = stats.binned_statistic(freq, amp_std**2, statistic='count',
                                                                 bins=bins)  # Sampling by bin means (smoothing)

        bin_std_means = np.sqrt(bin_std_means)/bin_std_count

    nan_idx = []
    if np.isnan(np.sum(bin_means)):
        print('*** Data gap in re-binned data ***')
        print('Set a lower SNR threshold OR set a greater nbin.')
        nan_idx = np.argwhere(np.isnan(bin_means))

    bin_width = (bin_edges[1] - bin_edges[0])
    bin_centers = bin_edges[1:] - bin_width / 2

    return bin_means, bin_centers, bin_std_means, nan_idx
