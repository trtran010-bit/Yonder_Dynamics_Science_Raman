import argparse

SPEC_CALLIBRATION = [0, -0.00000383008, -0.179129, 717.783]

def pos_int(v):
    v = int(v)
    if v <= 0:
        raise ValueError
    return v

# Imports take ~2 seconds, so don't bother if the command line is wrong
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='raman_analyze',
        description='Analyzes Raman spectrography data',
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '-s', '--from-spec',
        action='store_true',
        help=(
            'If given, data is read from the spectrometer. --integration-time and --num-avgs '
            'configure the settings used to read the spectrum. Must not be given with '
            '--spectrum-file.'
        ),
    )
    input_group.add_argument(
        '-f', '--spectrum-file',
        dest='spectrum',
        help=(
            'If given, data is read from SPECTRUM_FILE, which is interpreted as a Spectrum '
            'Studio CSV of analyte spectrum. Must not be given with --from-spec. '
            'Callibration coefficients: '
        ) + ', '.join(f'C{i} = {c}' for i, c in zip(range(3, -1, -1), SPEC_CALLIBRATION)),
    )
    parser.add_argument(
        '-i', '--integration-time',
        type=pos_int,
        default=10000,
        help=(
            'Integration time when reading from spectrometer. Must not be given '
            'with --spectrum-file.'
        ),
    )
    parser.add_argument(
        '-n', '--num-avgs',
        type=pos_int,
        default=3,
        help=(
            'Number of averages when reading from spectrometer. Must not be given '
            'with --spectrum-file.'
        ),
    )
    parser.add_argument(
        '--blank',
        help=(
            'Path to Spectrum Studio CSV of blank spectrum, used in blank subtraction if given.'
        ),
    )
    parser.add_argument(
        '--no-show',
        dest='show_graph',
        action='store_false',
        help='Do not show the graph onscreen',
    )
    args = parser.parse_args()
    if (args.num_avgs or args.integration_time) and args.spectrum:
        parser.error(
            'Cannot specify spectrometer settings when reading data from file.'
        )

from matplotlib.lines import Line2D
from scipy import signal, sparse
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, firwin, lfilter
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import sys
from spectrometer_decode import read_spectrometer, find_port

class RamanDenoiser:
    def __init__(self, wavelengths=None, intensities=None):
        if wavelengths is not None and intensities is not None:
            self.wavelengths = np.array(wavelengths)
            self.wavenumbers = (1 / 532 - 1 / wavelengths) * 10**7
            self.initial_intensities = np.array(intensities)
        else:
            self.wavelengths = None
            self.initial_intensities = None
        self.intensities = self.initial_intensities

    def clone(self):
        '''Returns a deep copy of the denoiser in its current state.'''
        new = type(self)()
        new.initial_intensities = self.initial_intensities.copy()
        new.wavelengths = self.wavelengths.copy()
        new.wavenumbers = self.wavenumbers.copy()
        new.intensities = self.intensities.copy()
        return new

    @classmethod
    def from_csv(cls, filepath, wavenumber_col=0, intensity_col=1, skiprows=0):
        try:
            #pandas
            df = pd.read_csv(filepath, skiprows=skiprows)
            df = df[df.iloc[:,wavenumber_col] > 540]

            if isinstance(wavenumber_col, int):
                wavelengths = df.iloc[:, wavenumber_col].values
            else:
                wavelengths = df[wavenumber_col].values

            if isinstance(intensity_col, int):
                intensities = df.iloc[:, intensity_col].values
            else:
                intensities = df[intensity_col].values

            print(f"Successfully loaded {len(wavelengths)} data points from {filepath}")
            return cls(wavelengths, intensities)

        except Exception as e:
            raise Exception(f"Error loading CSV '{filepath}'") from e

    @classmethod
    def from_spectrometer(cls, integration_time, num_avgs):
        port = find_port()
        intensities = []
        for _ in range(num_avgs):
            spectrum = read_spectrometer(integration_time, port)
            intensities.append(spectrum)
            print(f'took spectrum: [{spectrum.min()}, {spectrum.max()}]')
        intensities = np.sum(intensities, axis=0) / num_avgs
        wavelengths = np.polyval(SPEC_CALLIBRATION, np.arange(len(intensities)))
        return RamanDenoiser(wavelengths, intensities)

    def savitzky_golay(self, window_length=11, polyorder=3):
        if window_length % 2 == 0:
            window_length += 1
        self.intensities = signal.savgol_filter(
            self.intensities,
            window_length,
            polyorder
        )

    def gaussian_filter(self, sigma=2):
        self.intensities = gaussian_filter1d(self.intensities, sigma)

    def median_filter(self, kernel_size=5):
        self.intensities = signal.medfilt(self.intensities, kernel_size)

    def wiener_filter(self, noise_power=None):
        self.intensities = signal.wiener(self.intensities, mysize=5, noise=noise_power)

    def fir_filter(self, cutoff_freq=0.1, numtaps=51, window='hamming'):
        fir_coeff = firwin(numtaps, cutoff_freq, window=window)

        self.intensities = lfilter(fir_coeff, 1.0, self.intensities)

    def hilbert_vibration_decomposition(self, num_components=3):
        #hilbert transform method
        analytic_signal = hilbert(self.intensities)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)

        #storing denoiser signal
        self.intensities = amplitude_envelope

        #decomposition results
        self.hvd_results = {
            'analytic_signal': analytic_signal,
            'envelope': amplitude_envelope,
            'instantaneous_phase': instantaneous_phase,
            'instantaneous_frequency': instantaneous_frequency
        }

        return self.hvd_results

    def wavelet_denoise(self, wavelet='sym4', level=None, threshold_mode='soft'):
        if level is None:
            level = int(np.log2(len(self.intensities))) - 1

        #wavelet decomposition
        coeffs = pywt.wavedec(self.intensities, wavelet, level=level)

        #calculate threshold using MAD (Median Absolute Deviation)
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(self.intensities)))

        coeffs_thresh = [coeffs[0]]
        for i in range(1, len(coeffs)):
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode=threshold_mode))

        # recounstruction
        self.intensities = pywt.waverec(coeffs_thresh, wavelet)[:len(self.intensities)]

    def als_baseline(self, lam=1e6, p=0.01, niter=10):
        L = len(self.intensities)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2), dtype=np.float64)
        w = np.ones(L)

        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            baseline = spsolve(Z, w * self.intensities)
            w = p * (self.intensities > baseline) + (1 - p) * (self.intensities < baseline)

        self.intensities -= baseline

    def polynomial_baseline(self, degree=3):
        coeffs = np.polyfit(self.wavenumbers, self.intensities, degree)
        baseline = np.polyval(coeffs, self.wavenumbers)
        self.intensities -= baseline

    def normalize(self, method='max'):
        if method == 'max':
            self.intensities /= np.max(self.intensities)
        elif method == 'area':
            self.intensities /= np.trapz(self.intensities, self.wavenumbers)
        elif method == 'minmax':
            self.intensities = ((self.intensities - np.min(self.intensities)) \
                / (np.max(self.intensities) - np.min(self.intensities)))

    def trim(self, low=float('-inf'), high=float('inf')):
        mask = (low <= self.wavenumbers) & (self.wavenumbers <= high)
        self.intensities = self.intensities[mask]
        self.wavenumbers = self.wavenumbers[mask]

    def subtract_blank(self, blank, factor=2):
        if np.any(self.wavenumbers != blank.wavenumbers):
            raise ValueError('Wavenumber lists of operands do not match')
        self.intensities = np.maximum(self.intensities - blank.intensities * factor, 0)

    def find_peaks(self, prominence=None, distance=10, height=None, width=None, auto_adapt=True):
        # find peaks in the spectrum
        # if auto_adapt is True, it'll try to figure out good parameters for that material
        if auto_adapt and prominence is None:
            # calculate adaptive prominence based on noise level
            # using the standard deviation of the baseline region as noise estimate
            noise_std = np.std(self.intensities[:int(len(self.intensities)*0.1)])
            prominence = max(3 * noise_std, 0.05 * np.max(self.intensities))
            print(f"auto-detected prominence: {prominence:.4f}")

        if auto_adapt and height is None:
            # set minimum height as mean + 2*std
            height = np.mean(self.intensities) + 2 * np.std(self.intensities)
            print(f"auto-detected height threshold: {height:.4f}")

        peaks, properties = signal.find_peaks(
            self.intensities, prominence=prominence, distance=distance, height=height, width=width
        )

        return peaks, properties

    # testing new method
    def find_all_peaks_unbiased(self, min_prominence_ratio=0.01, min_distance=5):
        # completely unbiased peak detection so we can find ALL local maxima above minimal threshold
        # min_prominence_ratio: fraction of max intensity (e.g., 0.01 = 1% of max signal)
        # this way you see everything and can decide what matters for your material
        min_prominence = min_prominence_ratio * np.max(self.intensities)

        peaks, properties = signal.find_peaks(
            self.intensities,
            prominence=min_prominence,
            distance=min_distance

        )

        peak_data = []
        for i, peak_idx in enumerate(peaks):
            peak_data.append({
                'index': peak_idx,
                'wavenumber': self.wavenumbers[peak_idx],
                'intensity': self.intensities[peak_idx],
                'prominence': properties['prominences'][i],
                'relative_intensity': self.intensities[peak_idx] / np.max(self.intensities)
            })

        peak_data_sorted = sorted(peak_data, key=lambda x: x['intensity'], reverse=True)

        return peak_data_sorted

    def plot_comparison(self, title="Raman Spectrum Processing", show_peak_labels=True, fig_axs=None, defer=False, label=None):
        if fig_axs:
            fig, (ax1, ax2), lines = fig_axs
            ax1.plot(self.wavelengths, self.initial_intensities, '-', linewidth=1, alpha=0.7)
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Intensity (a.u.)')
            ax1.set_title('Original Spectrum')
            ax1.grid(True, alpha=0.3)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            lines = []

        lines.append(ax2.plot(self.wavenumbers, self.intensities, '-', linewidth=1.5, label=label)[0])
        peaks, properties = self.find_peaks(auto_adapt=True)
        classifications = self.classify_peaks(peaks, properties)

        colors = {'strong': 'darkgreen', 'medium': 'orange', 'weak': 'lightblue'}
        for peak, classification in zip(peaks, classifications):
            ax2.plot(self.wavenumbers[peak], self.intensities[peak], 'o',
                    color=colors[classification], markersize=8)

            if show_peak_labels and classification in ['strong', 'medium']:
                ax2.text(self.wavenumbers[peak], self.intensities[peak],
                       f'{self.wavenumbers[peak]:.0f}',
                       fontsize=8, ha='center', va='bottom')

        # legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen',
                  markersize=8, label='Strong peaks'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
                  markersize=8, label='Medium peaks'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                  markersize=8, label='Weak peaks'),
            *[line for line in lines if not line.get_label().startswith('_')],
        ]
        ax2.legend(handles=legend_elements)

        ax2.set_xlabel('Raman Shift (cm⁻¹)')
        ax2.set_ylabel('Normalized intensity')
        ax2.set_title('Processed Spectrum')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig, (ax1, ax2), lines

    def classify_peaks(self, peaks, properties):
        return ['strong' for _ in peaks]

    def save_to_file(self, filepath):
        df = pd.DataFrame({
            'wavenumber': self.wavenumbers,
            'intensity': self.intensities
        })
        df.to_csv(filepath, index=False)
        print(f"saved processed spectrum to {filepath}")

def raman_analysis(denoiser):
    denoiser.als_baseline(lam=1e5, p=0.01)
    denoiser.fir_filter(cutoff_freq=0.1, numtaps=51)
    #denoiser.wavelet_denoise(wavelet='db4', threshold_mode='soft', level=3)
    denoiser.trim(low=0)
    denoiser.normalize(method='max')
    #peaks, properties = denoiser.find_peaks(prominence=0.1, distance=20)
    #print(f"Found {len(peaks)} peaks")

if __name__ == "__main__":
    if args.spectrum:
        spectrum_fn_split = args.spectrum.split('.')
        spectrum_basename = '.'.join(
            spectrum_fn_split[slice(None, -1 if len(spectrum_fn_split) > 1 else None)]
        )
        spectrum = RamanDenoiser.from_csv(
            args.spectrum,
            wavenumber_col=1,
            intensity_col=3,
            skiprows=5
        )
    else:
        spectrum_basename = 'spectrum'
        spectrum = RamanDenoiser.from_spectrometer(args.integration_time, args.num_avgs)
    raman_analysis(spectrum)
    fig, axs, lines = spectrum.plot_comparison(label=("No blank sub" if args.blank else None))

    if args.blank is not None:
        blank = RamanDenoiser.from_csv(
            args.blank,
            wavenumber_col=1,
            intensity_col=3,
            skiprows=5
        )
        raman_analysis(blank)
        blank_subtracted = spectrum.clone()
        blank_subtracted.subtract_blank(blank)
        blank_subtracted.plot_comparison(fig_axs=(fig, axs, lines), label="Blank subtracted")

    fig.tight_layout()
    graph_path = spectrum_basename + '-graph.png'
    fig.savefig(graph_path)
    print(f"saved figure to {graph_path}")
    spectrum.save_to_file(spectrum_basename + '-denoised.csv')
    if args.show_graph:
        plt.show()
