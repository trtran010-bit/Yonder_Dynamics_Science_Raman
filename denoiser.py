import argparse
from unicodedata import name

SPEC_CALLIBRATION = [0, -0.00000383008, -0.179129, 717.783]
DEFAULT_BLANK_SUB_FAC = 1.0

def pos_int(v):
    v = int(v)
    if v <= 0:
        raise ValueError
    return v

def pos_float(v):
    v = float(v)
    if v <= 0:
        raise ValueError
    return v

# Imports take ~2 seconds, so don't bother if the command line is wrong
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='raman_analyze',
        description='Analyzes Raman spectrography data',
    )
    parser.add_argument(
        'spectrum',
        nargs='+',
        help=(
            'If given, data is read from SPECTRUM_FILE, which is interpreted as a Spectrum '
            'Studio CSV of analyte spectrum.'
        ),
    )
    parser.add_argument(
        '--blank',
        dest='blanks',
        nargs='+',
        help=(
            'Path to Spectrum Studio CSV of blank spectrum, used in blank subtraction if given. '
            'If specified multiple times, the normalized average is used as the blank.'
        ),
        # action='append',
    )
    parser.add_argument(
        '--blank-factor',
        type=pos_float,
        help=(
            'Scaling factor for blank subtraction. Larger number -> harsher subtraction. Defaults '
            f'to {DEFAULT_BLANK_SUB_FAC}.'
        ),
    )
    parser.add_argument(
        '--hide',
        dest='hide_peak_classes',
        action='append',
        choices=['weak', 'medium', 'strong'],
        default=[],
        help='Hides a peak classification',
    )
    parser.add_argument(
        '-o', '--output',
        help='Output directory. Defaults to saving in same folder as input files.'
    )
    args = parser.parse_args()
    if args.blank_factor is not None and not args.blanks:
        parser.error(
            'Cannot specify blank factor when not using blank subtraction. '
            'Specify at least one --blank.'
        )
    if args.blank_factor is None:
        args.blank_factor = DEFAULT_BLANK_SUB_FAC

from matplotlib.lines import Line2D
from scipy import signal, sparse
from scipy.ndimage import gaussian_filter1d
from scipy.signal import hilbert, firwin, lfilter
from scipy.sparse.linalg import spsolve
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pywt
import sys
from spectrometer_decode import read_spectrometer, find_port
from pptx import Presentation
from pptx.util import Inches, Pt

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
        return cls(wavelengths, intensities)

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

    def subtract_blanks(self, blanks, factor):
        valid_blanks = [blank for blank in blanks if blank is not None]
        if not valid_blanks:
            print("No valid blanks provided")
            return
        if any(np.any(blank.wavenumbers != self.wavenumbers) for blank in valid_blanks):
            raise ValueError('Wavenumber lists of operands do not match')
        
        #normalizing check - if any blank is not max-normalized, raise error.
        for blank in valid_blanks:
            if abs(blank.intensities.max() - 1.0) > 1e-5:
                raise ValueError('Blanks must be max-normalized')

        avg_blank_intensities = np.mean([blank.intensities for blank in valid_blanks], axis=0)
        self.intensities = np.maximum(self.intensities - avg_blank_intensities * factor, 0)

    def find_peaks(self):
        signal_dir = np.sign(np.diff(self.intensities))
        signal_dir = np.insert(signal_dir, 0, signal_dir[1])
        extr_mask = (signal_dir[:-1] != signal_dir[1:]) & (signal_dir[1:] != 0)
        extr_mask = np.append(extr_mask, False)
        extr_int = self.intensities[extr_mask]
        extr_type = signal_dir[extr_mask] # 1 for maxima, -1 for minima
        max_idx = np.argwhere(extr_type == 1)
        max_idx = max_idx[~np.isin(max_idx, [0, len(extr_int) - 1])]
        side_avg_heights = np.mean(extr_int[max_idx] - extr_int[[max_idx - 1, max_idx + 1]], axis=0)
        return np.argwhere(extr_mask).flat[max_idx], {
            'side_avg_heights': side_avg_heights,
            'global_avg_height': np.mean(side_avg_heights),
        }

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
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            lines = []
            ax1.plot(self.wavelengths, self.initial_intensities, '-', linewidth=1, alpha=0.7)
            ax1.set_xlabel('Wavelength (nm)')
            ax1.set_ylabel('Intensity (a.u.)')
            ax1.set_title('Original Spectrum')
            ax1.grid(True, alpha=0.3)

        lines.append(ax2.plot(self.wavenumbers, self.intensities, '-', linewidth=1.5, label=label)[0])
        peaks, properties = self.find_peaks()
        classifications = self.classify_peaks(peaks, properties)

        colors = {'strong': 'darkgreen', 'medium': 'orange', 'weak': 'lightblue'}
        for peak, classification in zip(peaks, classifications):
            if classification in args.hide_peak_classes:
                continue
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

    def classify_peaks(self, _peaks, properties):
        return [
            'strong' if sah > 1.7 * properties['global_avg_height'] else
            'medium' if sah > 1.5 * properties['global_avg_height'] else
            'weak'
            for sah in properties['side_avg_heights']
        ]

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

def generate_full_report(prs, spectrum_obj, output_basename, standard_graph, blank_sub_obj=None, blank_graph=None):
    """
    The master reporting function. 
    Creates the presentation, adds the data, and saves the file.
    """
    #title slide
    slide = prs.slides.add_slide(prs.slide_layouts[5])
    slide.shapes.title.text = f"Raman Analysis Report: {output_basename}"

    #internal helper to add content slides
    def add_content(prs_internal, denoiser, title_prefix, graph_path):
        # Graph Slide
        s1 = prs_internal.slides.add_slide(prs_internal.slide_layouts[5])
        s1.shapes.title.text = f"{title_prefix}: Spectrum Graph"
        s1.shapes.add_picture(graph_path, Inches(0.5), Inches(1.5), width=Inches(9))
        
        #table
        s2 = prs_internal.slides.add_slide(prs_internal.slide_layouts[5])
        s2.shapes.title.text = f"{title_prefix}: Peak Data"
        
        peaks = denoiser.find_all_peaks_unbiased()[:10]
        table = s2.shapes.add_table(len(peaks)+1, 3, Inches(1), Inches(1.5), Inches(8), Inches(4)).table
        
        table.cell(0, 0).text = "Shift (cm⁻¹)"
        table.cell(0, 1).text = "Intensity"
        table.cell(0, 2).text = "Relative %"
        
        for i, p in enumerate(peaks):
            table.cell(i+1, 0).text = f"{p['wavenumber']:.1f}"
            table.cell(i+1, 1).text = f"{p['intensity']:.2f}"
            table.cell(i+1, 2).text = f"{p['relative_intensity']:.1%}"

    #add the Standard Data
    add_content(prs, spectrum_obj, "Standard Analysis", standard_graph)
    #add the Blank Subtracted Data (if applicable))
    if blank_sub_obj and blank_graph:
        add_content(prs, blank_sub_obj, "Blank Subtracted", blank_graph)

if __name__ == "__main__":
    #presentation creation
    args = parser.parse_args()

    prs = Presentation()
    title_slidet = prs.slides.add_slide(prs.slide_layouts[0])
    title_slidet.shapes.title.text = "Spectrum Analysis Report"
    
    #load blanks before loop starts
    blanks = []
    if args.blanks:
            for blank_path in args.blanks:
                print(f"Loading blank spectrum: {blank_path}")
                
                blank_item = RamanDenoiser.from_csv(
                    blank_path,
                    wavenumber_col=1,
                    intensity_col=3,
                    skiprows=5
                )
                raman_analysis(blank_item)
                blanks.append(blank_item)

    #loop through every file
    spectrum_files = args.spectrum if isinstance(args.spectrum, list) else [args.spectrum]

    for spec_file in spectrum_files: 
        print(f"\nProcessing: {spec_file}")

        path_obj = pathlib.Path(spec_file)
        current_basename = path_obj.stem

        if args.output:
            output_dir = pathlib.Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path_prefix = str(output_dir / current_basename)
        else:
            output_path_prefix = current_basename
        
        spectrum = RamanDenoiser.from_csv(
            spec_file,
            wavenumber_col=1,
            intensity_col=3,
            skiprows=5
        )
        
        raman_analysis(spectrum)
        fig, axs, lines = spectrum.plot_comparison(label="Standard processing")
        
        path1 = output_path_prefix + '-standard.png'
        fig.tight_layout()
        fig.savefig(path1, dpi=300, bbox_inches='tight')
        print(f"Saved standard figure to {path1}")

        #blank subtraction and graphing
        blank_subtracted = None 
        path2 = None

        if blanks:
            blank_subtracted = spectrum.clone()
            factor = args.blank_factor if args.blank_factor is not None else 1.0
            blank_subtracted.subtract_blanks(blanks, factor)
            blank_subtracted.plot_comparison(fig_axs=(fig, axs, lines), label="Blank subtracted")

            #updated graph with both lines saved
            path2 = output_path_prefix + '-blank-subtracted.png'
            fig.tight_layout()
            fig.savefig(path2, dpi=300, bbox_inches='tight')
            print(f"Saved blank-subtracted figure to {path2}")
        
        #generating report
        generate_full_report(prs, spectrum, current_basename, path1, blank_subtracted, path2)

        spectrum.save_to_file(output_path_prefix + '-denoised.csv')
        plt.close(fig)
        
    final_pptx = "report.pptx"
    if args.output:
        final_pptx = str(pathlib.Path(args.output) / final_pptx)

    prs.save(final_pptx)
