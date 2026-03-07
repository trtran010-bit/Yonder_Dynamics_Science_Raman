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

class RamanDenoiser:
    def __init__(self, wavenumbers=None, intensities=None):
        if wavenumbers is not None and intensities is not None:
            self.wavenumbers = np.array(wavenumbers)
            self.intensities = np.array(intensities)
        else:
            self.wavenumbers = None
            self.intensities = None
        self.processed = self.intensities
        self.processed_wavenumbers = self.wavenumbers
        self.baseline = None

    def clone(self):
        '''Returns a deep copy of the denoiser in its current state.'''
        new = type(self)()
        new.wavenumbers = self.wavenumbers.copy()
        new.intensities = self.intensities.copy()
        new.processed = self.processed.copy()
        new.processed_wavenumbers = self.processed_wavenumbers.copy()
        new.baseline = self.baseline.copy()
        return new
    
    @classmethod
    def from_csv(cls, filepath, wavenumber_col=0, intensity_col=1, skiprows=0):
        try:
            #pandas
            df = pd.read_csv(filepath, skiprows=skiprows)
            df = df[df.iloc[:,wavenumber_col] > 540]
            
            if isinstance(wavenumber_col, int):
                wavenumbers = df.iloc[:, wavenumber_col].values
            else:
                wavenumbers = df[wavenumber_col].values
            incident_wn = 10**7 / 532
            wavenumbers = incident_wn - 10**7 / wavenumbers
                
            if isinstance(intensity_col, int):
                intensities = df.iloc[:, intensity_col].values
            else:
                intensities = df[intensity_col].values
            
            print(f"Successfully loaded {len(wavenumbers)} data points from {filepath}")
            return cls(wavenumbers, intensities)
            
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return None
    
    def savitzky_golay(self, window_length=11, polyorder=3):
        if window_length % 2 == 0:
            window_length += 1
        self.processed = signal.savgol_filter(
            self.intensities, 
            window_length, 
            polyorder
        )
    
    def gaussian_filter(self, sigma=2):
        self.processed = gaussian_filter1d(self.intensities, sigma)
    
    def median_filter(self, kernel_size=5):
        self.processed = signal.medfilt(self.intensities, kernel_size)
    
    def wiener_filter(self, noise_power=None):
        self.processed = signal.wiener(self.intensities, mysize=5, noise=noise_power)
    
    def fir_filter(self, cutoff_freq=0.1, numtaps=51, window='hamming'):
        fir_coeff = firwin(numtaps, cutoff_freq, window=window)
        
        self.processed = lfilter(fir_coeff, 1.0, self.processed)
    
    def hilbert_vibration_decomposition(self, num_components=3):
        #hilbert transform method
        analytic_signal = hilbert(self.processed)
        amplitude_envelope = np.abs(analytic_signal)
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) / (2.0 * np.pi)

        #storing denoiser signal
        self.processed = amplitude_envelope
        
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
            level = int(np.log2(len(self.processed))) - 1
        
        #wavelet decomposition
        coeffs = pywt.wavedec(self.processed, wavelet, level=level)
        
        #calculate threshold using MAD (Median Absolute Deviation)
        sigma = np.median(np.abs(coeffs[-level])) / 0.6745
        threshold = sigma * np.sqrt(2 * np.log(len(self.processed)))
        
        coeffs_thresh = [coeffs[0]]
        for i in range(1, len(coeffs)):
            coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode=threshold_mode))

        # recounstruction
        self.processed = pywt.waverec(coeffs_thresh, wavelet)[:len(self.processed)]
    
    def als_baseline(self, lam=1e6, p=0.01, niter=10):
        L = len(self.processed)
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        w = np.ones(L)
        
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            Z = W + lam * D.dot(D.transpose())
            baseline = spsolve(Z, w * self.processed)
            w = p * (self.processed > baseline) + (1 - p) * (self.processed < baseline)
        
        self.baseline = baseline
        self.processed = self.processed - baseline
    
    def polynomial_baseline(self, degree=3):
        coeffs = np.polyfit(self.wavenumbers, self.processed, degree)
        baseline = np.polyval(coeffs, self.wavenumbers)
        self.baseline = baseline
        self.processed = self.processed - baseline
    
    def normalize(self, method='max'):
        if method == 'max':
            self.processed = self.processed / np.max(self.processed)
        elif method == 'area':
            self.processed = self.processed / np.trapz(self.processed, self.wavenumbers)
        elif method == 'minmax':
            self.processed = (self.processed - np.min(self.processed))
                / (np.max(self.processed) - np.min(self.processed))

    def trim(self, low=None, high=None):
        if low == None:
            low = float('-inf')
        if high == None:
            high = float('inf')
        mask = low < self.wavenumbers
        self.processed = self.processed[mask]
        self.processed_wavenumbers = self.wavenumbers[mask]

    def subtract_blank(self, blank, factor=2):
        if self.processed_wavenumbers != blank.processed_wavenumbers:
            raise ValueError('Wavenumber lists of operands do not match')
        self.processed = np.maximum(self.processed - blank.processed * factor, 0)
    
    def find_peaks(self, prominence=None, distance=10, height=None, width=None, auto_adapt=True):
        # find peaks in the spectrum
        # if auto_adapt is True, it'll try to figure out good parameters for that material
        if auto_adapt and prominence is None:
            # calculate adaptive prominence based on noise level
            # using the standard deviation of the baseline region as noise estimate
            noise_std = np.std(self.processed[:int(len(self.processed)*0.1)])  
            prominence = max(3 * noise_std, 0.05 * np.max(self.processed))  
            print(f"auto-detected prominence: {prominence:.4f}")
            
        if auto_adapt and height is None:
            # set minimum height as mean + 2*std
            height = np.mean(self.processed) + 2 * np.std(self.processed)
            print(f"auto-detected height threshold: {height:.4f}")
                
        peaks, properties = signal.find_peaks(
            self.processed, prominence=prominence, distance=distance, height=height, width=width
        )
        
        return peaks, properties

    # testing new method
    def find_all_peaks_unbiased(self, min_prominence_ratio=0.01, min_distance=5):
        # completely unbiased peak detection so we can find ALL local maxima above minimal threshold
        # min_prominence_ratio: fraction of max intensity (e.g., 0.01 = 1% of max signal)
        # this way you see everything and can decide what matters for your material
        min_prominence = min_prominence_ratio * np.max(self.processed)
        
        peaks, properties = signal.find_peaks(
            self.processed,
            prominence=min_prominence,
            distance=min_distance
        
        )
        
        peak_data = []
        for i, peak_idx in enumerate(peaks):
            peak_data.append({
                'index': peak_idx,
                'wavenumber': self.wavenumbers[peak_idx],
                'intensity': self.processed[peak_idx],
                'prominence': properties['prominences'][i],
                'relative_intensity': self.processed[peak_idx] / np.max(self.processed)
            })
        
        peak_data_sorted = sorted(peak_data, key=lambda x: x['intensity'], reverse=True)
        
        return peak_data_sorted

    def plot_comparison(self, title="Raman Spectrum Processing", show_peak_labels=True):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        ax1.plot(self.wavenumbers, self.intensities, 'b-', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Raman Shift (cm⁻¹)')
        ax1.set_ylabel('Intensity (a.u.)')
        ax1.set_title('Original Spectrum')
        ax1.grid(True, alpha=0.3)
        
        if self.processed is not None:
            ax2.plot(self.processed_wavenumbers, self.processed, 'r-', linewidth=1.5)
            
            try:
                peaks, properties = self.find_peaks(auto_adapt=True)
                classifications = self.classify_peaks(peaks, properties)
                
                colors = {'strong': 'darkgreen', 'medium': 'orange', 'weak': 'lightblue'}
                for peak, classification in zip(peaks, classifications):
                    ax2.plot(self.wavenumbers[peak], self.processed[peak], 'o', 
                            color=colors[classification], markersize=8)
                    
                    if show_peak_labels and classification in ['strong', 'medium']:
                        ax2.text(self.wavenumbers[peak], self.processed[peak], 
                               f'{self.wavenumbers[peak]:.0f}',
                               fontsize=8, ha='center', va='bottom')
                
                # legend
                legend_elements = [
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='darkgreen', 
                          markersize=8, label='Strong peaks'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', 
                          markersize=8, label='Medium peaks'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue', 
                          markersize=8, label='Weak peaks')
                ]
                ax2.legend(handles=legend_elements, loc='upper right')
            except Exception as e:
                print(f"couldn't detect peaks: {e}")
        else:
            ax2.plot(self.wavenumbers, self.intensities, 'r-', linewidth=1.5)
        
        ax2.set_xlabel('Raman Shift (cm⁻¹)')
        ax2.set_ylabel('Intensity (a.u.)')
        ax2.set_title('Processed Spectrum')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def classify_peaks(self, peaks, properties):
        return ['strong' for _ in peaks]
    
    def save_processed(self, filepath):
        df = pd.DataFrame({
            'wavenumber': self.wavenumbers,
            'intensity': self.processed
        })
        df.to_csv(filepath, index=False)
        print(f"saved processed data to {filepath}")

def raman_analysis(denoiser):
    denoiser.als_baseline(lam=1e5, p=0.01)
    denoiser.fir_filter(cutoff_freq=0.1, numtaps=51)
    #denoiser.wavelet_denoise(wavelet='db4', threshold_mode='soft', level=3)
    denoiser.trim(low=1000)
    denoiser.normalize(method='max')
    #peaks, properties = denoiser.find_peaks(prominence=0.1, distance=20)
    #print(f"Found {len(peaks)} peaks")

if __name__ == "__main__":
    #load csv
    spectrum = RamanDenoiser.from_csv(
        sys.argv[1],
        wavenumber_col=1,
        intensity_col=3,
        skiprows=5
    )
    blank = RamanDenoiser.from_csv(
        sys.argv[2],
        wavenumber_col=1,
        intensity_col=3,
        skiprows=5
    )

    raman_analysis(spectrum)
    raman_analysis(blank)
    spectrum.subtract_blank(blank)

    #plot less noisy plot
    spectrum.plot_comparison()
    spectrum.save_processed(sys.argv[1][:-4] + '-processed.csv')
