# Yonder_Dynamics_Science_Raman
The post processing for Raman experiement for Yonder Dynamics. 

It incorporates: 
Baseline corrections: takes out significant amounts of residual Rayleigh scattering or fluorescence: Asymmetric Least Squares (ALS) and SmoothingPolynomial Fitting

Digital Filtering/Smoothing (4): 
1. Savisky Golay -  fits low-degree polynomials to adjacent data points
2. Finite Impulse Response (FIR) Filter - cuts frequencies at a cut off point 
3. Gaussian Smoothing - convolution based operator to reduce graininess 
4. Wiener Filter - performs 1D noise reduction 

Decomposition (2): 
1. PyWavelet - decomposes signal into approx coefficient by using threshold, Median Absolute Deviation (MAD), to mute noise while still keeping structural features 
2. Hilbert Vibration Decomposition (HVD) (not sure if we adding this cause it doesn't work)

Peak detection and Classification (3): 
1. Local maxima - changes in the first derivative to find where signal turns around 
2. Topological Prominence - peak standing out from surrounding baseline uses scipy 
3. Heuristic Classification - labels peak as strong med and weak based on height to global avg height

Blank subtraction - record the cuvette signal then subtract it from analysis signal to remove systematic interference. 
