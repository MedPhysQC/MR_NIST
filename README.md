# MR_NIST (fiducial marker geometry)
 MRI module for WAD-QC 2.0 for analysis of the fiducial markers of the NIST phantom. Reports the non-linear geometric distortion of a MR scanner by the NIST phantom.

## Python dependencies
To install additional dependency of the openpyxl package:
```
workon wad2env3
pip3 install openpyxl
```

In a development setup you may need to install scikit-image plotly as well

## Phantom
This module is intended to detect the position of fiducial markers in the ISMRM/NIST system phantom, also known as HPD or QalibreMD/CaliberMRI phantom. Note that the phantom has various further features such as T1, T2 and PD spheres, a resolution insert etc, but these are ignored by the current version of this MR_NIST module.

## Concept of this module
The phantom contains fiducial markers on a 3D grid with 40 mm spacing. The module detects a fiducial marker in 2D by template matching. This process is repeated in three orthogonal projections. The 3x 2D positions are then combined into 3D positions, and errors relative to the 40 mm design spec is calculated. Additionally, a linear fit to the measured errors yields both the gradient amplitude calibration error (linear distortion / scaling error) and estimates residual non-linear distortion for a perfectly calibrated gradient amplitude.

## Scan protocol
The module has been tested with 3D inversion recovery T1-weighted scans on both 1.5T and 3T scanners. MP-RAGE or IR-FSPGR with a spatial resolution of 1x1x1 mm should work. A head coil typically yield sufficient SNR, and allows for some parallel imaging. Typically a scan time of ~5 mins per series should be fine. Scan orientation may be TRA, SAG or COR, but not oblique. Avoiding any angulation of scan planes aligns the image coordinates with gradient directions, which allows for easier interpretation of results.
The module expects two scans with reversed readout gradient polarity. On a Siemens scanner, this can be set up by reversing the phase encoding direction. E.g., for a SAG scan, choose phase encoding L>>R and R>>L. On a Philips scanner, reversing the fat shift direction reverses the readout gradient polarity. Detected marker positions from both scans are averaged, which effectively cancels any distortion from B0 inhomogeneities.

Summary:
- For reliable marker detection, carefully set up the phantom. Use the spirit level insert, keep bubble within the ring. Respect I/S and R/L markers in the phantom. Avoid rotation around A/P axis. If the phantom has substantial misalignments the markers may not be detected accurately.
- head coil
- 3D T1-w scan (MP-RAGE, IR-FSPGR)
- any orthogonal scan orientation is allowed: SAG, COR or TRA. Do not angulate the scan plane
- 1x1x1 mm spatial resolution
- approximately 5 mins scan time per series should yield appropriate SNR
- scan two series with reversed phase encoding (Siemens) / reversed fat shift direction (Philips) that have exactly identical slice positions


TODO: add more info/documentation
