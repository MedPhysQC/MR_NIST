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
See [Wiki](../../wiki) for documentation.

## Scan protocol
See [Wiki](../../wiki) for more information.
Summary:
- For reliable marker detection, carefully set up the phantom. Use the spirit level insert, keep bubble within the ring. Respect I/S and R/L markers in the phantom. Avoid rotation around A/P axis. If the phantom has substantial misalignments the markers may not be detected accurately.
- head coil
- 3D T1-w scan (MP-RAGE, IR-FSPGR)
- any orthogonal scan orientation is allowed: SAG, COR or TRA. Do not angulate the scan plane
- 1x1x1 mm spatial resolution
- approximately 5 mins scan time per series should yield appropriate SNR
- scan two series with reversed phase encoding (Siemens) / reversed fat shift direction (Philips) that have exactly identical slice positions
