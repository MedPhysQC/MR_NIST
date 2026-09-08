#!/usr/bin/env python
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# This code is an analysis module for WAD-QC 2.0: a server for automated 
# analysis of medical images for quality control.
#
# The WAD-QC Software can be found on 
# https://codeberg.org/MedPhysNL/wadqc
# 
# The analysis modules can be found on
# https://github.com/MedPhysQC


"""
Analysis of MRI images of the NIST phantom

This is an analysis module for QC of MRI scanners. The intended use is 
the automated analysis of geometric distortion using images of the NIST phantom. 
Metrics include 3D spatial displacement (Procrustes rigid body registration), 
gradient scaling factors, maximum and mean deviations (Delta R), and linear vs. 
non-linear distortions.

input: study containing two series (e.g., reversed readout gradients like RL and LR) 
       targeted at geometric QA.

Functionality is implemented across three main Python scripts developed in Spyder: 
'main.py' handles the WAD-QC arguments, orchestration, and JSON generation. 
'template_matching.py' performs sub-pixel fiducial marker detection (56 markers) 
using Fourier-based template matching and 2D Taylor expansion. 'boekhouding.py' 
executes the mathematical Procrustes analysis and generates comprehensive outputs 
including an Excel report, 3x3 distortion plots, interactive 3D HTML models, 
and histograms.

Limitations:
Generally, the first series that matches the filter criteria in the JSON config 
is used for input. Consequently, if multiple versions of test data series exist 
in the same study folder, only a single series will be analysed. Furthermore, 
the module currently relies on the fixed geometric ground truth of the 56-marker 
NIST phantom layout and requires valid DICOM spatial tags (ImagePositionPatient 
and ImageOrientationPatient) to function correctly.

Changelog:
    20260611: initial version of the Python module, written by
              A Bachmid (VU/Amsterdam UMC) with supervision from 
              J Kuijer (Amsterdam UMC).
    20260902: JK
    - use pyWADinput() to get the WAD-QC input data and results objects.
    - use results object to write the output files and metadata to WAD-QC.
    - removed degrees symbol from print statement to avoid log file break in WAD-QC.
    - various translations of results output to English.
    - minor code cleanup and added comments.
    20260907: JK
    - more translations of code comments and output.

"""

__version__ = '20260907'
__author__ = 'jkuijer, abachmid'

#import logging
from datetime import datetime
from wad_qc.module import pyWADinput
import os
import pydicom
import glob
import zipfile
from template_matching import run_template_matching
from boekhouding import run_boekhouding


def vind_dicoms_voor_series(hoofdmap, gezochte_series_naam):
    """
    Search all subdirectories and collect only the DICOM files with the matching SeriesDescription.
    """
    gevonden_bestanden = []
    print(f"  -> Searching for scans with label: '{gezochte_series_naam}'...")
    
    for root, dirs, files in os.walk(hoofdmap):
        for file in files:
            pad = os.path.join(root, file)
            try:
                # stop_before_pixels=True makes reading very fast; only the label is needed
                dcm = pydicom.dcmread(pad, stop_before_pixels=True)
                if hasattr(dcm, 'SeriesDescription') and dcm.SeriesDescription == gezochte_series_naam:
                    gevonden_bestanden.append(pad)
            except Exception:
                #pass # Invalid DICOM file; ignore it
                print(f"  -> Error reading file: {pad}")
    print(f"  -> {len(gevonden_bestanden)} files found.")
    return gevonden_bestanden

def haal_dicom_info_op(dicom_pad):
    """Read the required WAD-QC metadata and datetime directly from the DICOM header."""
    dcm = pydicom.dcmread(dicom_pad, stop_before_pixels=True)
    
    # 1. Determine the WAD-QC-compliant DateTime (YYYY-MM-DD HH:MM:SS)
    date = str(getattr(dcm, 'AcquisitionDate', getattr(dcm, 'StudyDate', '19700101')))
    time = str(getattr(dcm, 'AcquisitionTime', getattr(dcm, 'StudyTime', '000000')))
    try:
        dt_format = f"{date[:4]}-{date[4:6]}-{date[6:8]} {time[:2]}:{time[2:4]}:{time[4:6]}"
    except:
        dt_format = "1970-01-01 00:00:00"

    # 2. Retrieve the other fields
    info = {
        "AcquisitionDateTime": dt_format,
        "StudyDescription": str(getattr(dcm, 'StudyDescription', 'Onbekend')),
        "SeriesDescription": str(getattr(dcm, 'SeriesDescription', 'Onbekend')),
        "Manufacturer": str(getattr(dcm, 'Manufacturer', 'Onbekend')),
        "ManufacturerModelName": str(getattr(dcm, 'ManufacturerModelName', 'Onbekend')),
        "DeviceSerialNumber": str(getattr(dcm, 'DeviceSerialNumber', 'Onbekend')),
        "StationName": str(getattr(dcm, 'StationName', 'Onbekend')),
        "SoftwareVersions": str(getattr(dcm, 'SoftwareVersions', 'Onbekend'))
    }
    return info

def tel_gevonden_bollen(bol_data_dict):
    """Count number of markers in the dictionary that were measured on all three axes."""
    teller = 0
    for bol_nr, coords in bol_data_dict.items():
        if len(coords['Coronaal']) > 0 and len(coords['Sagittaal']) > 0 and len(coords['Axiaal']) > 0:
            teller += 1
    return teller

def _parse_dt(value):
    if not value:
        return datetime(1970, 1, 1)
    try:
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime(1970, 1, 1)


def main():
    data, results, config = pyWADinput()

    params = config["actions"]["NIST_phantom_analysis"]["params"]
    series1_naam = params["NIST_series1_description"]
    series2_naam = params["NIST_series2_description"]
    ref_x = float(params["NIST_refPhantomDistX"])
    ref_y = float(params["NIST_refPhantomDistY"])
    ref_z = float(params["NIST_refPhantomDistZ"])

    # WAD-QC reads everything as text. Check whether "True" was entered.
    maak_bollen = params.get("NIST_output_all_individual_markers", "False") == "True"
    fixed_axes = params.get("NIST_fixed_plot_axes", "False") == "True"

    print("=== STARTING AUTOMATED MARKER DETECTION ===")

    # 3. Find the correct images in the directory structure via ModuleData
    series_1 = data.getSeriesByDescription(series1_naam, stop_before_pixels=True)
    series_2 = data.getSeriesByDescription(series2_naam, stop_before_pixels=True)
    bestanden_1 = []
    bestanden_2 = []

    for series in series_1:
        for ds in series:
            filename = getattr(ds, 'filename', None)
            if filename:
                bestanden_1.append(filename)
    if not bestanden_1:
        print(f"[ERROR] Not enough files found for the specified SeriesDescription(s): '{series1_naam}'.")

    for series in series_2:
        for ds in series:
            filename = getattr(ds, 'filename', None)
            if filename:
                bestanden_2.append(filename)
    if not bestanden_2:
        print(f"[ERROR] Not enough files found for the specified SeriesDescription(s): '{series2_naam}'.")

    if not bestanden_1 or not bestanden_2:
        print("[ERROR] Not enough files found for both series. Check the SeriesDescription values in the config.")
        return

    output_map = os.path.dirname(results._out_path)

    # 4. Run Template Matching
    print("\n>>> STAP 1: Template Matching Dataset 1...")
    map_1, _, data_1, view_1 = run_template_matching(bestanden_1, series1_naam, output_map, maak_plaatjes=maak_bollen)

    print("\n>>> STAP 2: Template Matching Dataset 2...")
    map_2, _, data_2, view_2 = run_template_matching(bestanden_2, series2_naam, output_map, maak_plaatjes=maak_bollen)

    # 5. Bookkeeping
    print("\n>>> STEP 3: Bookkeeping and Analysis...")
    excel_pad, png_pad, html_pad, int_plot_pad, lin_plot_pad, hist_r, hist_r_lin, wad_cijfers, timestamp = run_boekhouding(
        series1_naam, data_1, series2_naam, data_2, output_map, ref_x, ref_y, ref_z, fixed_axes
    )

    print("\n>>> STEP 4: Saving results via WAD-QC ModuleResults...")

    # 7A. Add metadata for Series 1
    meta_1 = haal_dicom_info_op(bestanden_1[0])
    results.addDateTime("AcquisitionDateTime", _parse_dt(meta_1["AcquisitionDateTime"]))
    for k, v in meta_1.items():
        if k != "AcquisitionDateTime":
            results.addString(f"Series1_{k}", str(v))

    # 7B. Add metadata for Series 2 (DateTime is written only once)
    meta_2 = haal_dicom_info_op(bestanden_2[0])
    results.addDateTime("Series2_AcquisitionDateTime", _parse_dt(meta_2["AcquisitionDateTime"]))
    for k, v in meta_2.items():
        if k != "AcquisitionDateTime":
            results.addString(f"Series2_{k}", str(v))

    # 7C. Add geometric quality results
    results.addFloat("Series1_Markers_Detected", tel_gevonden_bollen(data_1))
    results.addFloat("Series2_Markers_Detected", tel_gevonden_bollen(data_2))
    results.addString("Series1_Orientation", str(view_1))
    results.addString("Series2_Orientation", str(view_2))

    # 7D. Add calculated values (floats)
    for naam, waarde in wad_cijfers.items():
        results.addFloat(naam, waarde)

    # 7E. Add generated objects (Excel/PNG)
    results.addObject("Excel_Report", excel_pad)
    results.addObject("Error_Plot", png_pad)
    results.addObject("Error_Plot_+_Linear_Correction", lin_plot_pad)
    results.addObject("Histogram_Delta_R", hist_r)
    results.addObject("Histogram_Delta_R_+_Linear_Correction", hist_r_lin)
    results.addObject("Interactive_Error_Plot", int_plot_pad)

    # --- Find the template overview images in the subdirectories and add them ---
    # Search subdirectory 1 (use * for the unknown timestamp)
    overzicht_1 = glob.glob(os.path.join(map_1, "result template overview *.png"))
    if overzicht_1:
        results.addObject(f"Template_Overview_{series1_naam}", overzicht_1[0])

    # Search subdirectory 2
    overzicht_2 = glob.glob(os.path.join(map_2, "result template overview *.png"))
    if overzicht_2:
        results.addObject(f"Template_Overview_{series2_naam}", overzicht_2[0])

    # --- Zip the individual markers when the option is set to True ---
    if maak_bollen:
        print("\n>>> STEP 5: Packaging individual marker images in ZIP...")
        zip_naam = f"output_individual_markers_{timestamp}.zip"
        zip_pad = os.path.join(output_map, zip_naam)
        
        # Open a new ZIP file for writing (ZIP_DEFLATED provides compression)
        with zipfile.ZipFile(zip_pad, 'w', zipfile.ZIP_DEFLATED) as zipf:
            
            # Folder 1: Find all images beginning with 'Bol ' and place them in 'QC_NIST_COR_RL'
            for bol_img in glob.glob(os.path.join(map_1, "Bol *.png")):
                bestandsnaam = os.path.basename(bol_img)
                # 'arcname' specifies the path of the file INSIDE the ZIP archive
                zipf.write(bol_img, arcname=os.path.join(series1_naam, bestandsnaam))
                
            # Folder 2: Find all images and place them in 'QC_NIST_COR_LR'
            for bol_img in glob.glob(os.path.join(map_2, "Bol *.png")):
                bestandsnaam = os.path.basename(bol_img)
                zipf.write(bol_img, arcname=os.path.join(series2_naam, bestandsnaam))

        results.addObject("output_individual_markers", zip_pad)

    results.write()
    print(f"=== WAD-QC PROCESS COMPLETED! Results saved to: {results._out_path} ===")

if __name__ == "__main__":
    print(f"=== WAD-QC NIST MODULE VERSION {__version__} ===")
    main()
