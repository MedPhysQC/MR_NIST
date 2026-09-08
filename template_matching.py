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

# template_matching.py
# version 20260907

import pydicom
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time
from datetime import datetime
from skimage.feature import canny, match_template
from scipy.signal import convolve2d
from skimage.draw import circle_perimeter
import matplotlib.patches as patches

# ==========================================
# 1. Global marker constants
# ==========================================
# Phantom definition: {marker_number: (plate_index, grid_x, grid_z)}
# This is the ground-truth geometry of the 56 markers in the MRI phantom.
BOL_DEF = {
    1: (0, 0, 1), 2: (0, -1, 0), 3: (0, 1, 0), 4: (0, 0, -1),
    5: (1, 0, 2), 6: (1, -1, 1), 7: (1, 0, 1), 8: (1, 1, 1), 9: (1, -2, 0), 10: (1, -1, 0), 11: (1, 0, 0), 12: (1, 1, 0), 13: (1, 2, 0), 14: (1, -1, -1), 15: (1, 0, -1), 16: (1, 1, -1), 17: (1, 0, -2),
    18: (2, -1, 2), 19: (2, 0, 2), 20: (2, 1, 2), 21: (2, -2, 1), 22: (2, -1, 1), 23: (2, 0, 1), 24: (2, 1, 1), 25: (2, 2, 1), 26: (2, -2, 0), 27: (2, -1, 0), 28: (2, 0, 0), 29: (2, 1, 0), 30: (2, 2, 0), 31: (2, -2, -1), 32: (2, -1, -1), 33: (2, 0, -1), 34: (2, 1, -1), 35: (2, 2, -1), 36: (2, -1, -2), 37: (2, 0, -2), 38: (2, 1, -2),
    39: (3, 0, 2), 40: (3, -1, 1), 41: (3, 0, 1), 42: (3, 1, 1), 43: (3, -2, 0), 44: (3, -1, 0), 45: (3, 0, 0), 46: (3, 1, 0), 47: (3, 2, 0), 48: (3, -1, -1), 49: (3, 0, -1), 50: (3, 1, -1), 51: (3, 0, -2),
    52: (4, 0, 1), 53: (4, -1, 0), 54: (4, 0, 0), 55: (4, 1, 0), 56: (4, 0, -1)
}

# ==========================================
# 2. Standalone mathematical tools
# These functions are independent of the MRI volume, so they remain outside the class and only perform calculations.
# ==========================================
def maak_anti_aliased_stempel(spacing_row, spacing_col, core_val, ring_val):
    # Calculate a dynamic template based on the physical pixel spacing.
    # The marker has a fixed physical diameter of 14 mm.
    size_row = int(round(14.0 / spacing_row))
    size_col = int(round(14.0 / spacing_col))
    
    if size_row % 2 == 0: size_row += 1
    if size_col % 2 == 0: size_col += 1
    
    Y, X = np.ogrid[:size_row, :size_col]
    center_y, center_x = size_row // 2, size_col // 2
    
    # Euclidean distance in millimeters from the center
    dist_mm = np.sqrt(((Y - center_y) * spacing_row)**2 + ((X - center_x) * spacing_col)**2)
    
    # Anti-aliasing: create a smooth transition at the marker core and shell edges.
    core_mask = np.clip(5.0 - dist_mm + 0.5, 0, 1)
    shell_mask = np.clip(6.5 - dist_mm + 0.5, 0, 1) - core_mask
    
    template_raw = np.zeros((size_row, size_col), dtype=np.float32)
    template_raw += core_mask * core_val
    template_raw += shell_mask * ring_val
    
    active_mask = (core_mask + shell_mask) > 0
    
    # Normalize the template (mean = 0) for more robust cross-correlation (Zero-Normalized Cross-Correlation).
    template_norm = np.zeros_like(template_raw)
    mean_active = np.mean(template_raw[active_mask])
    template_norm[active_mask] = template_raw[active_mask] - mean_active
    
    return template_norm, active_mask

def maak_fourier_stempel_methode2_center_sample(spacing_row, spacing_col, img_slice, cx_col, cy_row):
    # Determine the marker-center intensity in the actual MRI data to make the template more realistic.
    cy_int, cx_int = int(round(cy_row)), int(round(cx_col))
    if 0 <= cy_int < img_slice.shape[0] and 0 <= cx_int < img_slice.shape[1]:
        exact_center_pixel_val = img_slice[cy_int, cx_int]
    else:
        exact_center_pixel_val = 1.0 
        
    pitch_black_val = np.min(img_slice) 
    return maak_anti_aliased_stempel(spacing_row, spacing_col, exact_center_pixel_val, pitch_black_val)

def bereken_fourier_match_m2_streng(image_slice, template_norm):
    # Perform the actual template matching. match_template internally uses the Fast Fourier Transform (FFT).
    try:
        return match_template(image_slice, template_norm, pad_input=True)
    except TypeError:
        result = match_template(image_slice, template_norm)
        pad_y = (image_slice.shape[0] - result.shape[0]) // 2
        pad_x = (image_slice.shape[1] - result.shape[1]) // 2
        padded = np.zeros_like(image_slice, dtype=np.float32)
        padded[pad_y:pad_y+result.shape[0], pad_x:pad_x+result.shape[1]] = result
        return padded

def plot_exact_template_overlay(ax, active_mask, px, py, rgb_color):
    # Helper function to plot the detected template over the MRI slice with an alpha channel (transparency).
    rows, cols = active_mask.shape
    rgba_overlay = np.zeros((rows, cols, 4))
    rgba_overlay[active_mask] = [*rgb_color, 0.35] 
    
    left = px - cols/2.0
    right = px + cols/2.0
    top = py - rows/2.0 
    bottom = py + rows/2.0
    
    ax.imshow(rgba_overlay, extent=[left, right, bottom, top], interpolation='none')

# ==========================================
# 3. CLASS: MriAnalyse
# ==========================================
class MriAnalyse:
    """
    This class contains the data for one MRI scan and all functions needed to create images or overviews for that scan.
    """
    def __init__(self, dicom_lijst, dataset_naam="WAD-QC_Dataset"):
        # Store basic data
        self.pad = "WAD-QC geautomatiseerd"
        self.dataset_naam = dataset_naam
        
        print(f"\nLoading volume for: {self.dataset_naam}...")
        
        # Load the data directly and store it on 'self.'
        self.volume, self.acquired_view, self.spacings = self._load_robust_volume(dicom_lijst)
        
        # If loading succeeds, immediately calculate the anchor point (the phantom center).
        if self.volume is not None:
            self.ankerpunt = self._bereken_ankerpunt()
        else:
            self.ankerpunt = None
            
# --- Internal data functions (note the 'self') ---
    def _load_robust_volume(self, files):
        # Read DICOM files and reconstruct a 3D NumPy array in a standardized orientation.
        if not files: return None, None, None
        ref_dcm = pydicom.dcmread(files[0])
        orient = ref_dcm.ImageOrientationPatient 
        row_vec = np.array(orient[:3]) 
        col_vec = np.array(orient[3:])
        normal_vec = np.cross(row_vec, col_vec)
        axis_idx = np.argmax(np.abs(normal_vec))

        if axis_idx == 2: acquired_view = 'Axial'
        elif axis_idx == 1: acquired_view = 'Coronal'
        else: acquired_view = 'Sagittal'
    
        # Sort the slices physically in space using ImagePositionPatient.
        files.sort(key=lambda x: pydicom.dcmread(x).ImagePositionPatient[axis_idx])
        datasets = [pydicom.dcmread(f) for f in files]
        volume_acquired = np.stack([d.pixel_array for d in datasets])

        # Calculate the physical distance between slices (Z-spacing).
        if len(files) > 1:
            pos1 = pydicom.dcmread(files[0]).ImagePositionPatient
            pos2 = pydicom.dcmread(files[1]).ImagePositionPatient
            slice_spacing = abs(float(pos1[axis_idx]) - float(pos2[axis_idx]))
            if slice_spacing == 0: slice_spacing = float(getattr(ref_dcm, 'SliceThickness', 1.0))
        else:
            slice_spacing = float(getattr(ref_dcm, 'SliceThickness', 1.0))

         # Determine pixel spacing (X and Y).
        pixel_spacing = getattr(ref_dcm, 'PixelSpacing', [1.0, 1.0])
        spacings_lps = [0.0, 0.0, 0.0]
        spacings_lps[axis_idx] = slice_spacing
        row_axis = np.argmax(np.abs(row_vec))
        col_axis = np.argmax(np.abs(col_vec))
        spacings_lps[row_axis] = float(pixel_spacing[0])
        spacings_lps[col_axis] = float(pixel_spacing[1])
        spacing_x, spacing_y, spacing_z = spacings_lps
    
        # Standardize the matrix to a (Z, Y, X) SHAPE regardless of how the scan was acquired.
        if acquired_view == 'Axial': # volume_acquired is already (Z, Y, X)
            volume_lps = volume_acquired 
        elif acquired_view == 'Coronal': # volume_acquired is (Y, Z, X) -> transpose puts Z at position 0 and Y at position 1
            volume_lps = np.transpose(volume_acquired, (1, 0, 2))
            volume_lps = np.flip(volume_lps, axis=0) 
        else: 
            volume_lps = np.transpose(volume_acquired, (1, 2, 0))  # volume_acquired is (X, Z, Y) -> transpose puts Z at position 0, Y at position 1, and X at position 2
            volume_lps = np.flip(volume_lps, axis=0) # Z-axis is now axis 0 -> flip the Z-axis
        
        print(f'Input: {acquired_view} with pixel spacing in mm (x, y, z) = {spacings_lps}')
        return volume_lps, acquired_view, (spacing_z, spacing_y, spacing_x)

    def _bereken_ankerpunt(self):
        # Find the absolute phantom center to use as the (0,0,0) reference.
        spacing_z, spacing_y, spacing_x = self.spacings
        size_z, size_y, size_x = self.volume.shape
    
        def find_2d_center(projection, spacing_row, spacing_col):
            # Normalize the 2D projection (mean of the 3D array along one axis).
            p_min, p_max = projection.min(), projection.max()
            if p_max > p_min: projection = (projection - p_min) / (p_max - p_min)
            
            # Create a 'donut' mask to detect the outer phantom edge.
            px_spacing = (spacing_row + spacing_col) / 2.0
            radius_px = int(round(5.0 / px_spacing))
            donut_mal = np.zeros((radius_px*2+5, radius_px*2+5), dtype=np.float32)
            rr, cc = circle_perimeter(len(donut_mal)//2, len(donut_mal)//2, radius_px)
            donut_mal[rr, cc] = 1.0
            
            # Use Canny edge detection and convolution to match the donut shape.
            edges = canny(projection, sigma=1.5)
            donut_score = convolve2d(edges.astype(np.float32), donut_mal, mode='same')
            
            # Define the expected pin/marker positions relative to the center.
            p21_offsets = []
            for dc in [-40, 0, 40]:
                for dr in [-80, -40, 0, 40, 80]: p21_offsets.append((dr, dc))
            for dc in [-80, 80]:
                for dr in [-40, 0, 40]: p21_offsets.append((dr, dc))
            
            # Create a mask of the expected grid and match it with the detected donuts.
            marge = int((80.0 / px_spacing) * 1.5)
            c_mal = np.zeros((marge*2+1, marge*2+1))
            for dr_mm, dc_mm in p21_offsets:
                r_idx = marge + int(dr_mm / spacing_row)
                c_idx = marge + int(dc_mm / spacing_col)
                r1, r2 = max(0, r_idx-2), min(c_mal.shape[0], r_idx+3)
                c1, c2 = max(0, c_idx-2), min(c_mal.shape[1], c_idx+3)
                c_mal[r1:r2, c1:c2] = 1.0
                
            c_score = convolve2d(donut_score, c_mal, mode='same')
            return np.unravel_index(np.argmax(c_score), c_score.shape)

        # Determine the 3D anchor point by analyzing 2D projections based on the acquisition orientation.
        if self.acquired_view == 'Sagittal':
            projection = np.mean(self.volume, axis=2)
            best_z, best_y = find_2d_center(projection, spacing_z, spacing_y)
            return int(best_z), int(best_y), int(size_x // 2)
        elif self.acquired_view == 'Coronal':
            projection = np.mean(self.volume, axis=1)
            best_z, best_x = find_2d_center(projection, spacing_z, spacing_x)
            return int(best_z), int(size_y // 2), int(best_x)
        else: 
            projection = np.mean(self.volume, axis=0)
            best_y, best_x = find_2d_center(projection, spacing_y, spacing_x)
            return int(size_z // 2), int(best_y), int(best_x)

# ==========================================
# 3a. Helpers for overview images
# ==========================================
    def _start_anatomie_motor(self, anatomie_override):
        """
        Central engine for determining labels, axes, and viewing direction.
        Prevents this code from being duplicated.
        """
        anatomie = anatomie_override if anatomie_override else {
            'Z': ('I', 'S'), # Z-axis: [0] is feet (Inferior),   [-1] is head (Superior)
            'Y': ('A', 'P'), # Y-axis: [0] is face (Anterior),   [-1] is back of head (Posterior)
            'X': ('R', 'L')  # X-axis: [0] is right ear (Right), [-1] is left ear (Left)
        }

        anatomie_namen = {
            'S': 'superior', 'I': 'inferior',
            'A': 'anterior', 'P': 'posterior',
            'R': 'right', 'L': 'left'
        }
    
        dir_z = 1 if anatomie['Z'] == ('I', 'S') else -1
        dir_y = 1 if anatomie['Y'] == ('A', 'P') else -1
        dir_x = 1 if anatomie['X'] == ('R', 'L') else -1

        def bepaal_labels(v_as, h_as, flip_v=False, flip_h=False):
            top = anatomie[v_as][1] if flip_v else anatomie[v_as][0]
            bot = anatomie[v_as][0] if flip_v else anatomie[v_as][1]
            left = anatomie[h_as][1] if flip_h else anatomie[h_as][0]
            right = anatomie[h_as][0] if flip_h else anatomie[h_as][1]
            return {'T': top, 'B': bot, 'L': left, 'R': right}

        def get_axis_info(letter):
            # Fixed anatomical directions, independent of the array orientation
            vaste_assen = {
                'L': 'x+', 'R': 'x-',
                'P': 'y+', 'A': 'y-',
                'S': 'z+', 'I': 'z-'
            }
            return vaste_assen.get(letter, "??")

        def bepaal_assen_text(labels_dict):
            return f"↑{labels_dict['T']} ({get_axis_info(labels_dict['T'])})\n→{labels_dict['R']} ({get_axis_info(labels_dict['R'])})"
    
        def bepaal_kijkrichting(labels_dict):
            vec = {
                'L': np.array([1, 0, 0]), 'R': np.array([-1, 0, 0]),
                'P': np.array([0, 1, 0]), 'A': np.array([0, -1, 0]),
                'S': np.array([0, 0, 1]), 'I': np.array([0, 0, -1])
            }
            screen_x = vec[labels_dict['R']]
            screen_y = vec[labels_dict['B']]
            
            camera_vec = np.cross(screen_x, screen_y)
            
            for letter, v in vec.items():
                if np.allclose(camera_vec, v):
                    return f"Viewing direction: {get_axis_info(letter)} ({anatomie_namen[letter]})"
            return "Viewing direction: Unknown"
            
        # Return only the values required by the other functions.
        return dir_z, dir_y, dir_x, bepaal_labels, bepaal_assen_text, bepaal_kijkrichting
    
    def _bepaal_titels(self, dir_x, dir_y, dir_z):
        """
        Prevents duplicating the lists of plate and grid-plane names.
        """
        coronaal_info = [("Plaat 5", 4), ("Plaat 4", 13), ("Plaat 3", 21), ("Plaat 2", 13), ("Plaat 1", 5)]
        cross_section_info = [("Gridvlak 5", 5), ("Gridvlak 4", 13), ("Gridvlak 3", 20), ("Gridvlak 2", 13), ("Gridvlak 1", 5)]
    
        sag_titles = cross_section_info if dir_x == 1 else cross_section_info[::-1]
        cor_titles = coronaal_info if dir_y == 1 else coronaal_info[::-1]
        ax_titles  = cross_section_info if dir_z == 1 else cross_section_info[::-1]
        
        return sag_titles, cor_titles, ax_titles

    def _teken_kaders_en_tekst(self, ax, labels, kijkrichting, assen):
        """
        Prevents duplicating the formatting (yellow T, B, L, R letters and text boxes).
        """
        for pos, txt in [('0.5,0.96', 'T'), ('0.5,0.04', 'B'), ('0.04,0.5', 'L'), ('0.96,0.5', 'R')]:
            x_p, y_p = map(float, pos.split(','))
            ax.text(x_p, y_p, labels[txt], transform=ax.transAxes, color='yellow', ha='center', va='center', fontweight='bold', fontsize=13)
        
        ax.text(0.5, -0.15, kijkrichting, transform=ax.transAxes, color='white', ha='center', fontsize=9, style='italic', bbox=dict(facecolor='#333333', alpha=0.7, lw=0))
        ax.text(0.02, 0.02, assen, transform=ax.transAxes, color='cyan', ha='left', va='bottom', fontsize=8, fontweight='bold', bbox=dict(facecolor='black', alpha=0.5, edgecolor='cyan', lw=1))
        ax.axis('off')
        
    def _bepaal_contrast(self):
        """Prevents calculating the same contrast values twice."""
        return np.percentile(self.volume, 1), np.percentile(self.volume, 90)

    def _maak_basis_figuur(self, titel):
        """Create the 3x5 grid with exactly the same formatting for both overviews."""
        fig, axes = plt.subplots(3, 5, figsize=(20, 16.5))
        fig.suptitle(titel, fontsize=20, fontweight='bold', y=0.98, color='black')
        plt.subplots_adjust(hspace=0.2, wspace=0.3, top=0.92)
        return fig, axes

    def _haal_beeld_info(self, view, slice_idx, col, dir_x, dir_y, dir_z, bepaal_labels, sag_titles, cor_titles, ax_titles):
        """
        The central 'slice machine'. Retrieves the correct image, rotates it if needed,
        calculates the display index, and retrieves the specific titles and labels.
        """
        size_z, size_y, size_x = self.volume.shape
        spacing_z, spacing_y, spacing_x = self.spacings

        if view == 'Sagittaal':
            # Calculate the physical (anatomical) display index
            disp_idx = slice_idx if dir_x == 1 else size_x - 1 - slice_idx
            img = np.flipud(self.volume[:, :, slice_idx])
            labels = bepaal_labels('Z', 'Y', flip_v=True, flip_h=False)
            vlak_naam, bollen = sag_titles[col]
            return img, disp_idx, labels, vlak_naam, bollen, 'x', spacing_z, spacing_y
            
        elif view == 'Coronaal':
            # Calculate the physical (anatomical) display index
            disp_idx = slice_idx if dir_y == 1 else size_y - 1 - slice_idx
            img = np.flipud(self.volume[:, slice_idx, :])
            labels = bepaal_labels('Z', 'X', flip_v=True, flip_h=False)
            vlak_naam, bollen = cor_titles[col]
            return img, disp_idx, labels, vlak_naam, bollen, 'y', spacing_z, spacing_x
            
        elif view == 'Axiaal':
            # Calculate the physical (anatomical) display index
            disp_idx = slice_idx if dir_z == 1 else size_z - 1 - slice_idx
            img = self.volume[slice_idx, :, :]
            labels = bepaal_labels('Y', 'X', flip_v=False, flip_h=False)
            vlak_naam, bollen = ax_titles[col]
            return img, disp_idx, labels, vlak_naam, bollen, 'z', spacing_y, spacing_x
    
# ==========================================
# 3b. Function for general overviews (pre-test)
# ==========================================
    # NOTE: 'volume', 'spacings', and 'ankerpunt' are no longer requested as arguments here!
    def genereer_overzicht(self, save_path, bollen_overlay=False, anatomie_override=None):
        # Trick: unpack the class data into the old local names so the code below works as originally written for the class.
        volume = self.volume
        spacings = self.spacings
        ankerpunt = self.ankerpunt

        spacing_z, spacing_y, spacing_x = spacings
        anker_z, anker_y, anker_x = ankerpunt
        # 1. Use the contrast helper
        v_min, v_max = self._bepaal_contrast() 
        size_z, size_y, size_x = volume.shape

        step_x = int(round(40.0 / spacing_x)) if spacing_x > 0 else 40
        step_y = int(round(40.0 / spacing_y)) if spacing_y > 0 else 40
        step_z = int(round(40.0 / spacing_z)) if spacing_z > 0 else 40
        

        # --- CALL THE ANATOMY ENGINE ---
        dir_z, dir_y, dir_x, bepaal_labels, bepaal_assen_text, bepaal_kijkrichting = self._start_anatomie_motor(anatomie_override)

        def get_indices(center, step):
            return [center + i * step for i in [-2, -1, 0, 1, 2]]
        idx_z = get_indices(anker_z, step_z)
        idx_y = get_indices(anker_y, step_y)
        idx_x = get_indices(anker_x, step_x)
    
        views = ['Sagittaal', 'Coronaal', 'Axiaal']
        sag_titles, cor_titles, ax_titles = self._bepaal_titels(dir_x, dir_y, dir_z)
    
        # 2. Determine the title and use the canvas helper
        titel = f"Overview (with assigned fiducial numbers) {self.dataset_naam}" if bollen_overlay else f"Overview {self.dataset_naam}"
        fig, axes = self._maak_basis_figuur(titel)
    
        for row, view in enumerate(views):
            for col in range(5):
                ax = axes[row, col]
                
                # We only determine the 'slice_idx' here.
                if view == 'Sagittaal': slice_idx = idx_x[col]
                elif view == 'Coronaal': slice_idx = idx_y[col]
                else: slice_idx = idx_z[col]
                
                # 3. Use the new 'slice machine' (ignore the last two values with '_').
                img, disp_idx, labels, vlak_naam, bollen, as_letter, _, _ = self._haal_beeld_info(
                    view, slice_idx, col, dir_x, dir_y, dir_z, bepaal_labels, sag_titles, cor_titles, ax_titles
                )
                
                assen = bepaal_assen_text(labels)
                kijkrichting = bepaal_kijkrichting(labels)
                title_text = f"{view} (slice: {disp_idx})\n[{vlak_naam} | Fiducials: {bollen}]"
                    
    
                ax.imshow(img, cmap='gray', vmin=v_min, vmax=v_max)
                ax.set_title(title_text, fontsize=11, fontweight='bold', pad=15, color='black')
                
                if bollen_overlay:
                    rad_x = 5.0 / spacing_x if spacing_x > 0 else 5
                    rad_y = 5.0 / spacing_y if spacing_y > 0 else 5
                    roi_px = int(round(12.0 / spacing_x))
                    fig.suptitle(f"Overview (with assigned fiducial numbers) {self.dataset_naam}", fontsize=20, fontweight='bold', y=0.98, color='black')
                    
                    for bol_nr, (p_idx, gx, gz) in BOL_DEF.items():
                        cx_col = (gx + 2) if dir_x == 1 else (2 - gx)
                        cy_col = p_idx if dir_y == 1 else (4 - p_idx)
                        cz_col = (gz + 2) if dir_z == 1 else (2 - gz)
                        
                        cx, cy, cz = idx_x[cx_col], idx_y[cy_col], idx_z[cz_col]
                        
                        if view == 'Sagittaal' and cx_col == col:
                            plot_x, plot_y, straal = cy, size_z - 1 - cz, rad_y
                        elif view == 'Coronaal' and cy_col == col:
                            plot_x, plot_y, straal = cx, size_z - 1 - cz, rad_x
                        elif view == 'Axiaal' and cz_col == col:
                            plot_x, plot_y, straal = cx, cy, rad_x
                        else:
                            continue 
                            
                        knip_rechthoek = patches.Rectangle(
                            (plot_x - roi_px, plot_y - roi_px), 
                            roi_px * 2, roi_px * 2, 
                            linewidth=1.5, edgecolor='magenta', facecolor='none', linestyle=':', alpha=0.9
                        )
                        ax.add_patch(knip_rechthoek)
    
                        ax.add_patch(patches.Circle((plot_x, plot_y), radius=straal, edgecolor='yellow', facecolor='none', lw=1.5, alpha=0.8))
                        ax.text(plot_x, plot_y, str(bol_nr), color='red', fontsize=12, ha='center', va='center', fontweight='bold')
    
                self._teken_kaders_en_tekst(ax, labels, kijkrichting, assen)
    
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        print(f"Saved: {save_path}")

# ==========================================
# 3c. Function for the 56 markers
# ==========================================
# The class already knows the volume and spacing internally.
    def genereer_bol_plaatjes(self, output_map, timestamp, maak_plaatjes=True, anatomie_override=None):
        # Main function for detecting the exact sub-pixel coordinates of all 56 markers.
        volume = self.volume
        spacings = self.spacings
        ankerpunt = self.ankerpunt
        dataset_naam = self.dataset_naam

        # --- NEW: START THE ANATOMY ENGINE HERE ---
        dir_z, dir_y, dir_x, _, _, _ = self._start_anatomie_motor(anatomie_override)

        spacing_z, spacing_y, spacing_x = spacings
        anker_z, anker_y, anker_x = ankerpunt
        size_z, size_y, size_x = volume.shape
        txt_lijnen = []
        bol_data_dict = {} # Collect the data here for direct transfer to boekhouding.py.

        # Determine the step size in pixels (40 mm physical distance between markers).
        step_x = int(round(40.0 / spacing_x)) if spacing_x > 0 else 40
        step_y = int(round(40.0 / spacing_y)) if spacing_y > 0 else 40
        step_z = int(round(40.0 / spacing_z)) if spacing_z > 0 else 40
    
        idx_x = [anker_x + i*step_x for i in range(-2, 3)]
        idx_y = [anker_y + i*step_y for i in range(-2, 3)] 
        idx_z = [anker_z + i*step_z for i in range(-2, 3)]
    
        roi = int(round(12.0 / spacing_x)) 
        
        search_roi_x = int(round(4.5 / spacing_x))
        search_roi_y = int(round(4.5 / spacing_y))
        search_roi_z = int(round(4.5 / spacing_z))
    
        clamp = lambda val, max_val: max(0, min(max_val - 1, val))
    
        def vind_piek_in_roi(heatmap_disp, center_x, center_y, rx, ry):
            # Find the exact peak in the correlation heatmap with sub-pixel precision.
            xmin = max(0, int(center_x - rx))
            xmax = min(heatmap_disp.shape[1], int(center_x + rx))
            ymin = max(0, int(center_y - ry))
            ymax = min(heatmap_disp.shape[0], int(center_y + ry))
            
            sub_map = heatmap_disp[ymin:ymax, xmin:xmax]
            
            if sub_map.size == 0: return center_x, center_y, center_x, center_y
                
            sub_map_pos = sub_map - np.min(sub_map)
            
            # Set the edges to 0 to prevent the peak from falling on the ROI boundary.
            if sub_map_pos.shape[0] > 2 and sub_map_pos.shape[1] > 2:
                sub_map_pos[0, :] = 0     
                sub_map_pos[-1, :] = 0    
                sub_map_pos[:, 0] = 0     
                sub_map_pos[:, -1] = 0    
            
            sub_map_weighted = sub_map_pos
            
            # 1. Find the raw pixel peak (integer coordinates).
            py_int, px_int = np.unravel_index(np.argmax(sub_map_weighted), sub_map_weighted.shape)
            px_int_abs = px_int + xmin
            py_int_abs = py_int + ymin
            px_2d, py_2d = px_int_abs, py_int_abs
            
            # 2. Sub-pixel calculation using 2D Taylor expansion (Hessian matrix).
            if 0 < py_int < sub_map_weighted.shape[0]-1 and 0 < px_int < sub_map_weighted.shape[1]-1:
                # === STRICT 2D CALCULATION ===
                w_links = sub_map_weighted[py_int, px_int - 1]
                w_mid_x = sub_map_weighted[py_int, px_int]
                w_rechts = sub_map_weighted[py_int, px_int + 1]
                w_boven = sub_map_weighted[py_int - 1, px_int]
                w_mid_y = sub_map_weighted[py_int, px_int]
                w_onder = sub_map_weighted[py_int + 1, px_int]
                
                # First derivative (gradients)
                g_x = (w_rechts - w_links) / 2.0
                g_y = (w_onder - w_boven) / 2.0
                
                # Second derivative (Hessian matrix elements -> curvature/convexity)
                h_xx = w_rechts - 2.0 * w_mid_x + w_links
                h_yy = w_onder - 2.0 * w_mid_y + w_boven
                h_xy = (sub_map_weighted[py_int + 1, px_int + 1] -
                        sub_map_weighted[py_int + 1, px_int - 1] -
                        sub_map_weighted[py_int - 1, px_int + 1] +
                        sub_map_weighted[py_int - 1, px_int - 1]) / 4.0
    
                det = h_xx * h_yy - h_xy**2
                
                # Solve the system for the sub-pixel shift (dx, dy).
                if abs(det) > 1e-8: # Avoid division by extremely small values.
                    dx_2d = (-g_x * h_yy + g_y * h_xy) / det
                    dy_2d = (-g_y * h_xx + g_x * h_xy) / det
                    
                    # Cap the shift at 1.5 pixels to prevent instability.
                    dx_2d = max(-1.5, min(1.5, dx_2d))
                    dy_2d = max(-1.5, min(1.5, dy_2d))
                    
                    px_2d = px_int_abs + dx_2d
                    py_2d = py_int_abs + dy_2d
    
            return px_2d, py_2d, px_int_abs, py_int_abs
    
        # --- HELPER FUNCTION FOR EXACTLY SCALING THE RAW TEMPLATE ---
        def plot_standalone_template(ax, template_norm, center_x, center_y, roi_val, flip=False):
            t_max = np.max(np.abs(template_norm))
            if t_max == 0: t_max = 1e-5
            
            disp_temp = np.flipud(template_norm) if flip else template_norm
            rows, cols = template_norm.shape
            
            extent = [center_x - cols/2.0, center_x + cols/2.0, center_y + rows/2.0, center_y - rows/2.0]
            
            ax.set_facecolor('#808080')
            ax.imshow(disp_temp, cmap='gray', vmin=-t_max, vmax=t_max, extent=extent, interpolation='none')
            ax.set_xlim(center_x - roi_val, center_x + roi_val)
            ax.set_ylim(center_y + roi_val, center_y - roi_val)
            ax.axis('off')
            
        # --- NEW 2D CONTOUR PLOT ---
        def teken_2d_contour(ax, hm_disp, px_sub, py_sub, int_x, int_y):
            # Create a tight 5x5 square around the raw pixel peak.
            rad = 2
            ymin, ymax = max(0, int_y-rad), min(hm_disp.shape[0], int_y+rad+1)
            xmin, xmax = max(0, int_x-rad), min(hm_disp.shape[1], int_x+rad+1)
            
            sub_map = hm_disp[ymin:ymax, xmin:xmax]
            if sub_map.size == 0: return # Failsafe
            
            # Create X and Y grids based on the actual absolute pixel coordinates.
            X, Y = np.meshgrid(np.arange(xmin, xmax), np.arange(ymin, ymax))
            
            # 1. Plot the filled 3D height map.
            ax.contourf(X, Y, sub_map, levels=15, cmap='plasma', alpha=0.8)
            
            # 2. Plot the tight black contour lines.
            ax.contour(X, Y, sub_map, levels=15, colors='black', linewidths=0.5, alpha=0.5)
            
            # 3. Draw the 25 raw 'digital' pixels as white dots.
            ax.plot(X.flatten(), Y.flatten(), '.', color='white', markersize=4, alpha=0.6)
            
            # 4. Overlay the calculated mathematical sub-pixel cross.
            ax.plot(px_sub, py_sub, 'x', color='lime', markersize=10, markeredgewidth=2)
            
            # UI settings
            ax.set_xlim(int_x - 1.5, int_x + 1.5)
            ax.set_ylim(int_y + 1.5, int_y - 1.5) # y-axis is inverted for the correct matrix orientation
            ax.set_xticks([int_x-1, int_x, int_x+1])
            ax.set_yticks([int_y-1, int_y, int_y+1])
            ax.grid(True, alpha=0.3, linestyle='--')
    
        for bol_nr, (p_idx, gx, gz) in BOL_DEF.items():
            bol_data_dict[bol_nr] = {'Coronaal': [], 'Sagittaal': [], 'Axiaal': []}
            start_tijd = time.time()
            plaat_nr = 5 - p_idx 
            
            # --- NEW: LET CX, CY, AND CZ FOLLOW THE ENGINE DIRECTIONS ---
            cx_col = (gx + 2) if dir_x == 1 else (2 - gx)
            cy_col = p_idx if dir_y == 1 else (4 - p_idx)
            cz_col = (gz + 2) if dir_z == 1 else (2 - gz)
            
            cx = idx_x[cx_col]
            cy = idx_y[cy_col]
            cz = idx_z[cz_col]
            
            z_start, z_eind = max(0, cz - roi), min(size_z, cz + roi)
            y_start, y_eind = max(0, cy - roi), min(size_y, cy + roi)
            x_start, x_eind = max(0, cx - roi), min(size_x, cx + roi)
                    
            bol_gebied = volume[z_start:z_eind, y_start:y_eind, x_start:x_eind]
                    
            v_min = np.percentile(bol_gebied, 1)
            v_max = np.percentile(bol_gebied, 90)
    
            if maak_plaatjes:
                # New layout: 15 rows in one image
                fig, axes = plt.subplots(15, 5, figsize=(15, 50))
                formule_tekst = r"Sub-Pixel 2D Formule: $\Delta x = \frac{-g_x h_{yy} + g_y h_{xy}}{h_{xx} h_{yy} - h_{xy}^2}$, $\Delta y = \frac{-g_y h_{xx} + g_x h_{xy}}{h_{xx} h_{yy} - h_{xy}^2}$"
                fig.suptitle(f"Bol {bol_nr} | Plaat {plaat_nr} | {dataset_naam}\nMethoden Vergelijking: Streng + 2D Sub-Pixel\n{formule_tekst}", fontsize=20, fontweight='bold', y=0.98, color='black')
                
                # Adjust Hspace to provide space between the text.
                plt.subplots_adjust(left=0.15, hspace=0.6, wspace=0.1, top=0.94)
    
                rij_labels = [
                    "Coronaal\n[Origineel]", "Coronaal\n[Template]", "Coronaal\n[Heatmap]", "Coronaal\n[Contour 2D]", "Coronaal\n[Matrix Overlap]",
                    "Sagittaal\n[Origineel]", "Sagittaal\n[Template]", "Sagittaal\n[Heatmap]", "Sagittaal\n[Contour 2D]", "Sagittaal\n[Matrix Overlap]",
                    "Axiaal\n[Origineel]", "Axiaal\n[Template]", "Axiaal\n[Heatmap]", "Axiaal\n[Contour 2D]", "Axiaal\n[Matrix Overlap]"
                ]
                c_m2 = [1.0, 0.0, 1.0] # Magenta for the overlap
    
            txt_cor = ["Coronaal"]
            txt_sag = ["Sagittaal"]
            txt_ax = ["Axiaal"]
    
            for i, dy in enumerate([-2, -1, 0, 1, 2]):
                
                # ==========================================
                # --- 1. CORONAAL ---
                # ==========================================
                s_idx = clamp(cy + dy, size_y)
                full_slice_cor = volume[:, s_idx, :]
                
                # Create the template and find the sub-pixel peak.
                tn_m2_cor, mask_m2_cor = maak_fourier_stempel_methode2_center_sample(spacing_z, spacing_x, full_slice_cor, cx, cz)
                heatmap_m2_cor = bereken_fourier_match_m2_streng(full_slice_cor, tn_m2_cor)
                px_m2_cor_2d, py_m2_cor_2d, pxi_cor, pyi_cor = vind_piek_in_roi(heatmap_m2_cor, cx, cz, search_roi_x, search_roi_z)
                
                # --- VISUAL TRANSFORMATION (DISPLAY ONLY) ---
                hm_m2_cor_disp = np.flipud(heatmap_m2_cor)
                c_y_disp_cor = size_z - 1 - cz
                py_m2_cor_2d_disp = size_z - 1 - py_m2_cor_2d
                pyi_cor_disp = size_z - 1 - pyi_cor
                
                if maak_plaatjes:
                    img_cor_disp = np.flipud(full_slice_cor)
                    
                    # Row 0: Original
                    ax_cor = axes[0, i]
                    ax_cor.imshow(img_cor_disp, cmap='gray', vmin=v_min, vmax=v_max)
                    ax_cor.set_xlim(cx - roi, cx + roi); ax_cor.set_ylim(c_y_disp_cor + roi, c_y_disp_cor - roi)
                    ax_cor.set_title(f"y={s_idx}", fontsize=12)
                    ax_cor.axis('off')
                    if i == 2: 
                        ax_cor.text(0.5, 1.25, "━━━━━━━━━━━━━━━━━━━━━━━━ [ Coronaal Aanzicht ] ━━━━━━━━━━━━━━━━━━━━━━━━", transform=ax_cor.transAxes, fontsize=18, fontweight='bold', ha='center', color='black')
                    if i == 0: ax_cor.text(-0.35, 0.5, rij_labels[0], transform=ax_cor.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='black')
    
                    # Row 1: Template
                    plot_standalone_template(axes[1, i], tn_m2_cor, cx, c_y_disp_cor, roi, flip=True)
                    if i == 0: axes[1, i].text(-0.35, 0.5, rij_labels[1], transform=axes[1, i].transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='#929591')
    
                    # Row 2: Heatmap
                    ax_hm_c = axes[2, i]
                    ax_hm_c.imshow(hm_m2_cor_disp, cmap='plasma') 
                    ax_hm_c.plot(px_m2_cor_2d, py_m2_cor_2d_disp, 'x', color='lime', markersize=12, markeredgewidth=2) 
                    rect_m2_cor = patches.Rectangle((cx - search_roi_x, c_y_disp_cor - search_roi_z), search_roi_x*2, search_roi_z*2, linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
                    ax_hm_c.add_patch(rect_m2_cor)
                    ax_hm_c.set_xlim(cx - roi, cx + roi); ax_hm_c.set_ylim(c_y_disp_cor + roi, c_y_disp_cor - roi)
                    ax_hm_c.axis('off')
                    if i == 0: ax_hm_c.text(-0.35, 0.5, rij_labels[2], transform=ax_hm_c.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='m')
    
                    # Row 3: 2D contour
                    ax_cont_c = axes[3, i]
                    teken_2d_contour(ax_cont_c, hm_m2_cor_disp, px_m2_cor_2d, py_m2_cor_2d_disp, pxi_cor, pyi_cor_disp) 
                    if i == 0: ax_cont_c.text(-0.4, 0.5, rij_labels[3], transform=ax_cont_c.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='#DAA520')
    
                    # Row 4: Overlap
                    ax_over = axes[4, i]
                    ax_over.imshow(img_cor_disp, cmap='gray', vmin=v_min, vmax=v_max)
                    plot_exact_template_overlay(ax_over, np.flipud(mask_m2_cor), px_m2_cor_2d, py_m2_cor_2d_disp, c_m2) 
                    ax_over.plot(px_m2_cor_2d, py_m2_cor_2d_disp, 'x', color='lime', markersize=10, markeredgewidth=2) 
                    ax_over.set_xlim(cx - roi, cx + roi); ax_over.set_ylim(c_y_disp_cor + roi, c_y_disp_cor - roi)
                    ax_over.axis('off')
                    if i == 0: ax_over.text(-0.35, 0.5, rij_labels[4], transform=ax_over.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='#C875C4')
                    
                    # Coordinates on two lines with a dark background.
                    coord_tekst_cor = f"x={px_m2_cor_2d * spacing_x:.3f}, y={s_idx * spacing_y:.3f}\nz={py_m2_cor_2d * spacing_z:.3f}"
                    ax_over.text(0.5, -0.25, coord_tekst_cor, transform=ax_over.transAxes, color='cyan', ha='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
                
                # Store the detected physical coordinates (in mm) in the dictionary.
                bol_data_dict[bol_nr]['Coronaal'].append((px_m2_cor_2d * spacing_x, s_idx * spacing_y, py_m2_cor_2d * spacing_z))
                txt_cor.append(f"(x={px_m2_cor_2d * spacing_x:.3f}, y={s_idx * spacing_y:.3f}, z={py_m2_cor_2d * spacing_z:.3f})")
    
                # ==========================================
                # --- 2. SAGITTAAL ---
                # ==========================================
                s_idx = clamp(cx + dy, size_x)
                full_slice_sag = volume[:, :, s_idx]
                
                tn_m2_sag, mask_m2_sag = maak_fourier_stempel_methode2_center_sample(spacing_z, spacing_y, full_slice_sag, cy, cz)
                heatmap_m2_sag = bereken_fourier_match_m2_streng(full_slice_sag, tn_m2_sag)
                px_m2_sag_2d, py_m2_sag_2d, pxi_sag, pyi_sag = vind_piek_in_roi(heatmap_m2_sag, cy, cz, search_roi_y, search_roi_z)
                
                # --- VISUELE OMZETTING (DISPLAY ONLY) ---
                hm_m2_sag_disp = np.flipud(heatmap_m2_sag)
                c_x_disp_sag = cy
                c_y_disp_sag = size_z - 1 - cz
                py_m2_sag_2d_disp = size_z - 1 - py_m2_sag_2d
                pyi_sag_disp = size_z - 1 - pyi_sag
                
                if maak_plaatjes:
                    img_sag_disp = np.flipud(full_slice_sag)
                    
                    # Row 5: Original
                    ax_sag = axes[5, i]
                    ax_sag.imshow(img_sag_disp, cmap='gray', vmin=v_min, vmax=v_max)
                    ax_sag.set_xlim(c_x_disp_sag - roi, c_x_disp_sag + roi); ax_sag.set_ylim(c_y_disp_sag + roi, c_y_disp_sag - roi)
                    ax_sag.set_title(f"x={s_idx}", fontsize=12)
                    ax_sag.axis('off')
                    if i == 2:
                        ax_sag.text(0.5, 1.15, "━━━━━━━━━━━━━━━━━━━━━━━━ [ Sagittaal Aanzicht ] ━━━━━━━━━━━━━━━━━━━━━━━━", transform=ax_sag.transAxes, fontsize=18, fontweight='bold', ha='center', color='black')
                    if i == 0: ax_sag.text(-0.35, 0.5, rij_labels[5], transform=ax_sag.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='black')
    
                    # Row 6: Template
                    plot_standalone_template(axes[6, i], tn_m2_sag, c_x_disp_sag, c_y_disp_sag, roi, flip=True)
                    if i == 0: axes[6, i].text(-0.35, 0.5, rij_labels[6], transform=axes[6, i].transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='#929591')
    
                    # Row 7: Heatmap
                    ax_hm_s = axes[7, i]
                    ax_hm_s.imshow(hm_m2_sag_disp, cmap='plasma')
                    ax_hm_s.plot(px_m2_sag_2d, py_m2_sag_2d_disp, 'x', color='lime', markersize=12, markeredgewidth=2) 
                    rect_m2_sag = patches.Rectangle((c_x_disp_sag - search_roi_y, c_y_disp_sag - search_roi_z), search_roi_y*2, search_roi_z*2, linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
                    ax_hm_s.add_patch(rect_m2_sag)
                    ax_hm_s.set_xlim(c_x_disp_sag - roi, c_x_disp_sag + roi); ax_hm_s.set_ylim(c_y_disp_sag + roi, c_y_disp_sag - roi)
                    ax_hm_s.axis('off')
                    if i == 0: ax_hm_s.text(-0.35, 0.5, rij_labels[7], transform=ax_hm_s.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='m')
    
                    # Row 8: 2D contour
                    ax_cont_s = axes[8, i]
                    teken_2d_contour(ax_cont_s, hm_m2_sag_disp, px_m2_sag_2d, py_m2_sag_2d_disp, pxi_sag, pyi_sag_disp) 
                    if i == 0: ax_cont_s.text(-0.4, 0.5, rij_labels[8], transform=ax_cont_s.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='#DAA520')
    
                    # Row 9: Overlap
                    ax_over = axes[9, i]
                    ax_over.imshow(img_sag_disp, cmap='gray', vmin=v_min, vmax=v_max)
                    plot_exact_template_overlay(ax_over, np.flipud(mask_m2_sag), px_m2_sag_2d, py_m2_sag_2d_disp, c_m2) 
                    ax_over.plot(px_m2_sag_2d, py_m2_sag_2d_disp, 'x', color='lime', markersize=10, markeredgewidth=2) 
                    ax_over.set_xlim(c_x_disp_sag - roi, c_x_disp_sag + roi); ax_over.set_ylim(c_y_disp_sag + roi, c_y_disp_sag - roi)
                    ax_over.axis('off')
                    if i == 0: ax_over.text(-0.35, 0.5, rij_labels[9], transform=ax_over.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='#C875C4')
    
                    coord_tekst_sag = f"x={s_idx * spacing_x:.3f}, y={px_m2_sag_2d * spacing_y:.3f}\nz={py_m2_sag_2d * spacing_z:.3f}"
                    ax_over.text(0.5, -0.25, coord_tekst_sag, transform=ax_over.transAxes, color='cyan', ha='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
                    
                # Store the detected physical coordinates (in mm) in the dictionary.
                bol_data_dict[bol_nr]['Sagittaal'].append((s_idx * spacing_x, px_m2_sag_2d * spacing_y, py_m2_sag_2d * spacing_z))
                txt_sag.append(f"(x={s_idx * spacing_x:.3f}, y={px_m2_sag_2d * spacing_y:.3f}, z={py_m2_sag_2d * spacing_z:.3f})")
    
                # ==========================================
                # --- 3. AXIAAL ---
                # ==========================================
                s_idx = clamp(cz + dy, size_z)
                full_slice_ax = volume[s_idx, :, :]
                
                tn_m2_ax, mask_m2_ax = maak_fourier_stempel_methode2_center_sample(spacing_y, spacing_x, full_slice_ax, cx, cy)
                heatmap_m2_ax = bereken_fourier_match_m2_streng(full_slice_ax, tn_m2_ax)
                px_m2_ax_2d, py_m2_ax_2d, pxi_ax, pyi_ax = vind_piek_in_roi(heatmap_m2_ax, cx, cy, search_roi_x, search_roi_y)
                
                if maak_plaatjes:
                    # Row 10: Original
                    ax_ax = axes[10, i]
                    ax_ax.imshow(full_slice_ax, cmap='gray', vmin=v_min, vmax=v_max)
                    ax_ax.set_xlim(cx - roi, cx + roi); ax_ax.set_ylim(cy + roi, cy - roi)
                    ax_ax.set_title(f"z={s_idx}", fontsize=12)
                    ax_ax.axis('off')
                    if i == 2:
                        ax_ax.text(0.5, 1.15, "━━━━━━━━━━━━━━━━━━━━━━━━ [ Axiaal Aanzicht ] ━━━━━━━━━━━━━━━━━━━━━━━━", transform=ax_ax.transAxes, fontsize=18, fontweight='bold', ha='center', color='black')
                    if i == 0: ax_ax.text(-0.35, 0.5, rij_labels[10], transform=ax_ax.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='black')
    
                    # Row 11: Template M2
                    plot_standalone_template(axes[11, i], tn_m2_ax, cx, cy, roi, flip=False)
                    if i == 0: axes[11, i].text(-0.35, 0.5, rij_labels[11], transform=axes[11, i].transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='#929591')
    
                    # Row 12: Heatmap M2
                    ax_hm_a = axes[12, i]
                    ax_hm_a.imshow(heatmap_m2_ax, cmap='plasma')
                    ax_hm_a.plot(px_m2_ax_2d, py_m2_ax_2d, 'x', color='lime', markersize=12, markeredgewidth=2)
                    rect_m2_ax = patches.Rectangle((cx - search_roi_x, cy - search_roi_y), search_roi_x*2, search_roi_y*2, linewidth=2, edgecolor='lime', facecolor='none', linestyle='--')
                    ax_hm_a.add_patch(rect_m2_ax)
                    ax_hm_a.set_xlim(cx - roi, cx + roi); ax_hm_a.set_ylim(cy + roi, cy - roi)
                    ax_hm_a.axis('off')
                    if i == 0: ax_hm_a.text(-0.35, 0.5, rij_labels[12], transform=ax_hm_a.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='m')
    
                    # Row 13: 2D contour
                    ax_cont_a = axes[13, i]
                    teken_2d_contour(ax_cont_a, heatmap_m2_ax, px_m2_ax_2d, py_m2_ax_2d, pxi_ax, pyi_ax)
                    if i == 0: ax_cont_a.text(-0.4, 0.5, rij_labels[13], transform=ax_cont_a.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='#DAA520')
    
                    # Row 14: Overlap
                    ax_over = axes[14, i]
                    ax_over.imshow(full_slice_ax, cmap='gray', vmin=v_min, vmax=v_max)
                    plot_exact_template_overlay(ax_over, mask_m2_ax, px_m2_ax_2d, py_m2_ax_2d, c_m2)
                    ax_over.plot(px_m2_ax_2d, py_m2_ax_2d, 'x', color='lime', markersize=10, markeredgewidth=2)
                    ax_over.set_xlim(cx - roi, cx + roi); ax_over.set_ylim(cy + roi, cy - roi)
                    ax_over.axis('off')
                    if i == 0: ax_over.text(-0.35, 0.5, rij_labels[14], transform=ax_over.transAxes, fontsize=12, va='center', ha='center', fontweight='bold', color='#C875C4')
    
                    coord_tekst_ax = f"x={px_m2_ax_2d * spacing_x:.3f}, y={py_m2_ax_2d * spacing_y:.3f}\nz={s_idx * spacing_z:.3f}"
                    ax_over.text(0.5, -0.25, coord_tekst_ax, transform=ax_over.transAxes, color='cyan', ha='center', fontsize=8, fontweight='bold', bbox=dict(facecolor='black', alpha=0.6, edgecolor='none', pad=2))
                
                # Store the detected physical coordinates (in mm) in the dictionary.
                bol_data_dict[bol_nr]['Axiaal'].append((px_m2_ax_2d * spacing_x, py_m2_ax_2d * spacing_y, s_idx * spacing_z))
                txt_ax.append(f"(x={px_m2_ax_2d * spacing_x:.3f}, y={py_m2_ax_2d * spacing_y:.3f}, z={s_idx * spacing_z:.3f})")
    
    
            # Combine marker data in the main text-file list.
            txt_lijnen.append(f"Bol {bol_nr}:")
            txt_lijnen.extend(txt_cor)
            txt_lijnen.extend(txt_sag)
            txt_lijnen.extend(txt_ax)
            txt_lijnen.append("") # Empty line after each marker for readability.
    
            # Save as one image.
            if maak_plaatjes:
                save_path = os.path.join(output_map, f"Bol {bol_nr} plaat {plaat_nr} {timestamp}.png")
                fig.savefig(save_path, bbox_inches='tight', dpi=100)
                plt.close(fig)
    
            eind_tijd = time.time()  
            duur = eind_tijd - start_tijd  
            
            print(f"Calculated: marker {bol_nr}/56 (time: {duur:.2f}s)")
        
        # Return the dictionary with all detected coordinates. This is the input for boekhouding.py.
        return bol_data_dict

# ==========================================
# 3d. Function for the template overview (third overview)
# ==========================================
    def genereer_template_overzicht(self, bol_data_dict, save_path, anatomie_override=None, geflipt=None):
        volume = self.volume
        spacing_z, spacing_y, spacing_x = self.spacings
        size_z, size_y, size_x = volume.shape

        # --- 1. CALL THE DYNAMIC ANATOMY ENGINE ---
        dir_z, dir_y, dir_x, bepaal_labels, bepaal_assen_text, bepaal_kijkrichting = self._start_anatomie_motor(anatomie_override)

        # --- 2. CALCULATE AVERAGE COORDINATES PER MARKER ---
        avg_bol_mm = {}
        for bol in BOL_DEF.keys():
            if bol not in bol_data_dict: continue
            
            # Cross-validation
            x_coords = [t[0] for t in bol_data_dict[bol]['Coronaal']] + [t[0] for t in bol_data_dict[bol]['Axiaal']]
            y_coords = [t[1] for t in bol_data_dict[bol]['Sagittaal']] + [t[1] for t in bol_data_dict[bol]['Axiaal']]
            z_coords = [t[2] for t in bol_data_dict[bol]['Sagittaal']] + [t[2] for t in bol_data_dict[bol]['Coronaal']]
            
            avg_bol_mm[bol] = (
                np.mean(x_coords) if x_coords else 0,
                np.mean(y_coords) if y_coords else 0,
                np.mean(z_coords) if z_coords else 0
            )

        # --- 3. DRAW THE 15 PLATES ---
        views = ['Sagittaal', 'Coronaal', 'Axiaal']
        sag_titles, cor_titles, ax_titles = self._bepaal_titels(dir_x, dir_y, dir_z)

        # Use the contrast helpers.
        timestamp = datetime.now().strftime("%d/%m/%Y %H:%M")
        titel = f"Template matching resultaten | {self.dataset_naam} | {timestamp}"
        fig, axes = self._maak_basis_figuur(titel)
        
        subtitel = f'Input: {self.acquired_view} | pixelgrootte (x, y, z) = ({spacing_x:.2f}mm, {spacing_y:.2f}mm, {spacing_z:.2f}mm)'
        
        # If a list of flipped axes was provided, append it neatly.
        if geflipt and len(geflipt) > 0:
            geflipte_assen_tekst = ",".join(geflipt) # Convert ['X', 'Y'] into the text "X,Y".
            subtitel += f' | flipped {geflipte_assen_tekst}-axis'
            
        fig.text(0.5, 0.95, subtitel, fontsize=12, fontweight='normal', fontstyle='italic', fontfamily='sans-serif', color='gray', ha='center')
        
        v_min, v_max = self._bepaal_contrast()
        c_m2 = [1.0, 0.0, 1.0] # Magenta color for the mask

        for row, view in enumerate(views):
            for col in range(5):
                ax = axes[row, col]
                
                # Determine which markers belong in this specific image.
                bols_in_col = []
                for bol_nr, (p_idx, gx, gz) in BOL_DEF.items():
                    cx_col = (gx + 2) if dir_x == 1 else (2 - gx)
                    cy_col = p_idx if dir_y == 1 else (4 - p_idx)
                    cz_col = (gz + 2) if dir_z == 1 else (2 - gz)

                    if view == 'Sagittaal' and cx_col == col: bols_in_col.append(bol_nr)
                    elif view == 'Coronaal' and cy_col == col: bols_in_col.append(bol_nr)
                    elif view == 'Axiaal' and cz_col == col: bols_in_col.append(bol_nr)

                if not bols_in_col:
                    ax.axis('off')
                    continue

                # Determine the average slice plane (plane_mm) and calculate the index.
                if view == 'Sagittaal':
                    plane_mm = np.mean([avg_bol_mm[b][0] for b in bols_in_col])
                    slice_idx = max(0, min(size_x - 1, int(round(plane_mm / spacing_x))))
                elif view == 'Coronaal':
                    plane_mm = np.mean([avg_bol_mm[b][1] for b in bols_in_col])
                    slice_idx = max(0, min(size_y - 1, int(round(plane_mm / spacing_y))))
                elif view == 'Axiaal':
                    plane_mm = np.mean([avg_bol_mm[b][2] for b in bols_in_col])
                    slice_idx = max(0, min(size_z - 1, int(round(plane_mm / spacing_z))))

                # Use the slice machine.
                img, disp_idx, labels, vlak_naam, bollen, as_letter, spacing_row, spacing_col = self._haal_beeld_info(
                    view, slice_idx, col, dir_x, dir_y, dir_z, bepaal_labels, sag_titles, cor_titles, ax_titles
                )

                assen = bepaal_assen_text(labels)
                kijkrichting = bepaal_kijkrichting(labels)
                
                # FIX 2: The title layout now shows the physical mm coordinate as the matrix slice.
                title_text = f"{view} (slice: {disp_idx}) | \n{as_letter}-as: {plane_mm:.1f} mm\n[{vlak_naam} | Fiducials: {bollen}]"

                ax.imshow(img, cmap='gray', vmin=v_min, vmax=v_max)
                ax.set_title(title_text, fontsize=11, fontweight='bold', pad=15, color='black')

                # --- 4. OVERLAY TEMPLATE AND CROSS ---
                # FIX 1 (Accepted): Generate a purely shape-based mask (True/False) mathematically here.
                # Gray values do not matter for the magenta color; only the geometric shape is needed,
                # and it is identical everywhere because the physical marker size (14 mm) is fixed.
                _, active_mask = maak_anti_aliased_stempel(spacing_row, spacing_col, 1.0, 0.0)

                for bol_nr in bols_in_col:
                    # Find in the dictionary the image (-2, -1, 0, 1, or 2) closest to plane_mm.
                    if view == 'Sagittaal':
                        best_t = min(bol_data_dict[bol_nr]['Sagittaal'], key=lambda t: abs(t[0] - plane_mm))
                        plot_x = best_t[1] / spacing_y
                        plot_y = size_z - 1 - (best_t[2] / spacing_z)
                    elif view == 'Coronaal':
                        best_t = min(bol_data_dict[bol_nr]['Coronaal'], key=lambda t: abs(t[1] - plane_mm))
                        plot_x = best_t[0] / spacing_x
                        plot_y = size_z - 1 - (best_t[2] / spacing_z)
                    elif view == 'Axiaal':
                        best_t = min(bol_data_dict[bol_nr]['Axiaal'], key=lambda t: abs(t[2] - plane_mm))
                        plot_x = best_t[0] / spacing_x
                        plot_y = best_t[1] / spacing_y

                    # Add the template (magenta overlay).
                    plot_exact_template_overlay(ax, active_mask, plot_x, plot_y, c_m2)
                    
                    # FIX 4: Small, thin green cross (marker size 2).
                    ax.plot(plot_x, plot_y, 'x', color='lime', markersize=2, markeredgewidth=0.5)
                    
                    # FIX 3: Calculate 15 millimeters ABOVE the center for the label.
                    # Because Y is inverted on screen (0 is at the top), "up" means subtracting.
                    offset_y_pixels = 15.0 / spacing_row
                    ax.text(plot_x, plot_y - offset_y_pixels, str(bol_nr), color='white', fontsize=10, ha='center', va='center', fontweight='bold', bbox=dict(facecolor='black', edgecolor='none', alpha=0.5, pad=0.3))

                # Draw borders and axes.
                self._teken_kaders_en_tekst(ax, labels, kijkrichting, assen)

        # FIX 4: Increase DPI to 300 for zoom quality.
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)
        print(f"Saved: {save_path}")
        
# ==========================================
# 4. MAIN FUNCTION
# ==========================================
def run_template_matching(dicom_lijst, dataset_naam, output_basis, maak_plaatjes=True, maak_template_overzicht=True, maak_nummer_overzicht=True):
    # Entry point called by main.py.
    totale_start_tijd = time.time()
    scan = MriAnalyse(dicom_lijst, dataset_naam)     # This reads the files, calculates spacing, and finds the anchor point.
    
    if scan.volume is not None:
        timestamp = datetime.now().strftime("%d_%m_%H%M")
        
        # Set the default output directory to 'nothing'.
        output_map = None
        
        # ====================================================================
        # ====================================================================
        ''' 
        --- TEST ZONE (Flipping multiple axes is now possible!) ---
        Which axes do you want to flip? Enter the letters in the list, for example ['X', 'Y'] or ['Z'].
        Leave the list* empty [] if you do not want to test anything.'''
        test_assen = []  # <--- enter values here*
        
        # Start with the standard, normal anatomy.
        anatomie_test = {'Z': ('I', 'S'), 'Y': ('A', 'P'), 'X': ('R', 'L')}
        
        # Process the list step by step.
        if 'Z' in test_assen:
            scan.volume = np.flip(scan.volume, axis=0) # Flip the Z-axis.
            anatomie_test['Z'] = ('S', 'I')            # Reverse the Z labels.
            print("-> NOTE: TEST ZONE IS ON! Z-axis has been flipped.")
            
        if 'Y' in test_assen:
            scan.volume = np.flip(scan.volume, axis=1) # Flip the Y-axis.
            anatomie_test['Y'] = ('P', 'A')            # Reverse the Y labels.
            print("-> NOTE: TEST ZONE IS ON! Y-axis has been flipped.")
            
        if 'X' in test_assen:
            scan.volume = np.flip(scan.volume, axis=2) # Flip the X-axis.
            anatomie_test['X'] = ('L', 'R')            # Reverse the X labels.
            print("-> NOTE: TEST ZONE IS ON! X-axis has been flipped.")
            
        # If the list is empty, disable the test completely.
        if len(test_assen) == 0:
            anatomie_test = None # No test specified, so disable everything.
        # ====================================================================
        # ====================================================================
        
        if maak_plaatjes or maak_template_overzicht or maak_nummer_overzicht:
            map_naam = f"result {timestamp} {scan.dataset_naam}"
            output_map = os.path.join(output_basis, map_naam)
            os.makedirs(output_map, exist_ok=True)

        if maak_nummer_overzicht:
            pad_overzicht = os.path.join(output_map, f"result {timestamp}.png")
            scan.genereer_overzicht(pad_overzicht, bollen_overlay=False, anatomie_override=anatomie_test)
            pad_overzicht_genummerd = os.path.join(output_map, f"result numbered overview {timestamp}.png")
            scan.genereer_overzicht(pad_overzicht_genummerd, bollen_overlay=True, anatomie_override=anatomie_test)

        # Call the intensive marker-image generation through the class.
        bol_data_dict = scan.genereer_bol_plaatjes(output_map, timestamp, maak_plaatjes, anatomie_override=anatomie_test)
        
        # THE NEW THIRD OVERVIEW (called only after the dictionary has been populated)
        if maak_template_overzicht and bol_data_dict:
            pad_template_overzicht = os.path.join(output_map, f"result template overview {timestamp}.png")
            scan.genereer_template_overzicht(bol_data_dict, pad_template_overzicht, anatomie_override=anatomie_test, geflipt=test_assen)
            
        totale_eind_tijd = time.time()
        print(f"Finished {scan.dataset_naam}! Processing time: {int((totale_eind_tijd - totale_start_tijd) // 60)} min {(totale_eind_tijd - totale_start_tijd) % 60:.2f} sec.")
        # Return the output directory, name, orientation, and essential dictionary to main.py.
        return output_map, scan.dataset_naam, bol_data_dict, scan.acquired_view
    return None, None, None, None