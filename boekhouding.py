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

# boekhouding.py
# version 20260907

import matplotlib.pyplot as plt
import re
import os
import numpy as np
import math
from datetime import datetime
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openpyxl import Workbook
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.formatting.rule import CellIsRule

# ==========================================
# 1. GLOBAL CONSTANTS & TOOLS (Unrelated to the specific data)
# ==========================================
# Same ground-truth definition as in template_matching.py
BOL_DEF_REF = {
    1: (0, 0, 1), 2: (0, -1, 0), 3: (0, 1, 0), 4: (0, 0, -1),
    5: (1, 0, 2), 6: (1, -1, 1), 7: (1, 0, 1), 8: (1, 1, 1), 9: (1, -2, 0), 10: (1, -1, 0), 11: (1, 0, 0), 12: (1, 1, 0), 13: (1, 2, 0), 14: (1, -1, -1), 15: (1, 0, -1), 16: (1, 1, -1), 17: (1, 0, -2),
    18: (2, -1, 2), 19: (2, 0, 2), 20: (2, 1, 2), 21: (2, -2, 1), 22: (2, -1, 1), 23: (2, 0, 1), 24: (2, 1, 1), 25: (2, 2, 1), 26: (2, -2, 0), 27: (2, -1, 0), 28: (2, 0, 0), 29: (2, 1, 0), 30: (2, 2, 0), 31: (2, -2, -1), 32: (2, -1, -1), 33: (2, 0, -1), 34: (2, 1, -1), 35: (2, 2, -1), 36: (2, -1, -2), 37: (2, 0, -2), 38: (2, 1, -2),
    39: (3, 0, 2), 40: (3, -1, 1), 41: (3, 0, 1), 42: (3, 1, 1), 43: (3, -2, 0), 44: (3, -1, 0), 45: (3, 0, 0), 46: (3, 1, 0), 47: (3, 2, 0), 48: (3, -1, -1), 49: (3, 0, -1), 50: (3, 1, -1), 51: (3, 0, -2),
    52: (4, 0, 1), 53: (4, -1, 0), 54: (4, 0, 0), 55: (4, 1, 0), 56: (4, 0, -1)
}

def bepaal_plaat(bol_nummer):
    # Helper for linking markers to a physical plate in the phantom
    if 1 <= bol_nummer <= 4: return 5
    elif 5 <= bol_nummer <= 17: return 4
    elif 18 <= bol_nummer <= 38: return 3
    elif 39 <= bol_nummer <= 51: return 2
    elif 52 <= bol_nummer <= 56: return 1
    return 0

def set_outer_border(ws, min_col, min_row, max_col, max_row):
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            cell = ws.cell(row=r, column=c)
            b_left = Side(style='medium') if c == min_col else cell.border.left
            b_right = Side(style='medium') if c == max_col else cell.border.right
            b_top = Side(style='medium') if r == min_row else cell.border.top
            b_bottom = Side(style='medium') if r == max_row else cell.border.bottom
            cell.border = Border(left=b_left, right=b_right, top=b_top, bottom=b_bottom)

# (Keep the mathematics outside the class)
def bereken_procrustes_rigid_body(a, b):
    # Ordinary Procrustes Analysis (Rotation and translation only, NO scaling)
    # Goal: align the measured point cloud (b) with the ideal point cloud (a) as closely as possible
    a = np.asarray(a)
    b = np.asarray(b)
    
    aT = a.mean(0)                                                  # 1. Calculate the centroids
    bT = b.mean(0)
    A = a - aT                                                      # 2. Center the point clouds around their own origin (0,0,0)
    B = b - bT
    C = np.dot(B.T, A)                                              # 3. Calculate the covariance matrix (NO normalization/scaling)
    U, _, V = np.linalg.svd(C)                                      # 4. Singular Value Decomposition (SVD) to find the optimal rotation
    aR = np.dot(U, V)
    #aS = 1.0                                                       # 5. Force the scale factor to exactly 1.0 (Rigid Body)
    aT_vec = aT - bT.dot(aR)                                        # 6. Calculate the translation (shift) WITHOUT a scale factor
    B_fitted = b.dot(aR) + aT_vec                                   # 7. Transform the measured points (rotate and shift only)
    aD = np.sqrt(np.mean(np.sum((a - B_fitted)**2, axis=1)))        # 8. Calculate the actual RMSD (mean Euclidean distance in mm)
    
    return aR, aT_vec, aD, B_fitted

# ==========================================
# 2. THE CLASS: THE BOOKKEEPER'S OFFICE
# ==========================================
class MriBoekhouder:
    def __init__(self, dataset1_naam, data1, dataset2_naam, data2, opslag_map, ref_x=190.0, ref_y=190.0, ref_z=147.5, fixed_axes=False):
        # Receives the two dictionaries (data1 and data2) from main.py
        self.dataset1_naam = dataset1_naam
        self.data1 = data1
        self.dataset2_naam = dataset2_naam
        self.data2 = data2
        # If opslag_map is empty (e.g. with only "results.json"), use the current directory (".")
        self.opslag_map = opslag_map if opslag_map else "."
        self.timestamp = datetime.now().strftime("%d_%m_%H%M")
        os.makedirs(self.opslag_map, exist_ok=True)
        
        # Store them internally so the rest of the code can use them
        self.ref_x = ref_x
        self.ref_y = ref_y
        self.ref_z = ref_z
        self.ref_z = ref_z
        self.fixed_axes = fixed_axes

        # Internal storage for the processed data
        self.alle_bollen = sorted(list(set(self.data1.keys()) | set(self.data2.keys())))
        self.rijen_voor_excel = []
        self.samenvatting_rijen = []
        self.bol_tracking = [] 
        
        # Arrays for the Procrustes analysis: build two corresponding arrays in Python (without '-')
        self.punten_gemeten = []
        self.punten_ideaal = []
        self.geldige_bollen = []
        
        self.arr_gemeten = None
        self.arr_ideaal = None
        self.fitted_punten = None
        self.dict_fitted = {}

    def verwerk_data(self):
        # TASK 1: Combine all lists and perform the mathematics
        huidige_rij_index = 3 
        
        for bol in self.alle_bollen:
            plaat = bepaal_plaat(bol)
            
            # Keep the cell addresses for the formulas here
            xt_refs, yt_refs, zt_refs = [], [], []                  
            # Internal lists for storing the pairs in Python
            py_xt_vals, py_yt_vals, py_zt_vals = [], [], []
            
            bol_start = huidige_rij_index
            kr_tracking = []
            
            # Loop over the 3 viewing directions and match coordinates from datasets 1 and 2
            for kijkrichting in ['Coronaal', 'Sagittaal', 'Axiaal']:
                kr_start = huidige_rij_index
                coords1 = self.data1.get(bol, {}).get(kijkrichting, [])
                coords2 = self.data2.get(bol, {}).get(kijkrichting, [])
                match_index = {'Coronaal': 1, 'Sagittaal': 0, 'Axiaal': 2}[kijkrichting]
                
                dict1 = {round(c[match_index], 3): c for c in coords1}
                dict2 = {round(c[match_index], 3): c for c in coords2}
                alle_match_sleutels = sorted(list(set(dict1.keys()) | set(dict2.keys())))
                
                for i, sleutel in enumerate(alle_match_sleutels):
                    c1 = dict1.get(sleutel)
                    c2 = dict2.get(sleutel)
                    xt_f, yt_f, zt_f = "-", "-", "-"
                    
                    if c1 and c2:
                        # Calculate the average of the pair internally
                        row_xt = (c1[0] + c2[0]) / 2.0
                        row_yt = (c1[1] + c2[1]) / 2.0
                        row_zt = (c1[2] + c2[2]) / 2.0
                        
                        xt_f, yt_f, zt_f = row_xt, row_yt, row_zt
                        
                        if kijkrichting == 'Coronaal':
                            xt_refs.append(f"J{huidige_rij_index}")
                            zt_refs.append(f"L{huidige_rij_index}")
                            py_xt_vals.append(row_xt)
                            py_zt_vals.append(row_zt)
                        elif kijkrichting == 'Sagittaal':
                            yt_refs.append(f"K{huidige_rij_index}")
                            zt_refs.append(f"L{huidige_rij_index}")
                            py_yt_vals.append(row_yt)
                            py_zt_vals.append(row_zt)
                        elif kijkrichting == 'Axiaal':
                            xt_refs.append(f"J{huidige_rij_index}")
                            yt_refs.append(f"K{huidige_rij_index}")
                            py_xt_vals.append(row_xt)
                            py_yt_vals.append(row_yt)
                    
                    rij = [
                        int(plaat), int(bol), kijkrichting,
                        c1[0] if c1 else "-", c1[1] if c1 else "-", c1[2] if c1 else "-",
                        c2[0] if c2 else "-", c2[1] if c2 else "-", c2[2] if c2 else "-",
                        xt_f, yt_f, zt_f
                    ]
                    self.rijen_voor_excel.append(rij)
                    huidige_rij_index += 1
                    
                kr_end = huidige_rij_index - 1
                if kr_start <= kr_end:
                    kr_tracking.append((kijkrichting, kr_start, kr_end))
                    
            bol_end = huidige_rij_index - 1
            r_black = huidige_rij_index
            
            # CALCULATE THE PROCRUSTES ARRAY
            gemeten_x, gemeten_y, gemeten_z = None, None, None
            # Calculate the average of the paired values in Python
            if py_xt_vals and py_yt_vals and py_zt_vals:
                gemeten_x = np.mean(py_xt_vals)
                gemeten_y = np.mean(py_yt_vals)
                gemeten_z = np.mean(py_zt_vals)

            f_xt = gemeten_x if gemeten_x is not None else "-"
            f_yt = gemeten_y if gemeten_y is not None else "-"
            f_zt = gemeten_z if gemeten_z is not None else "-"
        
            # Link the measured (average) position to the ideal (ground-truth) position
            if gemeten_x is not None and bol in BOL_DEF_REF:
                p_idx, gx, gz = BOL_DEF_REF[bol]
                std_x = gx * 40
                std_y = (p_idx - 2) * 40
                std_z = gz * 40
                self.punten_gemeten.append([gemeten_x, gemeten_y, gemeten_z])
                self.punten_ideaal.append([std_x, std_y, std_z])
                self.geldige_bollen.append(bol)
        
            self.samenvatting_rijen.append([
                int(plaat), int(bol), 
                f"=Results!J{r_black}", f"=Results!K{r_black}", f"=Results!L{r_black}"
            ])
            
            eind_rij = [None]*9 + [f_xt, f_yt, f_zt]
            self.rijen_voor_excel.append(eind_rij)
            huidige_rij_index += 1
            
            self.bol_tracking.append({
                'bol': bol, 'start': bol_start, 'end': bol_end, 'black': r_black, 'kijkrichting_ranges': kr_tracking
            })
        
        # CALL THE PROCRUSTES CALCULATION
        print("--- Starting Procrustes Analysis (Rigid Body) ---")
        self.arr_gemeten = np.array(self.punten_gemeten)
        self.arr_ideaal = np.array(self.punten_ideaal)
        
        rot_matrix, translatie, rmsd, fitted_punten = bereken_procrustes_rigid_body(self.arr_ideaal, self.arr_gemeten)
        self.fitted_punten = fitted_punten
        
        print(f"RMS fit error : {rmsd:.5f} mm")
        print(f"Translation:\n  X: {translatie[0]:.3f} mm\n  Y: {translatie[1]:.3f} mm\n  Z: {translatie[2]:.3f} mm")
        print(f"Rotation matrix:\n{rot_matrix}")
        
        sy = math.sqrt(rot_matrix[0,0] * rot_matrix[0,0] +  rot_matrix[1,0] * rot_matrix[1,0])
        singular = sy < 1e-6
        if not singular:
            x_angle = math.atan2(rot_matrix[2,1] , rot_matrix[2,2])
            y_angle = math.atan2(-rot_matrix[2,0], sy)
            z_angle = math.atan2(rot_matrix[1,0], rot_matrix[0,0])
        else:
            x_angle = math.atan2(-rot_matrix[1,2], rot_matrix[1,1])
            y_angle = math.atan2(-rot_matrix[2,0], sy)
            z_angle = 0
        print(f"\nEstimated rotation:\n  X-axis: {math.degrees(x_angle):.3f} deg\n  Y-axis: {math.degrees(y_angle):.3f} deg\n  Z-axis: {math.degrees(z_angle):.3f} deg\n")
        
        self.dict_fitted = {bol: fitted_punten[i] for i, bol in enumerate(self.geldige_bollen)}     
        
    def genereer_excel(self):
        # TASK 2: Create the Excel report with formatting and colors
        print("Building Excel file with formulas...")
        bestandsnaam_excel = f"markers {self.dataset1_naam}_-_{self.dataset2_naam} {self.timestamp}.xlsx"
        opslaan_als = os.path.join(self.opslag_map, bestandsnaam_excel)
        wb = Workbook()
        
        # Colors
        c_yellow = PatternFill(start_color="FFFF00", end_color="FFFF00", fill_type="solid")
        c_blue   = PatternFill(start_color="00B0F0", end_color="00B0F0", fill_type="solid")
        c_red    = PatternFill(start_color="FF0000", end_color="FF0000", fill_type="solid")
        c_orange = PatternFill(start_color="FFC000", end_color="FFC000", fill_type="solid")
        c_green  = PatternFill(start_color="92D050", end_color="92D050", fill_type="solid")
        c_gray   = PatternFill(start_color="D9D9D9", end_color="D9D9D9", fill_type="solid")
        c_white  = PatternFill(start_color="FFFFFF", end_color="FFFFFF", fill_type="solid")
        c_black  = PatternFill(start_color="000000", end_color="000000", fill_type="solid")
        c_purple = PatternFill(start_color="D2B4DE", end_color="D2B4DE", fill_type="solid") 
        c_pink   = PatternFill(start_color="FFCCCC", end_color="FFCCCC", fill_type="solid")
        
        vetgedrukt = Font(bold=True)
        font_rood = Font(color="FF0000")
        rule_roze = CellIsRule(operator='greaterThan', formula=['0.1'], fill=c_pink, font=font_rood)
        centreren  = Alignment(horizontal="center", vertical="center")
        thin_border = Border(left=Side(style='thin'), right=Side(style='thin'), top=Side(style='thin'), bottom=Side(style='thin'))
        
        ws = wb.active
        ws.title = "Results"
        
        # Headers for worksheet 1
        ws.merge_cells('A1:A2'); ws['A1'] = 'Plate'
        ws.merge_cells('B1:B2'); ws['B1'] = 'Marker'
        ws.merge_cells('C1:C2'); ws['C1'] = 'Viewing Direction'
        ws.merge_cells('D1:F1'); ws['D1'] = self.dataset1_naam
        ws.merge_cells('G1:I1'); ws['G1'] = self.dataset2_naam
        ws.merge_cells('J1:L1'); ws['J1'] = "Total"
        
        headers_rij2 = ['Plate', 'Marker', 'Viewing Direction', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'xt', 'yt', 'zt']
        ws.append(headers_rij2)
        
        for rij in self.rijen_voor_excel:
            ws.append(rij)
        
        # Formatting
        for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=12):
            for cell in row:
                cell.border = thin_border
                cell.alignment = centreren
                cell.fill = c_white
                if isinstance(cell.value, float) or (isinstance(cell.value, str) and cell.value.startswith('=')):
                    cell.number_format = '0.000'
                elif isinstance(cell.value, int):
                    cell.number_format = '0'
        
        for r in [1, 2]:
            for c in range(1, 13):
                cell = ws.cell(row=r, column=c)
                cell.fill = c_gray
                cell.font = vetgedrukt
        set_outer_border(ws, 1, 1, 12, 2)
        
        for b_data in self.bol_tracking:
            r_start, r_end, r_black = b_data['start'], b_data['end'], b_data['black']
            
            for r in range(r_start, r_end + 2):
                ws.cell(row=r, column=2).fill = c_blue 
                kr = None
                for kr_name, kr_s, kr_e in b_data['kijkrichting_ranges']:
                    if kr_s <= r <= kr_e: kr = kr_name; break
                for c in range(4, 13):
                    cell = ws.cell(row=r, column=c)
                    if cell.value == "-": cell.fill = c_red
                    else:
                        if kr == 'Coronaal': cell.fill = c_orange if c in [5, 8, 11] else c_green
                        elif kr == 'Sagittaal': cell.fill = c_orange if c in [4, 7, 10] else c_green
                        elif kr == 'Axiaal': cell.fill = c_orange if c in [6, 9, 12] else c_green
        
            for c in range(3, 10): ws.cell(row=r_black, column=c).fill = c_black
            for c in range(10, 13):
                cell = ws.cell(row=r_black, column=c)
                cell.fill = c_red if cell.value == "-" else c_yellow
                cell.font = vetgedrukt
            
            for r in range(r_start + 1, r_end + 1): ws.cell(row=r, column=2).value = None 
            for kr_name, kr_s, kr_e in b_data['kijkrichting_ranges']:
                for r in range(kr_s + 1, kr_e + 1): ws.cell(row=r, column=3).value = None
        
            if r_start < r_end: ws.merge_cells(start_row=r_start, start_column=2, end_row=r_end + 1, end_column=2)
            for kr_name, kr_s, kr_e in b_data['kijkrichting_ranges']:
                if kr_s < kr_e: ws.merge_cells(start_row=kr_s, start_column=3, end_row=kr_e, end_column=3)
            ws.merge_cells(start_row=r_black, start_column=3, end_row=r_black, end_column=9)
            set_outer_border(ws, 1, r_start, 12, r_black)
            
            # STDEV TABLE
            # 1. Top title (e.g. STDEV marker 1)
            bol_nr = b_data['bol']
            ws.merge_cells(start_row=r_start, start_column=14, end_row=r_start, end_column=19)
            cel_stdev_hdr = ws.cell(row=r_start, column=14)
            cel_stdev_hdr.value = f"STDEV marker {bol_nr}"
            cel_stdev_hdr.fill, cel_stdev_hdr.font, cel_stdev_hdr.alignment = c_gray, vetgedrukt, centreren
            set_outer_border(ws, 14, r_start, 19, r_start)
            
            # 2. The subheadings (x1, y1, z1, x2, y2, z2)
            for idx, sh in enumerate(['x1', 'y1', 'z1', 'x2', 'y2', 'z2']):
                c_idx = 14 + idx
                cel_sh = ws.cell(row=r_start + 1, column=c_idx)
                cel_sh.value, cel_sh.fill, cel_sh.font, cel_sh.alignment, cel_sh.border = sh, c_gray, vetgedrukt, centreren, thin_border
            
            # 3. Fill the calculation rows for the 3 viewing directions
            for idx, kr in enumerate(['Coronaal', 'Sagittaal', 'Axiaal']):
                r_stdev = r_start + 2 + idx
                kr_s, kr_e = None, None
                # Find the Excel rows containing the original data
                for kr_name, s, e in b_data['kijkrichting_ranges']:
                    if kr_name == kr: kr_s, kr_e = s, e; break
                    
                for col_offset, axis in enumerate(['x1', 'y1', 'z1', 'x2', 'y2', 'z2']):
                    c_idx = 14 + col_offset
                    data_c_letter = {'x1': 'D', 'y1': 'E', 'z1': 'F', 'x2': 'G', 'y2': 'H', 'z2': 'I'}[axis]
                    cel_data = ws.cell(row=r_stdev, column=c_idx)
                    cel_data.border, cel_data.alignment, cel_data.number_format = thin_border, centreren, '0.000000'
                    
                    # Determine which cells should be orange (and remain empty)
                    is_oranje = (kr == 'Coronaal' and axis in ['y1', 'y2']) or (kr == 'Sagittaal' and axis in ['x1', 'x2']) or (kr == 'Axiaal' and axis in ['z1', 'z2'])
                    if is_oranje:
                        cel_data.fill, cel_data.value = c_orange, None
                    else:
                        cel_data.fill = c_green
                        # Add the Excel formula
                        cel_data.value = f"=_xlfn.STDEV.P({data_c_letter}{kr_s}:{data_c_letter}{kr_e})" if kr_s and kr_e else "-"
            
            # Apply the pink highlight > 0.01 to the data just written
            ws.conditional_formatting.add(f"N{r_start + 2}:S{r_start + 4}", rule_roze)
            # Add a thick border around the data, as with the header
            set_outer_border(ws, 14, r_start + 1, 19, r_start + 4)
        
        plaat_ranges = {}
        for b_data in self.bol_tracking:
            r_start, r_black = b_data['start'], b_data['black']
            p_val = ws.cell(row=r_start, column=1).value
            if p_val is not None and p_val != "":
                if p_val not in plaat_ranges: plaat_ranges[p_val] = {'start': r_start, 'end': r_black}
                else: plaat_ranges[p_val]['end'] = r_black
        
        for p_val, p_data in plaat_ranges.items():
            s_row, e_row = p_data['start'], p_data['end']
            for r in range(s_row + 1, e_row + 1): ws.cell(row=r, column=1).value = None
            if s_row < e_row: ws.merge_cells(start_row=s_row, start_column=1, end_row=e_row, end_column=1)
            for r in range(s_row, e_row + 1):
                cell = ws.cell(row=r, column=1)
                b_top = Side(style='medium') if r == s_row else Side(style='thin')
                b_bot = Side(style='medium') if r == e_row else Side(style='thin')
                cell.border = Border(left=Side(style='medium'), right=Side(style='medium'), top=b_top, bottom=b_bot)
        
        for col in "ABCDEFGHIJKL": ws.column_dimensions[col].width = 12
        ws.column_dimensions['M'].width = 3 
        for col in "NOPQRS": ws.column_dimensions[col].width = 12
        
        # --- WORKSHEET 2: SUMMARY ---
        ws_sam = wb.create_sheet(title="Summary")
        ws_sam.append(['Plate', 'Marker', 'X (image mm)', 'Y (image mm)', 'Z (image mm)', 'templateX (mm)', 'templateY (mm)', 'templateZ (mm)', 'X (mm)', 'Y (mm)', 'Z (mm)', 'ErrorX (mm)', 'ErrorY (mm)', 'ErrorZ (mm)'])
        
        for s_rij in self.samenvatting_rijen:
            bol_nr = s_rij[1]
            if bol_nr in BOL_DEF_REF:
                p_idx, gx, gz = BOL_DEF_REF[bol_nr]
                std_x, std_y, std_z = gx * 40, (p_idx - 2) * 40, gz * 40
            else:
                std_x = std_y = std_z = "-"
                
            fit_x = fit_y = fit_z = afw_x = afw_y = afw_z = "-"
            if bol_nr in self.dict_fitted:
                fit_x, fit_y, fit_z = self.dict_fitted[bol_nr]
                if std_x != "-":
                    afw_x, afw_y, afw_z = fit_x - std_x, fit_y - std_y, fit_z - std_z
                    
            ws_sam.append(s_rij + [std_x, std_y, std_z, fit_x, fit_y, fit_z, afw_x, afw_y, afw_z])
            
        # Visual formatting of the Summary worksheet
        for row in ws_sam.iter_rows(min_row=1, max_row=ws_sam.max_row, min_col=1, max_col=14):
            for cell in row:
                cell.border, cell.alignment = thin_border, centreren
                if isinstance(cell.value, float) or (isinstance(cell.value, str) and cell.value.startswith('=')): cell.number_format = '0.000'
                elif isinstance(cell.value, int): cell.number_format = '0'
                
                if cell.row == 1: cell.fill, cell.font = c_gray, vetgedrukt
                else:
                    if cell.column == 2: cell.fill = c_blue 
                    elif cell.column in [3, 4, 5]: cell.fill = c_red if cell.value == "-" else c_yellow 
                    elif cell.column in [6, 7, 8]: cell.fill = c_orange
                    elif cell.column in [9, 10, 11]: cell.fill = c_red if cell.value == "-" else c_purple
                    elif cell.column in [12, 13, 14]: cell.fill = c_red if cell.value == "-" else c_pink
                    else: cell.fill = c_white
        
        plaat_sam_ranges = {}
        for r in range(2, ws_sam.max_row + 1):
            p_val = ws_sam.cell(row=r, column=1).value
            if p_val is not None and p_val != "":
                if p_val not in plaat_sam_ranges: plaat_sam_ranges[p_val] = {'start': r, 'end': r}
                else: plaat_sam_ranges[p_val]['end'] = r
        for p_val, p_data in plaat_sam_ranges.items():
            s_row, e_row = p_data['start'], p_data['end']
            for r in range(s_row + 1, e_row + 1): ws_sam.cell(row=r, column=1).value = None
            if s_row < e_row: ws_sam.merge_cells(start_row=s_row, start_column=1, end_row=e_row, end_column=1)
        
        set_outer_border(ws_sam, 1, 1, 14, ws_sam.max_row)
        for col_letter in "ABCDEFGHIJKLMN": ws_sam.column_dimensions[col_letter].width = 15
        
        wb.save(opslaan_als)
        return opslaan_als        
    
    def genereer_plots(self):
        # TASK 3: Build the Matplotlib and Plotly plots
        print("Generating 3x3 error plot and 3D model...")
        # 1. Separate the axes for convenience
        std_x, std_y, std_z = self.arr_ideaal[:, 0], self.arr_ideaal[:, 1], self.arr_ideaal[:, 2]
        
        # 2. Calculate the deviations (Fitted - Ideal)
        afw_x = self.fitted_punten[:, 0] - self.arr_ideaal[:, 0]
        afw_y = self.fitted_punten[:, 1] - self.arr_ideaal[:, 1]
        afw_z = self.fitted_punten[:, 2] - self.arr_ideaal[:, 2]
        
        # 3. Create the 3x3 figure (sharex and sharey keep the axes aligned)
        fig, axs = plt.subplots(3, 3, figsize=(12, 10), sharex='col', sharey='row')
        plt.subplots_adjust(wspace=0.1, hspace=0.15)
        fig.patch.set_facecolor('white')
        
        # 4. Data lists for easy iteration
        y_data, x_data = [std_x, std_y, std_z], [afw_x, afw_y, afw_z]
        x_labels = [u'ΔL-R [mm]', u'ΔP-A [mm]', u'ΔI-S [mm]']
        y_labels = [u'L-R [mm]', u'P-A [mm]', u'I-S [mm]']
        
        # Determine a symmetric X-axis limit so that 0 is always exactly in the middle
        max_afw = np.max(np.abs([afw_x, afw_y, afw_z]))
        # x_lim = max(max_afw * 1.2, 2.0) (minimum size is from -2 to 2)
        x_lim = max_afw * 1.2 if max_afw > 0 else 1.0
        print(f'The x-axis range is {-x_lim:.3f} to {x_lim:.3f}')
        
        # 5. Create the interactive figure (Plotly)
        fig_int = make_subplots(rows=3, cols=3, shared_xaxes='columns', shared_yaxes='rows', vertical_spacing=0.05, horizontal_spacing=0.02) 
        
        # 6. Plot the grid
        for row in range(3):
            for col in range(3):
                ax = axs[row, col]                                                          # --- Static plot (PNG) ---
                ax.scatter(x_data[col], y_data[row], marker='+', color='#377eb8', s=40)     # Plot with the specific blue plus signs
                ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)       # Light dotted lines at the origin (0,0) for reference
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                
                # --- Fixed or dynamic X-axis ---
                if self.fixed_axes:
                    ax.set_xlim(-2.0, 2.0)
                else:
                    ax.set_xlim(-x_lim, x_lim)                                              # Set the X-axis to our symmetric limit
                    
                if col == 0: ax.set_ylabel(y_labels[row], fontsize=12)                      # Set the outer labels
                if col == 0: ax.set_ylabel(y_labels[row], fontsize=12)                      # Set the outer labels
                if row == 2: ax.set_xlabel(x_labels[col], fontsize=12)
                
                # --- Interactive plot (HTML) ---
                fig_int.add_trace(
                    go.Scatter(x=x_data[col], y=y_data[row], 
                    mode='markers', marker=dict(symbol='cross', size=8, color='#377eb8'),
                    text=[f"Bol {b}" for b in self.geldige_bollen], 
                    hovertemplate="<b>%{text}</b><br>Afwijking: %{x:.3f} mm<br>", 
                    showlegend=False), 
                    row=row+1, col=col+1
                )
                
                fig_int.add_vline(x=0, line_width=0.5, line_dash="dash", line_color="gray", row=row+1, col=col+1)
                fig_int.add_hline(y=0, line_width=0.5, line_dash="dash", line_color="gray", row=row+1, col=col+1)
                
                # --- Fixed or dynamic X-axis for HTML ---
                if self.fixed_axes:
                    fig_int.update_xaxes(range=[-2.0, 2.0], row=row+1, col=col+1)
                else:
                    fig_int.update_xaxes(range=[-x_lim, x_lim], row=row+1, col=col+1)
                    
                fig_int.update_yaxes(range=[-100, 100], row=row+1, col=col+1)
                # Labels for the interactive axes
                if col == 0: fig_int.update_yaxes(title_text=y_labels[row], row=row+1, col=col+1)
                if row == 2: fig_int.update_xaxes(title_text=x_labels[col], row=row+1, col=col+1)
        
        # 7. Titles and saving
        # Save the static PNG
        fig.suptitle(f'Error plot\n{self.dataset1_naam} - {self.dataset2_naam} | {datetime.now().strftime("%d/%m/%Y %H:%M")}', fontsize=16, fontweight='bold', y=0.95, color='blue')
        plot_pad = os.path.join(self.opslag_map, f'ErrorPlot_{self.dataset1_naam}_-_{self.dataset2_naam}_{self.timestamp}.png')
        plt.savefig(plot_pad, bbox_inches='tight', dpi=150)
        plt.close(fig)
        
        # Save the interactive HTML
        fig_int.update_layout(height=900, width=1000, title_text=f"Interactive Error Plot - {self.dataset1_naam} - {self.dataset2_naam}", template="plotly_white")
        int_plot_pad = os.path.join(self.opslag_map, f'InteractiveErrorPlot_{self.dataset1_naam}_-_{self.dataset2_naam}_{self.timestamp}.html')
        fig_int.write_html(int_plot_pad)
        
        # =================================================================
        # TASK 3.5: THE LINEARIZED PLOT (Straightened + Original axes)
        # =================================================================
        # Create a new figure for the linearization
        fig_lin, axs_lin = plt.subplots(3, 3, figsize=(12, 10), sharex='col', sharey='row')
        plt.subplots_adjust(wspace=0.1, hspace=0.15)
        fig_lin.patch.set_facecolor('white')
        
        # 1. Calculate the linear trend (the sloped line) for each axis using least squares (y = mx + b)
        # np.polyfit(x, y, 1) returns [slope m, intercept b]
        m_x, b_x = np.polyfit(std_x, afw_x, 1)
        m_y, b_y = np.polyfit(std_y, afw_y, 1)
        m_z, b_z = np.polyfit(std_z, afw_z, 1)
        
        # --- NEW: Reference calculations for the console ---
        # Gradient scaling (1 + a_x), or (1 + slope)
        schaal_x = 1 + m_x
        schaal_y = 1 + m_y
        schaal_z = 1 + m_z
        
        # The calculation: reference_x * (1+a_x) - reference_x
        afw_ref_x = (self.ref_x * schaal_x) - self.ref_x
        afw_ref_y = (self.ref_y * schaal_y) - self.ref_y
        afw_ref_z = (self.ref_z * schaal_z) - self.ref_z
        
        # Print a clear summary table to the console
        print("\n" + "="*70)
        print("   CALCULATED DEVIATIONS BASED ON REFERENCE PHANTOM")
        print("="*70)
        print(f"X-axis (Reference: {self.ref_x:.1f} mm)  --> {self.ref_x * schaal_x:.5f} mm [{afw_ref_x:.3f} mm]")
        print(f"Y-axis (Reference: {self.ref_y:.1f} mm)  --> {self.ref_y * schaal_y:.5f} mm [{afw_ref_y:.3f} mm]")
        print(f"Z-axis (Reference: {self.ref_z:.1f} mm)  --> {self.ref_z * schaal_z:.5f} mm [{afw_ref_z:.3f} mm]")
        print("="*70 + "\n")
        # ---------------------------------------------------------
        
        # 2. Linearization: subtract the linear trend from the raw deviation
        # This "straightens the graph" and removes the systematic scaling error (e.g. due to incorrect gradient calibration)
        afw_x_lin = afw_x - (m_x * std_x + b_x)
        afw_y_lin = afw_y - (m_y * std_y + b_y)
        afw_z_lin = afw_z - (m_z * std_z + b_z)
        
        # 3. Calculate the (corrected) maximum deviations for the table
        max_dx_lin = np.max(np.abs(afw_x_lin))
        max_dy_lin = np.max(np.abs(afw_y_lin))
        max_dz_lin = np.max(np.abs(afw_z_lin))
        
        # 4. Set the axes as in the original plot
        # X-axis = The corrected deviation
        # Y-axis = The original position in the scanner
        lin_x_data = [afw_x_lin, afw_y_lin, afw_z_lin] 
        lin_y_data = [std_x, std_y, std_z]             
        
        lin_x_labels = [u'ΔL-R (corr) [mm]', u'ΔP-A (corr) [mm]', u'ΔI-S (corr) [mm]']
        lin_y_labels = [u'L-R [mm]', u'P-A [mm]', u'I-S [mm]']
        
        # Calculate a new symmetric x-limit for this plot,
        # so the remaining (actual) error can be viewed in detail.
        max_afw_lin = np.max([max_dx_lin, max_dy_lin, max_dz_lin])
        x_lim_lin = max_afw_lin * 1.2 if max_afw_lin > 0 else 1.0

        for row in range(3):
            for col in range(3):
                ax = axs_lin[row, col]
                
                # Plot the CORRECTED data (red plus signs)
                ax.scatter(lin_x_data[col], lin_y_data[row], marker='+', color='#e41a1c', s=40)
                
                # Draw guide lines
                ax.axvline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
                ax.axhline(0, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)

                # Finish the axes neatly
                if col == 0: ax.set_ylabel(lin_y_labels[row], fontsize=12)
                if row == 2: ax.set_xlabel(lin_x_labels[col], fontsize=12)
                # --- Fixed or dynamic X-axis ---
                if self.fixed_axes:
                    ax.set_xlim(-2.0, 2.0)
                else:
                    ax.set_xlim(-x_lim_lin, x_lim_lin)
                
        # Add the main title and subtitle with the corrected maximum deviations
        fig_lin.suptitle(f'Error plot after linear correction\n{self.dataset1_naam} - {self.dataset2_naam} | {datetime.now().strftime("%d/%m/%Y %H:%M")}', fontsize=16, fontweight='bold', y=0.98, color='blue')
        fig_lin.text(0.5, 0.91, f'Gradient scaling x: {schaal_x:.5f} | y: {schaal_y:.5f} | z: {schaal_z:.5f}', ha='center', fontsize=12, color='darkred')
        
        lin_plot_pad = os.path.join(self.opslag_map, f'ErrorPlotLinearCorrected{self.dataset1_naam}_-_{self.dataset2_naam}_{self.timestamp}.png')
        plt.savefig(lin_plot_pad, bbox_inches='tight', dpi=150)
        plt.close(fig_lin)
        # =================================================================
        
        # 3D MODEL
        fig_3d = go.Figure()
        # --- Helper function to create a large, transparent 3D sphere ---
        def voeg_grote_bol_toe(figuur, center_x, center_y, center_z, kleur, naam):
            # Calculate the mathematical shape of a sphere with a 100 mm radius (200 mm diameter)
            u, v = np.linspace(0, 2 * np.pi, 50), np.linspace(0, np.pi, 50)
            x = center_x + 100 * np.outer(np.cos(u), np.sin(v))
            y = center_y + 100 * np.outer(np.sin(u), np.sin(v))
            z = center_z + 100 * np.outer(np.ones(np.size(u)), np.cos(v))
            # Add the shell to the figure
            figuur.add_trace(go.Surface(
                x=x, y=y, z=z,
                name=naam,
                opacity=0.15, # Transparent enough to keep the 56 markers visible
                surfacecolor=np.zeros_like(x), # Ensures a uniform color
                colorscale=[[0, kleur], [1, kleur]], # The selected color
                showscale=False,
                hoverinfo='skip', # Prevents the mouse from constantly selecting the large outer surface
                showlegend=True
            ))
        
        # --- Determine the center for the large spheres (center = marker 28). If marker 28 is missing, use the average ---
        c_gemeten = self.arr_gemeten[self.geldige_bollen.index(28)] if 28 in self.geldige_bollen else np.mean(self.arr_gemeten, axis=0)
        c_ideaal = self.arr_ideaal[self.geldige_bollen.index(28)] if 28 in self.geldige_bollen else np.mean(self.arr_ideaal, axis=0)
        c_fitted = self.fitted_punten[self.geldige_bollen.index(28)] if 28 in self.geldige_bollen else np.mean(self.fitted_punten, axis=0)
        
        # 1. Final points (Measured) - Yellow
        # Size=10 creates a thicker display that visually approximates the 5 mm radius
        # Also draws a transparent 200 mm sphere around them
        fig_3d.add_trace(go.Scatter3d(x=self.arr_gemeten[:, 0], y=self.arr_gemeten[:, 1], z=self.arr_gemeten[:, 2], mode='markers', name='Eind (Gemeten)', marker=dict(size=10, color='yellow', line=dict(color='black', width=1))))
        voeg_grote_bol_toe(fig_3d, c_gemeten[0], c_gemeten[1], c_gemeten[2], 'yellow', 'Overkoepelende bol (Eind)')
        
        # 2. Standard points (Ideal) - Red
        fig_3d.add_trace(go.Scatter3d(x=self.arr_ideaal[:, 0], y=self.arr_ideaal[:, 1], z=self.arr_ideaal[:, 2], mode='markers', name='Standaard (Ideaal)', marker=dict(size=10, color='red', line=dict(color='black', width=1))))
        voeg_grote_bol_toe(fig_3d, c_ideaal[0], c_ideaal[1], c_ideaal[2], 'red', 'Overkoepelende bol (Standaard)')
        
        # 3. Fitted points (After Procrustes) - Purple
        fig_3d.add_trace(go.Scatter3d(x=self.fitted_punten[:, 0], y=self.fitted_punten[:, 1], z=self.fitted_punten[:, 2], mode='markers', name='Fitted (Procrustes)', marker=dict(size=10, color='purple', line=dict(color='black', width=1))))
        voeg_grote_bol_toe(fig_3d, c_fitted[0], c_fitted[1], c_fitted[2], 'purple', 'Overkoepelende bol (Fitted)')
        
        # Format the 3D space (axes, grid, and proportions)
        fig_3d.update_layout(
            title=f'Interactive 3D Model - {self.dataset1_naam}_-_{self.dataset2_naam} ',
            scene=dict(
                xaxis_title='X (mm)',
                yaxis_title='Y (mm)',
                zaxis_title='Z (mm)',
                aspectmode='data' # Important: ensures that one millimeter has the same length everywhere
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        pad_3d_html = os.path.join(self.opslag_map, f'Interactive3DModel_{self.dataset1_naam}_-_{self.dataset2_naam}_{self.timestamp}.html')
        fig_3d.write_html(pad_3d_html)
        
        # =================================================================
        # Extra statistics (SD, Delta R, histograms) for WAD-QC
        # =================================================================
        # Calculate the 3D Euclidean distance (Pythagoras in 3D) per marker
        delta_R = np.sqrt(afw_x**2 + afw_y**2 + afw_z**2)
        delta_R_lin = np.sqrt(afw_x_lin**2 + afw_y_lin**2 + afw_z_lin**2)
        
        # --- Histogram 1: Delta R (Uncorrected) ---
        fig_h1, ax_h1 = plt.subplots(figsize=(8, 6))
        fig_h1.patch.set_facecolor('white')
        ax_h1.hist(delta_R, bins=15, color='#377eb8', alpha=0.7, edgecolor='black')
        ax_h1.set_title(f'Histogram $\Delta$R\n{self.dataset1_naam} - {self.dataset2_naam}', fontsize=14, fontweight='bold', color='blue')
        ax_h1.set_xlabel(r'$\Delta$R (mm)', fontsize=12)
        ax_h1.set_ylabel('Number of markers', fontsize=12)
        ax_h1.set_xlim(0, 2.0)
        ax_h1.grid(axis='y', alpha=0.5, linestyle='--')
        hist_R_pad = os.path.join(self.opslag_map, f'HistogramDeltaR_{self.dataset1_naam}_-_{self.dataset2_naam}_{self.timestamp}.png')
        plt.savefig(hist_R_pad, bbox_inches='tight', dpi=150)
        plt.close(fig_h1)

        # --- Histogram 2: Delta R Linear (Corrected) ---
        fig_h2, ax_h2 = plt.subplots(figsize=(8, 6))
        fig_h2.patch.set_facecolor('white')
        ax_h2.hist(delta_R_lin, bins=15, color='#e41a1c', alpha=0.7, edgecolor='black')
        ax_h2.set_title(f'Histogram $\Delta$R (after linear correction)\n{self.dataset1_naam} - {self.dataset2_naam}', fontsize=14, fontweight='bold', color='darkred')
        ax_h2.set_xlabel(r'$\Delta$R after linear correction (mm)', fontsize=12)
        ax_h2.set_ylabel('Number of markers', fontsize=12)
        ax_h2.set_xlim(0, 2.0)
        ax_h2.grid(axis='y', alpha=0.5, linestyle='--')
        hist_R_lin_pad = os.path.join(self.opslag_map, f'HistogramDeltaR+Lin_{self.dataset1_naam}_-_{self.dataset2_naam}_{self.timestamp}.png')
        plt.savefig(hist_R_lin_pad, bbox_inches='tight', dpi=150)
        plt.close(fig_h2)

        # --- Build the complete WAD-QC final results list ---
        # ddof=1 is used for the standard 'Sample Standard Deviation'
        wad_cijfers = {
            "Scalefactor_X": schaal_x,
            "Scalefactor_Y": schaal_y,
            "Scalefactor_Z": schaal_z,
            "CalculatedDiameter_X_mm": self.ref_x * schaal_x,
            "CalculatedDiameter_Y_mm": self.ref_y * schaal_y,
            "CalculatedLength_Z_mm": self.ref_z * schaal_z,
            "Delta_R_Mean_mm": float(np.mean(delta_R)),
            "Delta_X_SD_mm": float(np.std(afw_x, ddof=1)),
            "Delta_Y_SD_mm": float(np.std(afw_y, ddof=1)),
            "Delta_Z_SD_mm": float(np.std(afw_z, ddof=1)),
            "Delta_R_Max_mm": float(np.max(delta_R)),
            "Delta_X_Max_mm": float(np.max(np.abs(afw_x))),
            "Delta_Y_Max_mm": float(np.max(np.abs(afw_y))),
            "Delta_Z_Max_mm": float(np.max(np.abs(afw_z))),
            "Delta_R_Lin_Max_mm": float(np.max(delta_R_lin)),
            "Delta_X_Lin_Max_mm": float(np.max(np.abs(afw_x_lin))),
            "Delta_Y_Lin_Max_mm": float(np.max(np.abs(afw_y_lin))),
            "Delta_Z_Lin_Max_mm": float(np.max(np.abs(afw_z_lin)))
        }
        
        # Also return the paths to the histograms
        return plot_pad, int_plot_pad, pad_3d_html, lin_plot_pad, hist_R_pad, hist_R_lin_pad, wad_cijfers
    
    def genereer_tekstrapport(self):
        # TASK 4: Write the .txt output
        pad_txt_uitvoer = os.path.join(self.opslag_map, f'MarkerCoordinates_{self.dataset1_naam}_-_{self.dataset2_naam}_{self.timestamp}.txt')
        with open(pad_txt_uitvoer, 'w') as f_out:
            f_out.write(f"Marker Coordinates - {self.dataset1_naam}_-_{self.dataset2_naam}\n")
            f_out.write("="*90 + "\n")
            f_out.write(f"{'Marker':<5} | {'X':<10} {'Y':<10} {'Z':<10} | {'ErrorX':<12} {'ErrorY':<12} {'ErrorZ':<12}\n")
            f_out.write("-" * 90 + "\n")
            
            for idx, bol_nr in enumerate(self.geldige_bollen):
                e_x, e_y, e_z = self.arr_gemeten[idx]                       # Retrieve the correct values from the Python lists
                f_x, f_y, f_z = self.fitted_punten[idx]                     
                p_idx, gx, gz = BOL_DEF_REF[bol_nr]                         # Retrieve the standard ideal values
                s_x, s_y, s_z = gx * 40, (p_idx - 2) * 40, gz * 40
                
                # Calculate the deviation (Fitted - Ideal)
                a_x = f_x - s_x
                a_y = f_y - s_y
                a_z = f_z - s_z
                
                f_out.write(f"{bol_nr:<5} | {e_x:<10.3f} {e_y:<10.3f} {e_z:<10.3f} | {a_x:<12.3f} {a_y:<12.3f} {a_z:<12.3f}\n")
                
# ==========================================
# 3. THE MAIN FUNCTION
# ==========================================
def run_boekhouding(dataset1_naam, data1, dataset2_naam, data2, opslag_map, ref_x=190.0, ref_y=190.0, ref_z=147.5, fixed_axes=False):
    # Entry point called by main.py
    # Step 1: Create the bookkeeping object and provide it with all data
    boekhouder = MriBoekhouder(dataset1_naam, data1, dataset2_naam, data2, opslag_map, ref_x, ref_y, ref_z, fixed_axes)
    
    # Step 2: Tell the bookkeeping object what to do, step by step.
    boekhouder.verwerk_data()
    excel_pad = boekhouder.genereer_excel()
    png_pad, int_plot_pad, html_pad, lin_plot_pad, hist_R_pad, hist_R_lin_pad, wad_cijfers = boekhouder.genereer_plots()
    boekhouder.genereer_tekstrapport()
    
    print(f"Done! The Excel file, including formulas and plots, is located in:\n{opslag_map}")
    return excel_pad, png_pad, html_pad, int_plot_pad, lin_plot_pad, hist_R_pad, hist_R_lin_pad, wad_cijfers, boekhouder.timestamp


# --- LEGACY HELPER FUNCTIONS FOR RUNNING THIS SCRIPT STANDALONE ---
def vind_juiste_bestand(map_pad):
    for bestandsnaam in os.listdir(map_pad):
        if bestandsnaam.startswith("resultaten_alle_coordinaten") and bestandsnaam.endswith(".txt"):
            return os.path.join(map_pad, bestandsnaam), bestandsnaam
    raise FileNotFoundError(f"Note: No results file found in the directory:\n{map_pad}")

def lees_bestand(pad):
    data = {}
    with open(pad, 'r') as f:
        huidige_bol, huidige_kijk = None, None
        for regel in f:
            regel = regel.strip()
            if regel.startswith("Bol"):
                huidige_bol = int(re.search(r'\d+', regel).group())
                data[huidige_bol] = {'Coronaal': [], 'Sagittaal': [], 'Axiaal': []}
            elif regel in ['Coronaal', 'Sagittaal', 'Axiaal']:
                huidige_kijk = regel
            elif regel.startswith("(x="):
                match = re.search(r'x=([-\d.]+),\s*y=([-\d.]+),\s*z=([-\d.]+)', regel)
                if match:
                    x, y, z = map(float, match.groups())
                    data[huidige_bol][huidige_kijk].append((x, y, z))
    return data

if __name__ == "__main__":
    m1 = r"Results\Resultaat 30_04_1248 MR QC_HPD_TRA_RL"
    m2 = r"Results\Resultaat 30_04_1244 MR QC_HPD_TRA_LR"
    opslag = r"Results Python Excel"
    
    pad1, bestand1 = vind_juiste_bestand(m1)
    pad2, bestand2 = vind_juiste_bestand(m2)
    naam1 = bestand1.replace("resultaten_alle_coordinaten ", "").replace(".txt", "")
    naam2 = bestand2.replace("resultaten_alle_coordinaten ", "").replace(".txt", "")

    losse_data1 = lees_bestand(pad1)
    losse_data2 = lees_bestand(pad2)

    run_boekhouding(naam1, losse_data1, naam2, losse_data2, opslag)            