import json, numpy as np, pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter

df = pd.read_csv('furnace_master_dataset.csv.gz'); info = json.load(open('build_info.json'))
use = df[~df.is_duplicate_panel]

T1 = pd.DataFrame({'Cell number': [1686502, 3183898, 5468416], 'Outlet temperature (K)': [764, 717, 713]})
T2 = pd.DataFrame([
 ['Moisture (received basis)', '%', 31.75], ['Ash (received basis)', '%', 9.84], ['Volatiles (received basis)', '%', 42.42],
 ['Fixed carbon (received basis)', '%', 33.63], ['Net calorific value (received basis)', 'kJ/kg', 16310], ['C (received basis)', '%', 56.12],
 ['H (received basis)', '%', 3.77], ['O (received basis)', '%', 9.5], ['N (received basis)', '%', 0.95], ['S (received basis)', '%', 0.92],
 ['Total air', 't/h', 1334.3], ['Pulverized coal', 't/h', 193.5], ['Primary air temperature', 'degC', 60.0], ['Secondary air temperature', 'degC', 366.0],
 ['Primary air', 'kg/s', 110.07], ['Secondary air', 'kg/s', 260.54], ['Over-fire air', 'kg/s', 111.182]], columns=['Variable', 'Unit', 'Value'])
T3rows = []
def add(model, kind, combos, size):
    for c in combos: T3rows.append([model, kind, c, 18, 1, size])
add('Model for 0 deg tilt', 'Training set', ['-20,-10,10', '-20,-10,20', '-20,10,20', '-10,10,20', '-30,-20,20', '-30,-20,10', '-20,20,30', '-10,20,30'], 159142)
add('Model for 0 deg tilt', 'Test set', ['-10,0,10'], 3183898)
add('Model for +10 deg tilt', 'Training set', ['-10,0,30', '-10,20,30', '-10,0,20', '0,20,30', '-20,-10,20', '-20,20,30', '-30,0,20', '-30,20,30'], 159142)
add('Model for +10 deg tilt', 'Test set', ['0,10,20'], 3183898)
add('Model for -10 deg tilt', 'Training set', ['-30,-20,10', '-30,-20,0', '-20,0,10', '-30,-20,0', '-20,10,20', '-30,-20,20', '-20,0,30', '-20,-30,30'], 159142)
add('Model for -10 deg tilt', 'Test set', ['-20,-10,0'], 3183898)
T3 = pd.DataFrame(T3rows, columns=['Model', 'Dataset type', 'Included burner tilts (middle one = output)', 'Inputs', 'Outputs', 'Size of dataset (rows)'])
T4 = pd.DataFrame([
 [0,'DNN',0.9794,35.79,3.61,64.43],[0,'ELM',0.9755,47.94,4.67,71.35],[0,'MLP',0.9558,62.32,6.26,91.03],[0,'ESN',0.9032,95.35,10.22,123.86],
 [10,'DNN',0.9769,41.85,4.21,64.01],[10,'ELM',0.9721,45.80,4.51,73.35],[10,'MLP',0.9579,63.94,6.14,86.78],[10,'ESN',0.9318,90.46,10.44,141.87],
 [-10,'DNN',0.9729,40.62,4.35,73.34],[-10,'ELM',0.9713,49.79,5.09,74.06],[-10,'MLP',0.9451,73.12,7.07,101.77],[-10,'ESN',0.9197,102.02,12.72,114.31]],
 columns=['Predicted tilt (deg)', 'Model', 'R2', 'MAE (K)', 'MAPE (%)', 'RMSE (K)'])
T5 = pd.DataFrame([
 ['Burnout damper opening (%)', '60/60/60/60', '60/60/60/60'], ['Opening of secondary air door (%)', '30/30/30/30/30/30/30/30', '50/50/20/20/20/20/10/10'],
 ['Operation mode of coal mill', 'CDEF', 'CDEF'], ['Load (MW)', 176.873, 174.535], ['Total air volume (t/h)', 733.368, 701.883],
 ['Total coal volume (t/h)', 112.256, 116.855], ['Total primary air volume (t/h)', 325.075, 293.351], ['Total secondary air volume (t/h)', 406.027, 429.752],
 ['Secondary air temperature (K)', 614.252, 612.2445], ['Burner tilt (deg)', 0, 0]], columns=['Operating parameter', 'T1', 'T2'])
T6 = pd.DataFrame([
 ['T1','P1',720.2058,693.7626,3.67,685.4186,4.83],['T1','P2',715.8059,686.6587,4.07,680.0178,5.00],['T1','P3',672.5259,690.2866,2.64,694.8109,3.31],
 ['T1','P4',683.6787,706.7418,3.37,735.9600,7.65],['T1','P5',664.4302,660.8345,0.54,655.1446,1.40],['T1','P6',655.7321,641.9377,2.10,630.2241,3.89],
 ['T2','P1',685.3627,692.4517,1.05,676.2749,1.34],['T2','P2',682.8714,686.2042,0.50,670.3121,1.87],['T2','P3',663.6784,691.8176,4.16,676.7589,1.93],
 ['T2','P4',663.6797,711.2262,6.79,699.7639,5.16],['T2','P5',616.3666,660.7424,7.01,632.9126,2.61],['T2','P6',622.5044,641.9421,3.20,608.0657,2.37]],
 columns=['Condition','Measuring point','Measured temperature (K)','CFD result (K)','CFD error (%) as printed','Proposed-method result (K)','Proposed-method error (%) as printed'])
T6['CFD error (%) recomputed'] = (abs(T6['CFD result (K)'] - T6['Measured temperature (K)']) / T6['Measured temperature (K)'] * 100).round(2)
T6['Proposed error (%) recomputed'] = (abs(T6['Proposed-method result (K)'] - T6['Measured temperature (K)']) / T6['Measured temperature (K)'] * 100).round(2)

dd = pd.DataFrame([
 ['panel', 'Panel number in Fig. 6 (1-4 top row, 5-8 bottom row). Panel 5 is a second drawing of the 0 deg case.'],
 ['tilt_deg', 'Burner tilt angle of the CFD case (-30 ... +30). Model input.'],
 ['is_duplicate_panel', 'True for panel 5 (repeat of 0 deg). Used only for the digitisation repeatability check; excluded from training and testing.'],
 ['px_row, px_col', 'Pixel position inside the panel crop (traceability back to the figure).'],
 ['x_m', 'Horizontal position on the central vertical plane, metres from the left wall. Model input. (14.63 m wide shaft.)'],
 ['y_m', 'Height above the bottom of the hopper, metres (58.3 m total). Model input.'],
 ['band', 'Colour-bar band index 0-19 (0 = hottest 1917-2000 K, 19 = coldest 330-414 K).'],
 ['T_lo_K, T_hi_K', 'Lower and upper edge of that band.'],
 ['T_K', 'TARGET. Band midpoint in K. Quantised: the true CFD value lies within +/-41.75 K of it.'],
 ['colour_dist', 'RGB distance between the pixel and its matched band colour (small = confident). Pixels above 40 were dropped.'],
 ['zone', 'Approximate region: hopper, burner_zone, shaft_upper (y 32-43 m), nose, back_pass, horizontal_pass.'],
 ['split_random', 'train / test. Random 90 / 10 split of points within every tilt. Neighbouring pixels are near-identical, so this test is optimistic.'],
 ['split_block', 'train / test. 90 / 10 split by 4 m x 4 m spatial blocks, same blocks for every tilt, at least one block in every zone. Harder and more honest.']],
 columns=['Column', 'Meaning'])

qa = pd.DataFrame([
 ['Source figure', 'Fig. 6, Xue et al. (2024), Energy 286, 129568: CFD temperature, furnace central vertical plane, tilts -30 to +30 deg'],
 ['Scale', f"{info['scale_m_per_px']:.4f} m per pixel (58.3 m boiler height spans 481 px; shaft width then reads 14.5 m vs 14.63 m stated)"],
 ['Temperature resolution', '20 colour bands, 83.5 K wide. Labels are band midpoints, so a perfect continuous model has MAE about 21 K and RMSE about 24 K against them.'],
 ['Rows (all panels)', info['n_rows']], ['Rows used (panel 5 excluded)', info['n_usable']],
 ['Repeatability: pixels compared (panel 1 vs 5, both 0 deg)', info['qa']['n_pixels_compared']],
 ['Repeatability: exact same band (%)', round(info['qa']['exact_band_agreement_pct'], 2)],
 ['Repeatability: within 1 band (%)', round(info['qa']['within_1_band_pct'], 2)],
 ['Repeatability: mean absolute difference (K)', round(info['qa']['mean_abs_diff_K'], 2)],
 ['Dropped pixels', 'About 5% per panel: anti-aliased band edges, wall outlines and JPEG artefacts with colour distance above 40'],
 ['NOT in the paper', 'Raw 3-D CFD cells (3,183,898 per case), velocity fields, the 18-input training rows, the 162 plant conditions. "Data will be made available on request."']],
 columns=['Item', 'Value'])
split = pd.DataFrame([[c, s, int(((use[c] == s)).sum()), round(100 * (use[c] == s).mean(), 2)] for c in ['split_random', 'split_block'] for s in ['train', 'test']],
                     columns=['Split column', 'Set', 'Rows', 'Share (%)'])
zs = use.groupby('zone').agg(rows=('T_K', 'size'), test_random=('split_random', lambda s: int((s == 'test').sum())), test_block=('split_block', lambda s: int((s == 'test').sum()))).reset_index()
ts = use.groupby('tilt_deg').agg(rows=('T_K', 'size'), T_min=('T_K', 'min'), T_max=('T_K', 'max'), T_mean=('T_K', 'mean')).round(1).reset_index()

readme = pd.DataFrame([
 ['Master file for the FYP PIML work (KIG 4054)'],
 ['Built from: Xue W. et al. (2024) Energy 286, 129568. Everything numeric in the paper is here.'],
 [''],
 ['Sheets'],
 ['Fig6_sample: 3,000 rows of the digitised field (full file: furnace_master_dataset.csv.gz, 580,501 rows).'],
 ['Table1-Table6: the paper\'s tables, transcribed. Table 6 has the error % recomputed from the printed temperatures.'],
 ['DataDictionary, Digitisation_QA, Split_Summary, Zone_Summary, Tilt_Summary.'],
 [''],
 ['Read this first'],
 ['1. The paper does not publish its raw CFD data. The only spatial temperature fields in it are the contour plots, so the training data here is Fig. 6 digitised.'],
 ['2. Temperatures are quantised to 83.5 K colour bands. Metrics from this data are NOT directly comparable with the paper\'s Table 4 (3-D, unseen tilt, continuous CFD values).'],
 ['3. It is one vertical plane (2-D), not the 3-D furnace. Use it to build and debug the model; swap in real Fluent exports when the authors or your own CFD supply them.'],
 ['4. split_random is the 9/10 : 1/10 split asked for. Because neighbouring pixels are almost identical it is optimistic; report split_block alongside it.']]
, columns=['Master file'])

wb = Workbook(); wb.remove(wb.active)
head = PatternFill('solid', fgColor='1F3A5F'); hf = Font(bold=True, color='FFFFFF')
def sheet(name, d, widths=None, wrap=False):
    ws = wb.create_sheet(name); ws.append(list(d.columns))
    for r in d.itertuples(index=False): ws.append([None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, 'item') else v) for v in r])
    for c in ws[1]: c.fill = head; c.font = hf; c.alignment = Alignment(wrap_text=True, vertical='top')
    for i, col in enumerate(d.columns, 1):
        w = (widths or {}).get(col) or min(60, max(12, int(d[col].astype(str).str.len().max() if len(d) else 10) + 2, len(str(col)) + 2))
        ws.column_dimensions[get_column_letter(i)].width = w
    if wrap:
        for row in ws.iter_rows(min_row=2):
            for c in row: c.alignment = Alignment(wrap_text=True, vertical='top')
    ws.freeze_panes = 'A2'; return ws
ws = sheet('README', readme, {'Master file': 150}, wrap=True); ws['A1'].font = Font(bold=True, color='FFFFFF')
sheet('Fig6_sample', use.sample(3000, random_state=1).sort_values(['tilt_deg', 'y_m']).round(3))
sheet('Table1_Mesh', T1); sheet('Table2_BoundaryConds', T2, {'Variable': 40}); sheet('Table3_Datasets', T3, {'Included burner tilts (middle one = output)': 38})
sheet('Table4_Metrics', T4); sheet('Table5_OperatingConds', T5, {'Operating parameter': 38}); sheet('Table6_Sensors', T6)
sheet('DataDictionary', dd, {'Column': 22, 'Meaning': 130}, wrap=True); sheet('Digitisation_QA', qa, {'Item': 52, 'Value': 120}, wrap=True)
sheet('Split_Summary', split); sheet('Zone_Summary', zs); sheet('Tilt_Summary', ts)
wb.save('furnace_master.xlsx'); print('saved', [w.title for w in wb])
print(T6[['CFD error (%) as printed','CFD error (%) recomputed','Proposed-method error (%) as printed','Proposed error (%) recomputed']].round(2).T.to_string())
print('mean CFD err printed %.3f recomputed %.3f; proposed printed %.3f recomputed %.3f' % (T6.iloc[:,4].mean(), T6['CFD error (%) recomputed'].mean(), T6.iloc[:,6].mean(), T6['Proposed error (%) recomputed'].mean()))
