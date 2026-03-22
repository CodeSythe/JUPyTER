# workers/call_exotedrf.py
import os
import sys
import glob
import yaml
import argparse
import subprocess
import shutil
import warnings
import numpy as np
import pandas as pd
import re
from astropy.io import fits
from scipy.optimize import curve_fit

# =====================================================================
# YAML CUSTOM DUMPER (Forces inline lists [1,2,3] to prevent ParserErrors)
# =====================================================================
class InlineListDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

def inline_list_rep(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

InlineListDumper.add_representer(list, inline_list_rep)

# =====================================================================
# 1. DETECT METADATA
# =====================================================================
def extract_metadata_from_uncal(input_dir):
    search = glob.glob(os.path.join(input_dir, '*uncal.fits'))
    if not search:
        raise FileNotFoundError(f"No uncal.fits files found in {input_dir}")
        
    hdr = fits.getheader(search[0], 0)
    instr = hdr.get('INSTRUME', 'UNKNOWN').upper()
    detector = hdr.get('DETECTOR', 'UNKNOWN').upper()
    filt = hdr.get('FILTER', 'UNKNOWN').upper()
    
    if instr == 'NIRISS':
        obs_mode = 'NIRISS/SOSS'
        filt_det = filt if filt in ['CLEAR', 'F277W'] else 'CLEAR'
    elif instr == 'NIRSPEC':
        obs_mode = 'NIRSpec/G395H'
        filt_det = detector
    elif instr == 'MIRI':
        obs_mode = 'MIRI/LRS'
        filt_det = 'CLEAR' 
    else:
        obs_mode = 'NIRSpec/G395H'
        filt_det = 'NRS2'
        
    return obs_mode, filt_det

# =====================================================================
# 2. CUSTOM BINNING (USER CODE + STAGE 2 WCS RESCUE)
# =====================================================================
def run_custom_binning(s3_root, s4_dir, wave_min, wave_max, num_bins=None, bin_width=None):
    print("    [1/3] Executing Custom Multi-Order Binning...", flush=True)
    
    s3_search = glob.glob(os.path.join(s3_root, "**", "*_spectra_fullres.fits"), recursive=True)
    if not s3_search:
        s3_search = glob.glob(os.path.join(s3_root, "**", "*_x1d.fits"), recursive=True)
    if not s3_search:
        raise FileNotFoundError(f"Could not find Stage 3 fullres spectra in {s3_root}")
        
    input_file = max(s3_search, key=os.path.getmtime)
    output_file = os.path.join(s4_dir, "custom_binned_spectra.fits")

    if bin_width is None and num_bins is not None:
        bin_width = (wave_max - wave_min) / num_bins
    elif bin_width is None:
        bin_width = 0.042

    # --- DEFINE CORRECTED BINNING FUNCTION ---
    def bin_at_bins_corrected(inwave_low, inwave_up, flux, err, outwave_low, outwave_up):
        nints, ncols = np.shape(flux)
        wave = np.nanmean([inwave_low, inwave_up], axis=0)

        binspec = np.zeros((nints, len(outwave_up)))
        binvar = np.zeros((nints, len(outwave_up))) 

        for j in range(len(outwave_up)):
            low = outwave_low[j]
            up = outwave_up[j]
            for i in range(ncols):
                w = wave[i]
                if low <= w < up:
                    binspec[:, j] += flux[:, i]
                    binvar[:, j] += err[:, i]**2 

        binerr = np.sqrt(binvar)
        return binspec, binerr
    # ----------------------------------------

    hdul = fits.open(input_file)
    new_hdul = fits.HDUList()
    new_hdul.append(hdul[0]) # Keep primary header

    # Preserve TIME extension if it exists
    time_ext = None
    for ext in hdul:
        if ext.name and ext.name.upper() == 'TIME':
            time_ext = ext
            break

    # Find all base FLUX extensions (ignore ERR extensions)
    flux_exts = [ext for ext in hdul if ext.name and 'FLUX' in ext.name.upper() and 'ERR' not in ext.name.upper()]
    flux_exts.sort(key=lambda x: x.name)

    # Define edges once
    new_edges = np.arange(wave_min, wave_max + bin_width, bin_width)
    if len(new_edges) - 1 != num_bins:
        new_edges = np.linspace(wave_min, wave_max, num_bins + 1)
    outwave_low = new_edges[:-1]
    outwave_up = new_edges[1:]

    # Loop through all found spectral orders
    for f_ext in flux_exts:
        fname = f_ext.name
        # For SOSS, this extracts ' O1' or ' O2'. For NIRSpec, it extracts ''
        suffix = fname.upper().replace('FLUX', '').strip()
        
        try:
            # Dynamically pair the matching extensions for this specific order
            w_ext = [e for e in hdul if e.name and 'WAVE' in e.name.upper() and 'ERR' not in e.name.upper() and suffix in e.name.upper()][0]
            we_ext = [e for e in hdul if e.name and 'WAVE' in e.name.upper() and 'ERR' in e.name.upper() and suffix in e.name.upper()][0]
            fe_ext = [e for e in hdul if e.name and 'FLUX' in e.name.upper() and 'ERR' in e.name.upper() and suffix in e.name.upper()][0]
        except IndexError:
            print(f"          [!] Warning: Missing matching extensions for {fname}. Skipping order.")
            continue

        wave_full = w_ext.data
        wave_err_full = we_ext.data
        flux_full = f_ext.data
        err_full = fe_ext.data

        # --- CROP ---
        if wave_full.ndim == 1:
            w1d = wave_full
            we1d = wave_err_full
        else:
            w1d = wave_full.flatten()
            we1d = wave_err_full.flatten()

        pixel_mask = (w1d >= (wave_min - 0.05)) & (w1d <= (wave_max + 0.05))

        wave_cut = w1d[pixel_mask]
        wave_err_cut = we1d[pixel_mask]

        # Handle 2D vs 3D flux arrays dynamically
        if flux_full.ndim == 3:
            f2d = flux_full[:, 0, :]
            e2d = err_full[:, 0, :]
        else:
            f2d = flux_full
            e2d = err_full

        flux_cut = f2d[:, pixel_mask]
        err_cut = e2d[:, pixel_mask]

        inwave_low = wave_cut - wave_err_cut
        inwave_up = wave_cut + wave_err_cut

        # --- BINNING ---
        binspec, binerr = bin_at_bins_corrected(
            inwave_low, inwave_up, flux_cut, err_cut, outwave_low, outwave_up
        )

        new_wave_centers = (outwave_low + outwave_up) / 2.0
        new_wave_widths = (outwave_up - outwave_low) / 2.0

        # --- APPEND TO NEW HDU ---
        h1, h2, h3, h4 = w_ext.header.copy(), we_ext.header.copy(), f_ext.header.copy(), fe_ext.header.copy()
        
        # Pop dimensional keys to prevent exoTEDRF shape mismatch crashes
        for h in [h1, h2, h3, h4]:
            for key in ['NAXIS1', 'NAXIS2', 'NAXIS']:
                h.pop(key, None)

        new_hdul.append(fits.ImageHDU(data=new_wave_centers, name=w_ext.name, header=h1)) 
        new_hdul.append(fits.ImageHDU(data=new_wave_widths, name=we_ext.name, header=h2))
        new_hdul.append(fits.ImageHDU(data=binspec, name=f_ext.name, header=h3))
        new_hdul.append(fits.ImageHDU(data=binerr, name=fe_ext.name, header=h4))

    # Append Time block safely
    if time_ext is not None:
        new_hdul.append(time_ext)
    elif len(hdul) > 5 and hdul[5].name.upper() == 'TIME':
        new_hdul.append(hdul[5])

    new_hdul.writeto(output_file, overwrite=True)
    hdul.close()
    
    return output_file

# =====================================================================
# 3. DETRENDING 
# =====================================================================
def gaussian(x, amp, mu, sig, bg):
    sig = np.maximum(sig, 1e-5)
    return amp * np.exp(-0.5 * ((x - mu) / sig)**2) + bg

def get_guided_spatial_profile(cube, trace_pos, disp_axis='x'):
    nint, ny, nx = cube.shape
    if disp_axis == 'x':
        collapse_axis = 2 
        trace_indices = np.arange(nx) 
    else:
        collapse_axis = 1 
        trace_indices = np.arange(ny) 

    if trace_pos is None:
        return np.nanmedian(cube, axis=collapse_axis)

    if len(trace_pos) != len(trace_indices):
        if len(trace_pos) > len(trace_indices):
             current_trace = trace_pos[:len(trace_indices)]
        else:
             return np.nanmedian(cube, axis=collapse_axis)
    else:
        current_trace = trace_pos

    valid_slices = np.where(np.isfinite(current_trace))[0]
    if len(valid_slices) == 0:
        return np.nanmedian(cube, axis=collapse_axis)

    if disp_axis == 'x':
        profiles = np.nanmedian(cube[:, :, valid_slices], axis=2)
    else:
        profiles = np.nanmedian(cube[:, valid_slices, :], axis=1)
    return profiles

def measure_spatial_drift(profiles, guess_center, window_width):
    nint, n_pixels = profiles.shape
    drift = []
    fwhm = []
    flux = []
    ax = np.arange(n_pixels)
    
    safe_width = max(window_width, 5)
    min_p = int(max(0, guess_center - safe_width // 2))
    max_p = int(min(n_pixels, guess_center + safe_width // 2))
    
    for i in range(nint):
        prof = profiles[i]
        sub_ax = ax[min_p:max_p]
        sub_prof = prof[min_p:max_p]
        
        if len(sub_prof) == 0 or np.all(np.isnan(sub_prof)):
            drift.append(np.nan)
            fwhm.append(np.nan)
            flux.append(np.nan)
            continue
            
        amp_g = np.nanmax(sub_prof)
        mu_g = sub_ax[np.nanargmax(sub_prof)] if np.any(np.isfinite(sub_prof)) else guess_center
        sig_g = 1.0
        bg_g = np.nanmedian(sub_prof)
        flux.append(np.nansum(sub_prof))
        
        try:
            bounds = ([0, min_p, 0.1, -np.inf], [np.inf, max_p, safe_width, np.inf])
            popt, _ = curve_fit(gaussian, sub_ax, sub_prof, p0=[amp_g, mu_g, sig_g, bg_g], bounds=bounds, maxfev=500)
            drift.append(popt[1])
            fwhm.append(np.abs(popt[2]) * 2.355) 
        except:
            if np.nansum(sub_prof) > 0:
                com = np.nansum(sub_ax * sub_prof) / np.nansum(sub_prof)
            else:
                com = np.nan
            drift.append(com)
            fwhm.append(np.nan)
            
    return np.array(drift), np.array(fwhm), np.array(flux)

def measure_spectral_drift(cube, disp_axis='x'):
    nint, ny, nx = cube.shape
    drift = []
    
    if disp_axis == 'x':
        spectra = np.nansum(cube, axis=1)
    else:
        spectra = np.nansum(cube, axis=2)
    
    ref_spec = np.nanmedian(spectra, axis=0)
    valid = np.isfinite(ref_spec)
    ref_spec = ref_spec[valid]
    
    if len(ref_spec) < 10:
        return np.zeros(nint)

    for i in range(nint):
        spec = spectra[i][valid]
        if len(spec) == 0:
            drift.append(np.nan)
            continue
            
        lags = np.arange(-2, 3) 
        cc = []
        for l in lags:
            rolled = np.roll(spec, l)
            val = np.corrcoef(rolled, ref_spec)[0,1]
            cc.append(val)
        
        peak = np.argmax(cc)
        if 0 < peak < len(lags)-1:
            y1, y2, y3 = cc[peak-1], cc[peak], cc[peak+1]
            denom = (y1 - 2*y2 + y3)
            if denom != 0:
                shift = lags[peak] + (y1 - y3) / (2 * denom)
            else:
                shift = lags[peak]
        else:
            shift = lags[peak]
            
        drift.append(shift)
        
    return np.array(drift)

def get_background_level(cube, disp_axis='x'):
    if disp_axis == 'x':
        if cube.shape[1] > 10:
            bg = np.concatenate([cube[:, :5, :], cube[:, -5:, :]], axis=1)
        else:
            bg = cube
    else:
        if cube.shape[2] > 10:
            bg = np.concatenate([cube[:, :, :5], cube[:, :, -5:]], axis=2)
        else:
            bg = cube
            
    return np.nanmedian(bg, axis=(1,2))

def clean_outliers(data, window=10, sigma=4.0):
    series = pd.Series(data)
    trend = series.rolling(window=window, center=True, min_periods=1).median()
    resid = series - trend
    mad = resid.rolling(window=window, center=True, min_periods=1).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )
    rolling_sigma = mad * 1.4826
    rolling_sigma = rolling_sigma.replace(0, np.nanmedian(rolling_sigma))
    mask = np.abs(resid) > (sigma * rolling_sigma)
    cleaned = series.copy()
    if mask.sum() > 0:
        cleaned[mask] = trend[mask]
    return cleaned.values

def generate_timeseries_detrending(s2_dir, s3_dir, output_dir, obs_mode, smooth_window=50, trace_width=9):
    print("    [2/3] Executing Custom Time-Series Detrending...", flush=True)
    dispersion_axis = 'y' if 'MIRI' in obs_mode.upper() else 'x'
    
    trace_array = None
    trace_center = None
    
    centroid_files = glob.glob(os.path.join(s3_dir, "**", "*centroids.csv"), recursive=True)
    if centroid_files:
        cdf = pd.read_csv(max(centroid_files, key=os.path.getmtime), comment='#')
        if 'xpos' in cdf.columns and 'ypos' in cdf.columns:
            if dispersion_axis == 'y':
                indep_vals = cdf['ypos'].values.astype(int)
                dep_vals = cdf['xpos'].values
            else:
                indep_vals = cdf['xpos'].values.astype(int)
                dep_vals = cdf['ypos'].values
            
            max_indep = np.max(indep_vals) + 1
            trace_array = np.full(max_indep, np.nan)
            mask = (indep_vals >= 0) & (indep_vals < max_indep)
            trace_array[indep_vals[mask]] = dep_vals[mask]
            trace_center = np.nanmedian(dep_vals)
    
    file_pattern = '*badpixstep.fits'
    search_path = os.path.join(s2_dir, file_pattern)
    files = sorted(glob.glob(search_path))
    if not files:
        print(f"ERROR: No files found in {search_path}")
        return None, []

    results = {'time':[], 'x':[], 'y':[], 'width':[], 'flux':[], 'bkg':[]}
    
    for f in files:
        with fits.open(f) as hdul:
            try:
                if 'INT_TIMES' in hdul:
                    t = hdul['INT_TIMES'].data['int_mid_BJD_TDB']
                else:
                    t = hdul[5].data['int_mid_BJD_TDB']
            except:
                n_int = hdul[1].data.shape[0]
                t0 = hdul[0].header.get('EXPSTART', 0) + 2400000.5
                dt = hdul[0].header.get('EFFINTTM', 0) / 86400.0
                t = t0 + np.arange(n_int) * dt

            cube = hdul[1].data
            
            if trace_center is None:
                axis_to_collapse = 1 if dispersion_axis == 'y' else 2
                prof = np.nanmedian(cube, axis=(0, axis_to_collapse))
                trace_center = np.nanargmax(prof)
            
            profiles = get_guided_spatial_profile(cube, trace_array, dispersion_axis)
            spatial_d, width_d, flux_d = measure_spatial_drift(profiles, trace_center, trace_width)
            spectral_d = measure_spectral_drift(cube, dispersion_axis)
            bkg_d = get_background_level(cube, dispersion_axis)
            
            if dispersion_axis == 'y':
                results['x'].append(spectral_d) 
                results['y'].append(spatial_d)  
            else:
                results['x'].append(spectral_d) 
                results['y'].append(spatial_d)  

            results['time'].append(t)
            results['width'].append(width_d)
            results['flux'].append(flux_d)
            results['bkg'].append(bkg_d)

    full_t = np.concatenate(results['time'])
    full_x = np.concatenate(results['x'])
    full_y = np.concatenate(results['y'])
    full_w = np.concatenate(results['width'])
    full_f = np.concatenate(results['flux'])
    full_b = np.concatenate(results['bkg'])

    df = pd.DataFrame({
        'time': full_t,
        'spectral_drift': full_x - np.nanmedian(full_x),
        'spatial_drift': full_y - np.nanmedian(full_y),
        'width': full_w - np.nanmedian(full_w),
        'total_flux': full_f / np.nanmedian(full_f),
        'background': full_b - np.nanmedian(full_b)
    })
    
    window = 20
    sigma = 4.0
    iterations = 20
    for col in ['spectral_drift', 'spatial_drift', 'width', 'background', 'total_flux']:
        for i in range(iterations-1):
            df[col] = clean_outliers(df[col], window, sigma)
    
    df['x_pos_smooth'] = df['spectral_drift'].rolling(window=smooth_window, center=True, min_periods=1).median()
    df['y_pos_smooth'] = df['spatial_drift'].rolling(window=smooth_window, center=True, min_periods=1).median()
    df['width_smooth'] = df['width'].rolling(window=smooth_window, center=True, min_periods=1).median()

    df['x_pos'] = df['x_pos_smooth'] - np.nanmedian(df['x_pos_smooth'])
    df['y_pos'] = df['y_pos_smooth'] - np.nanmedian(df['y_pos_smooth'])
    df['width'] = df['width_smooth'] - np.nanmedian(df['width_smooth'])
    
    final_columns = ['time', 'x_pos', 'y_pos', 'width']
    output_csv = os.path.join(output_dir, "timeseries_detrend.csv")
    df[final_columns].to_csv(output_csv, index=False)
    
    return output_csv, ['x_pos', 'y_pos', 'width']

# =====================================================================
# 4. CONFIG GENERATORS
# =====================================================================
def update_yaml_text(yaml_text, key, new_value):
    """
    Safely updates a key's value in a YAML string without changing formatting.
    """
    if isinstance(new_value, str):
        if new_value not in ['None', 'True', 'False', 'run', 'skip', 'optimize']:
            val_str = f"'{new_value}'"
        else:
            val_str = new_value
    elif isinstance(new_value, bool):
        val_str = str(new_value)
    elif new_value is None:
        val_str = 'None'
    elif isinstance(new_value, list):
        val_str = "[" + ", ".join(str(x) for x in new_value) + "]"
    else:
        val_str = str(new_value)

    pattern = rf"^({key}\s*:\s*)(.*)$"
    
    if re.search(pattern, yaml_text, flags=re.MULTILINE):
        yaml_text = re.sub(pattern, rf"\g<1>{val_str}", yaml_text, flags=re.MULTILINE)
    else:
        yaml_text += f"\n{key} : {val_str}"
        
    return yaml_text

def generate_run_dms_yaml(master_config, template_path, run_dir, out_dir):
    """Uses Text Replacement to keep comments and exact lists."""
    if not os.path.exists(template_path):
        print(f"\n[!] Error: Missing template {template_path}")
        sys.exit(1)
        
    with open(template_path, 'r') as f:
        yaml_text = f.read()

    meta = master_config['meta']
    common = master_config.get('common_parameters', {})
    inst_spec = master_config.get('instrument_specific', {})
    
    top_dir = os.path.abspath(master_config['io']['top_dir'])
    input_dir = os.path.abspath(os.path.join(top_dir, master_config['io']['input_dir']))
    obs_mode, filt_det = extract_metadata_from_uncal(input_dir)
    
    # Base Parameters
    yaml_text = update_yaml_text(yaml_text, 'crds_cache_path', os.path.abspath(master_config['io']['crds_cache']) + '/')
    yaml_text = update_yaml_text(yaml_text, 'input_dir', input_dir + '/')
    yaml_text = update_yaml_text(yaml_text, 'input_filetag', 'uncal')
    yaml_text = update_yaml_text(yaml_text, 'output_tag', meta['target_name'])
    yaml_text = update_yaml_text(yaml_text, 'observing_mode', obs_mode)
    yaml_text = update_yaml_text(yaml_text, 'filter_detector', filt_det)

    yaml_text = update_yaml_text(yaml_text, 'extract_method', common.get('extraction_method', 'box'))
    yaml_text = update_yaml_text(yaml_text, 'jump_threshold', common.get('jump_rejection_threshold', 15))

    yaml_text = update_yaml_text(yaml_text, 'extract_width', (common.get('extraction_aperture_half_width', 4) * 2))
    #yaml_text = update_yaml_text(yaml_text, 'extract_width', 'optimize')


    yaml_text = update_yaml_text(yaml_text, 'do_plots', common.get('generate_plots'))

    # EXACT INSTRUMENT SPECIFIC ROUTING (Mirroring your successful manual run)
    yaml_text = update_yaml_text(yaml_text, 'DarkCurrentStep', 'run')
    yaml_text = update_yaml_text(yaml_text, 'FlatFieldStep', 'run')


    if 'MIRI' in obs_mode.upper():
        yaml_text = update_yaml_text(yaml_text, 'EmiCorrStep', 'run')
        yaml_text = update_yaml_text(yaml_text, 'ResetStep', 'run')
        yaml_text = update_yaml_text(yaml_text, 'Extract2DStep', 'skip')
        yaml_text = update_yaml_text(yaml_text, 'WaveCorrStep', 'skip')

    elif 'NIRSPEC' in obs_mode.upper():
        if common.get('correct_1f_noise'):
            pass
        else:
            yaml_text = update_yaml_text(yaml_text, 'OneOverFStep_grp', 'run')
            yaml_text = update_yaml_text(yaml_text, 'OneOverFStep_int', 'run')
        yaml_text = update_yaml_text(yaml_text, 'oof_method', 'median')
        
        yaml_text = update_yaml_text(yaml_text, 'Extract2DStep', 'run')
        yaml_text = update_yaml_text(yaml_text, 'WaveCorrStep', 'run')
        
    elif 'NIRISS' in obs_mode.upper():
        if common.get('correct_1f_noise'):
            pass
        else:
            yaml_text = update_yaml_text(yaml_text, 'OneOverFStep_grp', 'run')
            yaml_text = update_yaml_text(yaml_text, 'OneOverFStep_int', 'run')
        yaml_text = update_yaml_text(yaml_text, 'Extract2DStep', 'skip')
        yaml_text = update_yaml_text(yaml_text, 'WaveCorrStep', 'skip')
        yaml_text = update_yaml_text(yaml_text, 'BackgroundStep', 'run')

        yaml_text = update_yaml_text(yaml_text, 'soss_background_file', os.path.join(top_dir, 'exoTEDRF/files/model_background256.npy'))

        if inst_spec.get('NIRISS', {}).get('extract_width_soss2') is not None:
            yaml_text = update_yaml_text(yaml_text, 'extract_width_soss2', inst_spec['NIRISS']['extract_width_soss2'])

    # Stellar params for wavelength calibration mapping
    star = master_config.get('stellar_params', {})
    if 'teff' in star: yaml_text = update_yaml_text(yaml_text, 'st_teff', star['teff'])
    if 'logg' in star: yaml_text = update_yaml_text(yaml_text, 'st_logg', star['logg'])
    if 'metallicity' in star: yaml_text = update_yaml_text(yaml_text, 'st_met', star['metallicity'])

    run_config_path = os.path.join(run_dir, f"run_DMS_{meta['target_name']}.yaml")
    with open(run_config_path, 'w') as f:
        f.write(yaml_text)
        
    return run_config_path

def generate_exotedrf_fit_config(master_config, template_path, run_dir, binned_file, detrend_file, valid_cols, exo_root):
    """Uses Safe PyYAML with Custom Dumper to prevent ParserErrors on multi-line values."""
    print("    [3/3] Generating exoTEDRF Light Curve Config...", flush=True)
    if not os.path.exists(template_path):
        print(f"\n[!] Error: Missing template {template_path}")
        sys.exit(1)

    with open(template_path, 'r') as f:
        fit_dict = yaml.safe_load(f)

    meta = master_config['meta']
    sys_pars = master_config.get('system', {})
    fit_ctrl = master_config.get('fitting_control', {})
    exo_cfg = fit_ctrl.get('exotedrf', {})
    ld = master_config.get('limb_darkening', {})
    star = master_config.get('stellar_params', {})
    free_params = fit_ctrl.get('free_parameters', [])
    use_detrending = exo_cfg.get('use_custom_detrending', False)
    
    top_dir = os.path.abspath(master_config['io']['top_dir'])
    input_dir = os.path.abspath(os.path.join(top_dir, master_config['io']['input_dir']))
    obs_mode, filt_det = extract_metadata_from_uncal(input_dir)
    
    param_map = {'per': 'per_p1', 't0': 't0_p1', 'rp': 'rp_p1_inst', 'inc': 'inc_p1', 'ecc': 'ecc_p1', 'w': 'w_p1', 'a': 'a_p1'}
    p_names, p_dists, p_vals = [], [], []
    
    for key, exotedrf_name in param_map.items():
        p_names.append(exotedrf_name)
        if key in sys_pars:
            data = sys_pars[key]
            if key in free_params:
                p_dists.append('uniform')
                p_vals.append([data['prior1'], data['prior2']])
            else:
                p_dists.append('fixed')
                p_vals.append(data['val'])
        else:
            p_dists.append('fixed')
            p_vals.append(0.0)

    # 0. Add limb darkening parameters
    p_names.extend(['u1_inst', 'u2_inst'])
    p_dists.extend(['uniform', 'uniform'])
    p_vals.extend([[0.0, 1.0], [0.0, 1.0]])

    # 1. Add linear detrending parameters if requested
    if use_detrending and detrend_file is not None and valid_cols:
        systematics = exo_cfg.get('systematics', {})
        for i in range(len(valid_cols)):
            theta_name = f'theta{i+1}_inst' 
            if theta_name in systematics:
                p_names.append(theta_name)
                p_dists.append('uniform')
                p_vals.append([systematics[theta_name]['prior1'], systematics[theta_name]['prior2']])
            else:
                p_names.append(theta_name)
                p_dists.append('uniform')
                p_vals.append([-1.0, 1.0])
                
    # 2. ALWAYS add the mandatory noise (sigma) and baseline offset (zero) parameters
    p_names.extend(['sigma_inst', 'zero_inst'])
    p_dists.extend(['loguniform', 'uniform'])
    # Giving zero_inst a uniform prior from -0.01 to 0.01 (standard for normalized light curves)
    p_vals.extend([[0.0001, 0.1], [-0.01, 0.01]])

    fit_dict['output_dir'] = os.path.abspath(exo_root) + '/'
    fit_dict['output_tag'] = meta['target_name'] 
    fit_dict['infile'] = os.path.abspath(binned_file)
    fit_dict['observing_mode'] = obs_mode
    fit_dict['detector'] = filt_det
    fit_dict['res'] = 'prebin' 
    fit_dict['params'] = p_names
    fit_dict['dists'] = p_dists
    fit_dict['values'] = p_vals
    fit_dict['planet_letter'] = 'b'
    fit_dict['ncores'] = master_config.get('common_parameters', {}).get('max_cores', 4)

    if detrend_file and use_detrending and valid_cols:
        fit_dict['lm_file'] = os.path.abspath(detrend_file)
        fit_dict['lm_parameters'] = valid_cols
    else:
        fit_dict['lm_file'] = 'None'
        if 'lm_parameters' in fit_dict:
            fit_dict['lm_parameters'] = []

    if ld.get('compute_ld', False):
        fit_dict['ld_fit_type'] = 'prior'
        fit_dict['ld_model_type'] = ld.get('model', 'quadratic')
        fit_dict['m_h'] = star.get('metallicity', 0.0)
        fit_dict['logg'] = star.get('logg', 4.5)
        fit_dict['teff'] = star.get('teff', 5000)
        fit_dict['ld_data_path'] = ld.get('exotic_ld_direc', '')
        fit_dict['stellar_model_type'] = ld.get('exotic_ld_grid', 'phoenix')
    else:
        fit_dict['ld_fit_type'] = 'free'
        fit_dict['ld_model_type'] = 'quadratic'

    run_config_path = os.path.join(run_dir, f"fit_config_{meta['target_name']}.yaml")
    with open(run_config_path, 'w') as f:
        yaml.dump(fit_dict, f, Dumper=InlineListDumper, sort_keys=False, default_flow_style=False)
        
    return run_config_path

# =====================================================================
# 5. EXECUTION LOGIC
# =====================================================================
if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, required=True)
        parser.add_argument("--stages", type=int, nargs='+', required=True)
        args = parser.parse_args()

        with open(args.config, 'r') as f:
            master_cfg = yaml.safe_load(f)

        target = master_cfg['meta']['target_name']
        top_dir = os.path.abspath(master_cfg['io']['top_dir'])
        out_dir = os.path.join(top_dir, master_cfg['io']['output_dir'], "exotedrf", target)
        run_dir = os.path.join(top_dir, "configs", "exotedrf_runs", target)
        template_dir = os.path.join(top_dir, "configs", "default_exotedrf")
        
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True) 
        
        reduction_stages = [s for s in args.stages if s in [1, 2, 3]]
        fitting_stages = [s for s in args.stages if s == 4]
        
        if reduction_stages:
            print(f"Executing exoTEDRF Reduction globally (Stages 1-3)...", flush=True)
            template_dms = os.path.join(template_dir, "run_DMS.yaml")
            dms_yaml = generate_run_dms_yaml(master_cfg, template_dms, run_dir, out_dir)
            
            dms_yaml_basename = os.path.basename(dms_yaml)
            local_dms_yaml = os.path.join(out_dir, dms_yaml_basename)
            shutil.copy(dms_yaml, local_dms_yaml)
            
            print(f"    --- Starting exoTEDRF run_DMS ---", flush=True)
            try:
                subprocess.run([sys.executable, "-m", "exotedrf.run_DMS", dms_yaml_basename], cwd=out_dir, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[!] exoTEDRF run_DMS crashed.")
                sys.exit(1)

        if fitting_stages:
            print(f"Executing exoTEDRF Fitting for stage 4...", flush=True)
            
            pipeline_dirs = glob.glob(os.path.join(out_dir, f"pipeline_outputs_directory_{target}*"))
            if not pipeline_dirs:
                print(f"[!] Error: Could not find exoTEDRF pipeline_outputs_directory in {out_dir}")
                sys.exit(1)
                
            exo_output_root = max(pipeline_dirs, key=os.path.getmtime) 
            
            s2_dir = os.path.join(exo_output_root, "Stage2")
            s3_dir = os.path.join(exo_output_root, "Stage3")
            s4_dir = os.path.join(exo_output_root, "Stage4")
            os.makedirs(s4_dir, exist_ok=True)
            
            if master_cfg['meta']['instrument'] == 'NIRISS':
                wave_min = 0.6
            else:
                wave_min = master_cfg['common_parameters']['wavelength_min']
            
            wave_max = master_cfg['common_parameters']['wavelength_max']
            num_bins = master_cfg['common_parameters']['number_of_channels']
            
            binned_file = run_custom_binning(s3_dir, s4_dir, wave_min, wave_max, num_bins=num_bins)
            
            exo_cfg = master_cfg.get('fitting_control', {}).get('exotedrf', {})
            smooth_win = exo_cfg.get('detrend_smoothing_window', 50)
            
            detrend_tuple = generate_timeseries_detrending(s2_dir, s3_dir, s4_dir, master_cfg['meta']['observing_mode'], smooth_window=smooth_win)
            if detrend_tuple is not None:
                detrend_file, valid_cols = detrend_tuple
            else:
                detrend_file, valid_cols = None, []
            
            template_fit = os.path.join(template_dir, "fit_lightcurves.yaml")
            
            fit_yaml = generate_exotedrf_fit_config(master_cfg, template_fit, run_dir, binned_file, detrend_file, valid_cols, exo_output_root)
            
            fit_yaml_basename = os.path.basename(fit_yaml)
            local_fit_yaml = os.path.join(out_dir, fit_yaml_basename)
            shutil.copy(fit_yaml, local_fit_yaml)
            
            print(f"\n    --- Starting exoTEDRF fit_lightcurves ---", flush=True)
            try:
                subprocess.run([sys.executable, "-m", "exotedrf.fit_lightcurves", fit_yaml_basename], cwd=out_dir, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n[!] exoTEDRF fit_lightcurves crashed.")
                sys.exit(1)