# workers/call_sparta.py
import yaml
import argparse
import subprocess
import os
import glob
import sys
import shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings

def generate_sparta_constants(master_config, target_out_dir, sparta_repo, sample_fits_file):
    common = master_config.get('common_parameters', {})
    
    for npy_file in glob.glob(os.path.join(sparta_repo, "*.npy")):
        shutil.copy(npy_file, target_out_dir)
        
    crds_dir = os.path.abspath(master_config['io']['crds_cache'])
    
    with fits.open(sample_fits_file) as hdul:
        hdr = hdul[0].header
        instrume = hdr.get("INSTRUME", "NIRSPEC")
        detector = hdr.get("DETECTOR", "NRS2")
        filter_val = hdr.get("FILTER", "CLEAR")
        subarray = hdr.get("SUBARRAY", "SUB2048")
        
        raw_gain = hdr.get('R_GAIN', '').replace('crds://', '')
        raw_rnoise = hdr.get('R_READNO', '').replace('crds://', '')
        raw_wcs = hdr.get('R_SPECW', hdr.get('R_WAVEL', '')).replace('crds://', '')
        
        if not raw_wcs and instrume == "MIRI": 
            raw_wcs = "jwst_miri_specwcs_0006.fits"
        
        gain_path = glob.glob(os.path.join(crds_dir, "**", raw_gain), recursive=True)
        rnoise_path = glob.glob(os.path.join(crds_dir, "**", raw_rnoise), recursive=True)
        wcs_path = glob.glob(os.path.join(crds_dir, "**", raw_wcs), recursive=True) if raw_wcs else []
        
        gain_file = gain_path[0] if gain_path else ""
        rnoise_file = rnoise_path[0] if rnoise_path else ""
        
        if not wcs_path and instrume == "MIRI":
            fallback_wcs = sorted(glob.glob(os.path.join(crds_dir, "**", "jwst_miri_specwcs_*.fits"), recursive=True))
            wcs_file = fallback_wcs[-1] if fallback_wcs else ""
        else:
            wcs_file = wcs_path[0] if wcs_path else ""
        
        data_shape = hdul["SCI"].data.shape
        raw_y, raw_x = data_shape[-2], data_shape[-1]
        
        num_ints = min(10, data_shape[0]) if len(data_shape) == 3 else 1
        sci_block = hdul["SCI"].data[:num_ints] if len(data_shape) == 3 else hdul["SCI"].data

    if instrume == "MIRI":
        rotate_val = -1 # Rotates -90 deg. (416, 72) -> (72, 416)
        dispersion_dim = raw_y  
        spatial_dim = raw_x     
        
        # After rotation, Y is the spatial axis (72px).
        y_center = 36 
        
        x_min = 20
        x_max = 295
        
        left_val = 80
        right_val = 496
        
        ap_half_width = master_config.get('instrument_specific', {}).get('MIRI', {}).get('miri_extraction_half_width', 5)
        
        bkd_reg_top = [0, 0] 
        bkd_reg_bot = [0, 0]
        
    else:
        # NIRSpec G395H
        rotate_val = 0
        spatial_dim = raw_y  
        dispersion_dim = raw_x

        if detector == "NRS1":
            x_min = 20  
        else:
            x_min = 450
        x_max = dispersion_dim - 20
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if len(data_shape) == 3:
                spatial_profile = np.nanmedian(sci_block[:, :, x_min:x_max], axis=(0, 2))
            else:
                spatial_profile = np.nanmedian(sci_block[:, x_min:x_max], axis=1)
                
        if np.all(np.isnan(spatial_profile)):
            y_center = spatial_dim // 2
        else:
            y_center = int(np.nanargmax(spatial_profile))
        
        left_val = 0
        right_val = dispersion_dim
        
        ap_half_width = common.get('extraction_aperture_half_width', 3)
        bg_half_width = common.get('bkg_mask_half_width', 8)
        
        buffer = 2
        bkd_reg_bot = [max(0, y_center - ap_half_width - buffer - bg_half_width), 
                       max(0, y_center - ap_half_width - buffer)]
        bkd_reg_top = [min(spatial_dim, y_center + ap_half_width + buffer), 
                       min(spatial_dim, y_center + ap_half_width + buffer + bg_half_width)]

    overrides = f"""\n
# ==============================================================================
# UNIFIED WRAPPER DYNAMIC OVERRIDES
# ==============================================================================
INSTRUMENT = "{instrume}"
FILTER = "{filter_val}"
SUBARRAY = "{subarray}"
ROTATE = {rotate_val}

LEFT = {left_val}
RIGHT = {right_val}

Y_CENTER = {y_center}
X_MIN = {x_min}
X_MAX = {x_max}

OPT_EXTRACT_WINDOW = {ap_half_width}
BKD_REG_BOT = {bkd_reg_bot}
BKD_REG_TOP = {bkd_reg_top}

TOP_MARGIN = 0
N_REF = 4
ONE_OVER_F_WINDOW_LEFT = 5
ONE_OVER_F_WINDOW_RIGHT = {x_max}

GAIN_FILE = "{gain_file}"
RNOISE_FILE = "{rnoise_file}"
WCS_FILE = "{wcs_file}"
# ==============================================================================
"""
    target_constants = os.path.join(target_out_dir, "constants.py")
    shutil.copy(os.path.join(sparta_repo, "constants.py"), target_constants)
    with open(target_constants, "a") as f: f.write(overrides)

def inject_missing_sparta_extensions(fits_file):
    try:
        with fits.open(fits_file, mode='update') as hdul:
            needs_flush = False
            if 'ERR' in hdul:
                err_data = hdul['ERR'].data
                bad_mask = (err_data <= 0) | np.isnan(err_data)
                if np.any(bad_mask):
                    valid_median = np.nanmedian(err_data[~bad_mask])
                    if np.isnan(valid_median): valid_median = 1.0
                    err_data[bad_mask] = valid_median
                    hdul['ERR'].data = err_data
                    needs_flush = True

            if 'RNOISE' not in hdul and 'VAR_RNOISE' in hdul:
                rnoise_std = np.sqrt(np.abs(hdul['VAR_RNOISE'].data))
                if rnoise_std.ndim == 3:
                    rnoise_std = np.nanmedian(rnoise_std, axis=0)
                hdul.append(fits.ImageHDU(data=rnoise_std, name='RNOISE'))
                needs_flush = True
                    
            if needs_flush: hdul.flush()
    except Exception: pass

def sanitize_optx1d_files(target_out_dir):
    optx_files = glob.glob(os.path.join(target_out_dir, "optx1d_*.fits"))
    for f in optx_files:
        try:
            with fits.open(f, mode='update') as hdul:
                for i in range(2, len(hdul)):
                    flux, err = hdul[i].data['FLUX'], hdul[i].data['ERROR']
                    mask = np.isnan(flux) | np.isnan(err) | np.isinf(flux) | np.isinf(err)
                    if np.any(mask):
                        x = np.arange(len(flux))
                        valid = ~mask
                        if np.any(valid): 
                            flux[mask] = np.interp(x[mask], x[valid], flux[valid])
                            err[mask] = np.interp(x[mask], x[valid], err[valid])
                        else:
                            flux[:], err[:] = 0.0, 1e10
                        hdul[i].data['FLUX'], hdul[i].data['ERROR'] = flux, err
                hdul.flush()
        except Exception: pass

def execute_sparta_safely(script_name, args_list, target_out_dir, sparta_repo):
    script_path = os.path.join(sparta_repo, script_name)
    with open(script_path, "r") as f: code = f.read()

    patch_header = (
        "import os, sys, multiprocessing, matplotlib\n"
        "matplotlib.use('Agg')\n"
        "if sys.platform != 'win32':\n"
        "    try: multiprocessing.set_start_method('fork', force=True)\n"
        "    except RuntimeError: pass\n"
    )
    code = patch_header + code
    
    if script_name == "remove_bkd.py":
        miri_patch = """
def remove_bkd_miri(data, err, dq):
    import astropy.stats
    import numpy as np
    bkd_im = np.zeros(data.shape)
    bkd_var_im = np.zeros(err.shape)
    for i in range(data.shape[0]):
        bkd_cols = np.concatenate((data[i, :, 10:25], data[i, :, 47:62]), axis=1)
        bkd_cols = astropy.stats.sigma_clip(bkd_cols, axis=1)
        bkd_err_cols = np.concatenate((err[i, :, 10:25], err[i, :, 47:62]), axis=1)
        bkd = np.ma.mean(bkd_cols, axis=1)
        bkd_var = np.sum(bkd_err_cols**2, axis=1) / bkd_err_cols.shape[1]**2
        bkd_im[i] = bkd[:, np.newaxis]
        bkd_var_im[i] = bkd_var[:, np.newaxis]
    return data - bkd_im, err, bkd_im, np.sqrt(bkd_var_im), dq
"""
        code = code.replace("def remove_bkd(data, err, dq):", miri_patch + "\ndef remove_bkd(data, err, dq):")
        old_call = "data_no_bkd, err, bkd, err_bkd, dq = remove_bkd("
        new_call = "data_no_bkd, err, bkd, err_bkd, dq = remove_bkd_miri(hdul['SCI'].data, hdul['ERR'].data, hdul['DQ'].data) if hdul[0].header['INSTRUME'] == 'MIRI' else remove_bkd("
        code = code.replace(old_call, new_call)

    if script_name == "get_positions_and_median_image.py":
        code = code.replace("from constants import ", "from constants import ROTATE, ")
        
        # 1. Cleanly replace the dangerous original fix_outliers function
        old_fix_outliers = """def fix_outliers(data, badpix, sigma=5):
    for r in range(TOP_MARGIN, data.shape[0]):
        cols = np.arange(data.shape[1])
        good = ~badpix[r]
        data[r] = np.interp(cols, cols[good], data[r][good])

        good = ~astropy.stats.sigma_clip(data[r], sigma).mask
        data[r] = np.interp(cols, cols[good], data[r][good])"""
        
        new_fix_outliers = """def fix_outliers(data, badpix, sigma=5):
    for r in range(TOP_MARGIN, data.shape[0]):
        cols = np.arange(data.shape[1])
        good = ~badpix[r]
        if np.sum(good) > 0:
            data[r] = np.interp(cols, cols[good], data[r][good])
            good2 = ~astropy.stats.sigma_clip(data[r], sigma).mask
            if np.sum(good2) > 0:
                data[r] = np.interp(cols, cols[good2], data[r][good2])
        else:
            data[r] = 0.0"""
        code = code.replace(old_fix_outliers, new_fix_outliers)

        # 2. Inject the Rotation Logic natively in-memory so tracking bounds align perfectly
        rot_patch = """
        data = hdul["SCI"].data
        error = hdul["ERR"].data
        dq = hdul["DQ"].data
        if ROTATE != 0:
            data = np.array([np.rot90(d, k=ROTATE) for d in data])
            error = np.array([np.rot90(d, k=ROTATE) for d in error])
            dq = np.array([np.rot90(d, k=ROTATE) for d in dq])
"""
        code = code.replace('        data = hdul["SCI"].data\n        error = hdul["ERR"].data\n        dq = hdul["DQ"].data', rot_patch)

    if script_name == "optimal_extract.py":
        code = code.replace("from constants import ", "from constants import ROTATE, ")
        code = code.replace('kind="cubic"', 'kind="linear"')
        
        optx_patch = """
            sci_i = hdul['SCI'].data[i]
            err_i = hdul['ERR'].data[i]
            bkd_i = hdul['BKD'].data[i]
            dq_i = hdul['DQ'].data[i]
            rn_i = hdul['RNOISE'].data
            if ROTATE != 0:
                sci_i = np.rot90(sci_i, k=ROTATE)
                err_i = np.rot90(err_i, k=ROTATE)
                bkd_i = np.rot90(bkd_i, k=ROTATE)
                dq_i = np.rot90(dq_i, k=ROTATE)
                if rn_i.shape == hdul['SCI'].data[i].shape: rn_i = np.rot90(rn_i, k=ROTATE)
            
            data = sci_i[:, X_MIN:X_MAX]
            err = err_i[:, X_MIN:X_MAX]
"""
        old_slice = '            data = hdul["SCI"].data[i,:,X_MIN:X_MAX]\n            err = hdul["ERR"].data[i,:,X_MIN:X_MAX]'
        code = code.replace(old_slice, optx_patch)
        
        code = code.replace('hdul["SCI"].data[i][s]', 'sci_i[s]')
        code = code.replace('hdul["BKD"].data[i][s]', 'bkd_i[s]')
        code = code.replace('hdul["DQ"].data[i][s] != 0', 'dq_i[s] != 0')
        code = code.replace('hdul["RNOISE"].data[s]', 'rn_i[s]')
        code = code.replace('hdul["BKD"].data[i][s].mean(axis=0)', 'bkd_i[s].mean(axis=0)')

    plot_name = script_name.replace(".py", "_diagnostic.png")
    safe_plot_cmd = f"plt.savefig('diagnostics/{plot_name}', bbox_inches='tight'); plt.close('all')"
    code = code.replace("plt.show()", safe_plot_cmd)

    patched_script = os.path.join(target_out_dir, script_name)
    with open(patched_script, "w") as f: f.write(code)

    env = os.environ.copy()
    env["PYTHONPATH"] = sparta_repo + os.pathsep + env.get("PYTHONPATH", "")
    process = subprocess.Popen([sys.executable, script_name] + args_list, cwd=target_out_dir, stdout=sys.stdout, stderr=sys.stderr, env=env)
    process.wait()

    if os.path.exists(patched_script): os.remove(patched_script)
    return process.returncode

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stages", type=int, nargs='+', required=True) 
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        master_cfg = yaml.safe_load(f)

    target = master_cfg['meta']['target_name']
    top_dir = os.path.abspath(master_cfg['io']['top_dir'])
    input_dir = os.path.abspath(os.path.join(top_dir, master_cfg['io']['input_dir']))
    target_out_dir = os.path.abspath(os.path.join(master_cfg['io']['output_dir'], "sparta", target))
    os.makedirs(target_out_dir, exist_ok=True)
    sparta_repo = os.path.abspath(os.path.join(top_dir, "sparta"))
    diag_dir = os.path.join(target_out_dir, "diagnostics")
    os.makedirs(diag_dir, exist_ok=True)

    print(f"Executing SPARTA for stages {args.stages}...", flush=True)

    # =========================================================================
    # STAGE 1: STScI PIPELINE
    # =========================================================================
    if 1 in args.stages:
        existing_s1 = glob.glob(os.path.join(target_out_dir, "*rateints.fits"))
        if existing_s1:
            print("    [!] Stage 1 outputs already detected. Resuming...", flush=True)
        else:
            print("    --- Starting SPARTA Stage 1 (STScI Detector1Pipeline) ---", flush=True)
            uncal_files = sorted(glob.glob(os.path.join(input_dir, "*uncal.fits")))
            if not uncal_files: sys.exit(1)
            
            jump_threshold = master_cfg.get('common_parameters', {}).get('jump_rejection_threshold', 8.0)
            cores = master_cfg.get('common_parameters', {}).get('max_cores', 4)

            for uncal_file in uncal_files:
                print(f"    Processing {os.path.basename(uncal_file)}...", flush=True)
                cmd = ["strun", "calwebb_detector1", uncal_file, "--output_dir", target_out_dir, 
                       "--save_calibrated_ramp=false", f"--steps.ramp_fit.maximum_cores={cores}",
                       f"--steps.jump.rejection_threshold={jump_threshold}", f"--steps.jump.maximum_cores={cores}"]
                if subprocess.Popen(cmd, cwd=target_out_dir, stdout=sys.stdout, stderr=sys.stderr).wait() != 0: sys.exit(1)
                
            print("    [+] Generating Stage 1 Detector Pipeline Diagnostics...", flush=True)
            try:
                rateints_files = sorted(glob.glob(os.path.join(target_out_dir, "*rateints.fits")))
                if uncal_files and rateints_files:
                    with fits.open(uncal_files[0]) as hdul_uncal, fits.open(rateints_files[0]) as hdul_rate:
                        uncal_sci = hdul_uncal['SCI'].data[0, -1, :, :]
                        rate_sci = hdul_rate['SCI'].data[0, :, :]
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
                    im1 = ax1.imshow(uncal_sci, aspect='auto', origin='lower', vmin=np.nanpercentile(uncal_sci, 5), vmax=np.nanpercentile(uncal_sci, 95))
                    ax1.set_title("Raw Uncal (Integration 0, Last Group)")
                    fig.colorbar(im1, ax=ax1)
                    im2 = ax2.imshow(rate_sci, aspect='auto', origin='lower', vmin=np.nanpercentile(rate_sci, 5), vmax=np.nanpercentile(rate_sci, 95))
                    ax2.set_title("Fitted Rateints (Integration 0)")
                    fig.colorbar(im2, ax=ax2)
                    plt.savefig(os.path.join(diag_dir, "stage1_uncal_vs_rateints.png"), bbox_inches='tight', dpi=150)
                    plt.close()
            except Exception: pass
            print("\n=== STAGE 1 COMPLETE ===\n", flush=True)

    # =========================================================================
    # STAGE 2: BACKGROUND SUBTRACTION
    # =========================================================================
    if 2 in args.stages:
        existing_s2 = glob.glob(os.path.join(target_out_dir, "cleaned_*rateints.fits"))
        if existing_s2:
            print("    [!] Stage 2 outputs already detected. Resuming...", flush=True)
        else:
            print("    --- Starting SPARTA Stage 2 (Background Subtraction) ---", flush=True)
            rateints_files = sorted(glob.glob(os.path.join(target_out_dir, "*rateints.fits")))
            generate_sparta_constants(master_cfg, target_out_dir, sparta_repo, rateints_files[0])
            file_basenames = [os.path.basename(f) for f in rateints_files]
            
            if execute_sparta_safely("remove_bkd.py", file_basenames, target_out_dir, sparta_repo) != 0: sys.exit(1)

            print("    [+] Generating Background Diagnostics for all files...", flush=True)
            for f in file_basenames:
                try:
                    c_file = os.path.join(target_out_dir, "cleaned_" + f)
                    with fits.open(c_file) as hdul:
                        sci, bkd = hdul['SCI'].data[0], hdul['BKD'].data[0]
                        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
                        im1 = ax1.imshow(sci, aspect='auto', origin='lower', vmin=np.nanpercentile(sci, 5), vmax=np.nanpercentile(sci, 95))
                        ax1.set_title(f"Post-Cleaned SCI: {f}")
                        fig.colorbar(im1, ax=ax1)
                        im2 = ax2.imshow(bkd, aspect='auto', origin='lower', vmin=np.nanpercentile(bkd, 5), vmax=np.nanpercentile(bkd, 95))
                        ax2.set_title("Subtracted BKD Model")
                        fig.colorbar(im2, ax=ax2)
                        plt.tight_layout()
                        plt.savefig(os.path.join(diag_dir, f"stage2_bkd_{f.replace('.fits', '.png')}"), bbox_inches='tight')
                        plt.close()
                except Exception: pass
            print("\n=== STAGE 2 COMPLETE ===\n", flush=True)

    # =========================================================================
    # STAGE 3: OPTIMAL EXTRACTION
    # =========================================================================
    if 3 in args.stages:
        print("    --- Starting SPARTA Stage 3 (Optimal Spectral Extraction) ---", flush=True)
        cleaned_files = sorted(glob.glob(os.path.join(target_out_dir, "cleaned_*rateints.fits")))
        
        print("    [+] Preparing FITS arrays (ERR and RNOISE) for SPARTA math...", flush=True)
        for f in cleaned_files: 
            inject_missing_sparta_extensions(f)
            
        file_basenames = [os.path.basename(f) for f in cleaned_files]
        if execute_sparta_safely("get_positions_and_median_image.py", file_basenames, target_out_dir, sparta_repo) != 0: sys.exit(1)
        
        print("    [+] Generating Stage 3 Median Image Diagnostic...", flush=True)
        try:
            med_img_path = os.path.join(target_out_dir, "median_image.npy")
            if os.path.exists(med_img_path):
                med_img = np.load(med_img_path)
                plt.figure(figsize=(12, 4))
                plt.imshow(med_img, aspect='auto', origin='lower', vmin=np.nanpercentile(med_img, 2), vmax=np.nanpercentile(med_img, 98))
                plt.title("Stage 3 Diagnostic: Master Median Trace Image")
                plt.colorbar(label="Signal")
                plt.tight_layout()
                plt.savefig(os.path.join(diag_dir, "stage3_median_image_diagnostic.png"), bbox_inches='tight', dpi=150)
                plt.close()
        except Exception: pass

        print("    [+] Running optimal Horne extraction...", flush=True)
        if execute_sparta_safely("optimal_extract.py", file_basenames, target_out_dir, sparta_repo) != 0: sys.exit(1)
        
        print("    [+] Sanitizing extracted spectra to prevent NaN crashes...", flush=True)
        sanitize_optx1d_files(target_out_dir)

        print("    [+] Gathering and filtering extracted 1D spectra into data.pkl...", flush=True)
        optx_files = sorted(glob.glob(os.path.join(target_out_dir, "optx1d_*.fits")))
        optx_basenames = [os.path.basename(f) for f in optx_files]
        if execute_sparta_safely("gather_and_filter.py", optx_basenames, target_out_dir, sparta_repo) != 0: sys.exit(1)
        
        for png_file in glob.glob(os.path.join(target_out_dir, "*.png")):
            try: shutil.move(png_file, os.path.join(diag_dir, os.path.basename(png_file)))
            except Exception: pass
            
        print("\n=== STAGE 3 COMPLETE ===\n", flush=True)

    # =========================================================================
    # STAGE 4: TRANSIT FITTING
    # =========================================================================
    if 4 in args.stages:
        print("    --- Starting SPARTA Stage 4 (Transit Fitting) ---", flush=True)
        
        sparta_params = master_cfg.get("fitting_control", {}).get("sparta", {})
        max_cores = master_cfg.get('common_parameters', {}).get('max_cores', 4)
        
        system = master_cfg.get("system", master_cfg.get("planet_parameters", master_cfg.get("parameters", {})))
        star = master_cfg.get("stellar_params", master_cfg.get("star_parameters", {}))
        
        def fetch_prior(d, key, default_val):
            block = d.get(key, {})
            if isinstance(block, dict):
                return block.get('val', default_val), block.get('prior_type', 'U'), block.get('prior1', 0.0), block.get('prior2', 1.0)
            return block, 'U', 0.0, 1.0
            
        t0_val, t0_ptype, t0_p1, t0_p2 = fetch_prior(system, 't0', 0)
        per_val, per_ptype, per_p1, per_p2 = fetch_prior(system, 'per', 0)
        rp_val, rp_ptype, rp_p1, rp_p2 = fetch_prior(system, 'rp', 0)
        a_val, a_ptype, a_p1, a_p2 = fetch_prior(system, 'a', 0)
        inc_val, inc_ptype, inc_p1, inc_p2 = fetch_prior(system, 'inc', 0)
        ecc_val, _, _, _ = fetch_prior(system, 'ecc', 0)
        w_val, _, _, _ = fetch_prior(system, 'w', 90)
        teff_val, _, _, _ = fetch_prior(star, 'teff', 3000)

        # ---> AUTO BJD TIMELINE ALIGNER <---
        import pickle
        try:
            with open(os.path.join(target_out_dir, "data.pkl"), "rb") as f:
                pkl_data = pickle.load(f)
                data_times = pkl_data["times"]
                t_med = np.median(data_times)
                
                # Detect if YAML t0 is truncated (e.g. 59964) while JWST data is BJD (e.g. 2459964)
                if t_med > 2400000 and t0_val < 2400000:
                    diff = t_med - t0_val
                    # Automatically snap to the exact timezone offset (usually +2400000.0 or +2400000.5)
                    offset = round(diff * 2) / 2
                    t0_val += offset
                    print(f"    [+] Auto-Aligned YAML t0 to JWST BJD timeline (Added offset: {offset})", flush=True)
                    
                # Roll epochs to center the transit perfectly in this specific observation window
                epochs = round((t_med - t0_val) / per_val)
                t0_val = t0_val + epochs * per_val
        except Exception as e:
            print(f"    [!] Warning: Could not auto-align t0: {e}", flush=True)

        # ---> THE ULTIMATE MCMC PRIOR PATCH <---
        print(f"    [+] Patching SPARTA MCMC Engine with {max_cores} Cores and Strict YAML Priors...", flush=True)
        with open(os.path.join(sparta_repo, "emcee_methods.py"), "r") as f: emcee_code = f.read()
        
        # 1. Multiprocessing and Anti-Hang
        pool_inj = f"import multiprocessing\n    pool = multiprocessing.Pool({max_cores})\n    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, pool=pool,"
        emcee_code = emcee_code.replace("sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob,", pool_inj)
        emcee_code = emcee_code.replace("if np.random.randint(0, 1000) == 0:", "if False:") 
        emcee_code = emcee_code.replace("pdb.set_trace()", "return -np.inf") 
        
        # 2. Fix SPARTA's backward safety checks to permanently silence the 'arccos' spam
        bad_ld_code = "u1 = 2*np.sqrt(q1) * q2\n    u2 = np.sqrt(q1) * (1 - 2*q2)\n    if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1: return -np.inf"
        good_ld_code = "if q1 < 0 or q1 > 1 or q2 < 0 or q2 > 1: return -np.inf\n    u1 = 2*np.sqrt(q1) * q2\n    u2 = np.sqrt(q1) * (1 - 2*q2)"
        emcee_code = emcee_code.replace(bad_ld_code, good_ld_code)
        
        bad_inc_code = "inc = np.arccos(b/a_star) * 180/np.pi"
        good_inc_code = "if a_star <= 0 or abs(b) >= a_star: return -np.inf\n    inc = np.arccos(b/a_star) * 180/np.pi"
        emcee_code = emcee_code.replace(bad_inc_code, good_inc_code)
        
        # 3. Inject strict Uniform and Normal Priors natively mapped from your YAML
        base_lnprob = "result = -0.5*(np.sum(residuals**2/scaled_errors**2 - np.log(1.0/scaled_errors**2)))"
        prior_str = base_lnprob + "\n"
        if a_ptype == "N": prior_str += f"    result -= 0.5 * ((a_star - {a_val}) / {a_p2})**2\n"
        if inc_ptype == "N": prior_str += f"    result -= 0.5 * ((inc - {inc_val}) / {inc_p2})**2\n"
        if t0_ptype == "N": prior_str += f"    result -= 0.5 * (transit_offset / {t0_p2})**2\n"
        if rp_ptype == "U": prior_str += f"    if rp < {rp_p1} or rp > {rp_p2}: return -np.inf\n"
        emcee_code = emcee_code.replace(base_lnprob, prior_str)

        with open(os.path.join(target_out_dir, "emcee_methods.py"), "w") as f: f.write(emcee_code)

        # Generate the static transit parameter file
        cfg_content = f"""[DEFAULT]
t0: {t0_val}
per: {per_val}
rp: {rp_val}
a: {a_val}
inc: {inc_val}
ecc: {ecc_val}
w: {w_val}
limb_dark_coeffs: {master_cfg.get("limb_darkening", {}).get("default_ld_coeffs", [0.1, 0.1])}
Ts: {teff_val}
"""
        with open(os.path.join(target_out_dir, "sparta.cfg"), "w") as f: f.write(cfg_content)
            
        w_min_nm = float(master_cfg.get('common_parameters', {}).get('wavelength_min', 3.8)) * 1000
        w_max_nm = float(master_cfg.get('common_parameters', {}).get('wavelength_max', 5.1)) * 1000
        num_channels = int(master_cfg.get('common_parameters', {}).get('number_of_channels', 10))
        
        execution_plan = [("WLC", w_min_nm, w_max_nm)]
        if num_channels > 1:
            bin_edges = np.linspace(w_min_nm, w_max_nm, num_channels + 1)
            for i in range(num_channels):
                execution_plan.append((f"bin_{i:02d}", bin_edges[i], bin_edges[i+1]))
                
        print(f"    [+] Generated MCMC Execution Plan: {len(execution_plan)} light curves", flush=True)

        for name, b_start, b_end in execution_plan:
            print(f"\n    [+] Launching MCMC Fitter for [{name}] ({b_start:.1f} - {b_end:.1f} nm) across {sparta_params.get('num_walkers', 50)} walkers...", flush=True)
            
            out_folder = f"mcmc_chain_{name}"
            os.makedirs(os.path.join(target_out_dir, out_folder), exist_ok=True)
            
            args_list = [
                "sparta.cfg", str(b_start), str(b_end), 
                "-b", str(sparta_params.get('bin_size', 1)), 
                "--burn-in-runs", str(sparta_params.get('burn_in_runs', 500)),
                "--production-runs", str(sparta_params.get('production_runs', 1000)),
                "--num-walkers", str(sparta_params.get('num_walkers', 50)),
                "-e", str(sparta_params.get('exclude_beginning_integrations', 10)),
                "-o", out_folder
            ]
            
            if execute_sparta_safely("extract_transit.py", args_list, target_out_dir, sparta_repo) != 0: 
                print(f"\n[!] SPARTA crashed during transit fitting for {name}.", flush=True)
                sys.exit(1)
                
            default_plot = os.path.join(diag_dir, "extract_transit_diagnostic.png")
            if os.path.exists(default_plot): os.rename(default_plot, os.path.join(diag_dir, f"stage4_corner_plot_{name}.png"))
            if os.path.exists(os.path.join(target_out_dir, "fit_and_residuals.png")):
                shutil.move(os.path.join(target_out_dir, "fit_and_residuals.png"), os.path.join(diag_dir, f"stage4_lc_fit_{name}.png"))
                
            for txt_file in ["white_lightcurve.txt", "white_light_result.txt"]:
                src = os.path.join(target_out_dir, txt_file)
                if os.path.exists(src): shutil.move(src, os.path.join(target_out_dir, out_folder, txt_file))
                
        print("\n    [+] Compiling Final Transmission Spectrum...", flush=True)
        waves, depths, errs_lower, errs_upper = [], [], [], []
        
        for name, _, _ in execution_plan:
            if name == "WLC": continue
            res_file = os.path.join(target_out_dir, f"mcmc_chain_{name}", "white_light_result.txt")
            if os.path.exists(res_file):
                try:
                    data = np.loadtxt(res_file, comments="#")
                    if data.size > 0 and data.ndim > 0:
                        # SPARTA's internal text file outputs are scaled to Microns natively by algorithms.py
                        waves.append(np.mean([data[0], data[1]]))
                        depths.append(data[2] * 1e6) 
                        errs_lower.append(data[3] * 1e6)
                        errs_upper.append(data[4] * 1e6)
                except Exception: pass
                
        if waves:
            plt.figure(figsize=(10, 5))
            plt.errorbar(waves, depths, yerr=[errs_lower, errs_upper], fmt='o', color='black', capsize=3)
            plt.title(f"Transmission Spectrum ({target}) - SPARTA")
            plt.xlabel("Wavelength (um)")
            plt.ylabel("Transit Depth (ppm)")
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(diag_dir, "stage4_transmission_spectrum.png"), bbox_inches='tight', dpi=200)
            plt.close()
            print(f"        -> Saved to: {os.path.join(diag_dir, 'stage4_transmission_spectrum.png')}", flush=True)

        if os.path.exists(os.path.join(target_out_dir, "emcee_methods.py")):
            os.remove(os.path.join(target_out_dir, "emcee_methods.py"))
            
        print("\n=== STAGE 4 COMPLETE ===\n", flush=True)