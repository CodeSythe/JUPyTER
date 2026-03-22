# workers/call_eureka.py

import os
import sys
import yaml
import argparse
import multiprocessing
import glob

def get_eureka_template_name(stage, instrument, observing_mode):
    inst = instrument.lower()
    mode = observing_mode.split('/')[-1].lower() if '/' in observing_mode else observing_mode.lower()

    if stage == 1:
        if inst in ['nirspec', 'niriss', 'nircam']: return "S1_nirx_template.ecf"
        elif inst == 'miri': return "S1_miri_template.ecf"
    elif stage in [2, 3]:
        if inst == 'nirspec': return f"S{stage}_nirspec_fs_template.ecf"
        elif inst == 'niriss': return f"S{stage}_niriss_soss_template.ecf"
        elif inst == 'miri':
            if mode == 'lrs': return f"S{stage}_miri_lrs_template.ecf"
            else: return f"S{stage}_miri_photometry_template.ecf"
        elif inst == 'nircam':
            if mode in ['wfss', 'grism']: return f"S{stage}_nircam_wfss_template.ecf"
            else: return f"S{stage}_nircam_photometry_template.ecf"
    elif stage == 4:
        return "S4_template.ecf"
    elif stage == 5:
        return "S5_template.ecf"
    elif stage == 6:
        return "S6_template.ecf"
    return None

def get_eureka_core_string(requested_cores):
    try: req = int(requested_cores)
    except ValueError: return "'all'"
    total_system_cores = multiprocessing.cpu_count()
    if req >= total_system_cores: return "'all'"
    elif req >= total_system_cores / 2: return "'half'"
    elif req >= total_system_cores / 4: return "'quarter'"
    else: return "'quarter'"

def find_eureka_input_dir(target_out_root, prev_stage, top_dir):
    base_path = os.path.join(target_out_root, f"Stage{prev_stage}")
    if prev_stage == 3: search = glob.glob(os.path.join(base_path, "**", "*SpecData.h5"), recursive=True)
    elif prev_stage == 4: search = glob.glob(os.path.join(base_path, "**", "*LCData.h5"), recursive=True)
    elif prev_stage == 5: search = glob.glob(os.path.join(base_path, "**", "*FitData.h5"), recursive=True)
    else: return os.path.relpath(base_path, top_dir) + "/"
        
    if search:
        latest_file = max(search, key=os.path.getctime)
        abs_dir = os.path.dirname(latest_file)
        return os.path.relpath(abs_dir, top_dir) + "/"
    return os.path.relpath(base_path, top_dir) + "/"

def edit_and_save_ecf(master_config, template_file, run_file, target_stage, top_dir, target_out_root):
    common = master_config.get('common_parameters', {})
    meta = master_config.get('meta', {})
    inst_spec = master_config.get('instrument_specific', {})
    
    if target_stage == 1:
        input_dir = master_config['io']['input_dir']
        if not input_dir.endswith('/'): input_dir += '/'
    else:
        input_dir = find_eureka_input_dir(target_out_root, target_stage - 1, top_dir)
        
    rel_out_root = os.path.relpath(target_out_root, top_dir)
    output_dir = os.path.join(rel_out_root, f"Stage{target_stage}") + "/"
    os.makedirs(os.path.join(top_dir, output_dir), exist_ok=True)

    overrides = {
        "topdir": f"{top_dir}/",
        "inputdir": input_dir,
        "outputdir": output_dir,
        "ncpu": common.get('max_cores', 1),
    }

    if target_stage == 1:
        overrides["maximum_cores"] = get_eureka_core_string(common.get('max_cores', 1))
        overrides["jump_rejection_threshold"] = common.get('jump_rejection_threshold', 4.0)
        if meta.get('instrument', '').lower() == 'miri':
            if inst_spec.get('MIRI', {}).get('run_emicorr'):
                overrides["skip_emicorr"] = False
    elif target_stage == 3:
        overrides["bg_hw"] = common.get('bkg_mask_half_width', 8)
        overrides["spec_hw"] = common.get('extraction_aperture_half_width', 4)
        overrides["bg_method"] = common.get('bkg_method', 'median')
        if inst_spec.get('NIRSpec', {}).get('sensor').upper() == 'NRS1':
            overrides["xwindow"] = [550,2040]
    elif target_stage == 4:
        num_bins = common.get('number_of_channels', 1)
        overrides["compute_white"] = "False" if num_bins == 1 else "True"
        overrides["nspecchan"] = num_bins
        if meta.get('instrument', '').lower() == 'niriss':
            overrides['s4_order'] = inst_spec.get('NIRISS', {}).get('s4_order', 1)
        if num_bins > 1: overrides["nspecchan"] = num_bins
        if meta.get('instrument', '').lower() == 'niriss' and inst_spec.get('NIRISS', {}).get('order') == 2:
            overrides["wave_min"] = 0.63
            overrides["wave_max"] = 1.0
        else:    
            overrides["wave_min"] = common.get('wavelength_min')
            overrides["wave_max"] = common.get('wavelength_max')
        #overrides["sigma"] = 4
        #overrides["fittype"] = "smooth"
        #overrides["window"] = 20
        
        ld = master_config.get('limb_darkening', {})
        if ld.get('compute_ld', False):
            overrides["compute_ld"] = "True"
            star = master_config.get('stellar_params', {})
            overrides["metallicity"] = star.get('metallicity', 0.0)
            overrides["teff"] = star.get('teff', 5000)
            overrides["logg"] = star.get('logg', 4.5)
            overrides["exotic_ld_direc"] = ld.get('exotic_ld_direc', '')
            overrides["exotic_ld_grid"] = ld.get('exotic_ld_grid', 'mps1')
            overrides["exotic_ld_file"] = ld.get('exotic_ld_file', 'Custom_throughput')
        else: overrides["compute_ld"] = "False"
    
    elif target_stage == 5:
        fit = master_config.get('fitting_control', {})
        eureka_cfg = fit.get('eureka', {})
        
        overrides['fit_par'] = f"S5_{meta['target_name']}_fit_par.epf"
        overrides["fit_method"] = fit.get('sampler', 'dynesty')
        overrides["run_nlive"] = 500
        overrides['run_sample'] = 'rwalk'
        
        # --- inject run_myfuncs ---
        if 'run_myfuncs' in eureka_cfg:
            funcs_str = ", ".join(eureka_cfg['run_myfuncs'])
            overrides["run_myfuncs"] = f"[{funcs_str}]"
        else:
            overrides["run_myfuncs"] = "[batman_tr, polynomial]" # Fallback
            
        ld = master_config.get('limb_darkening', {})
        if ld.get('compute_ld', False):
            overrides["use_generate_ld"] = "exotic-ld"
            overrides['recenter_ld_prior'] = True

    used_keys = set()
    with open(template_file, 'r') as f_in, open(run_file, 'w') as f_out:
        for line in f_in:
            parts = line.split()
            if len(parts) > 0:
                key = parts[0]
                if key in overrides:
                    f_out.write(f"{key}  {overrides[key]}\n")
                    used_keys.add(key)
                    continue
            f_out.write(line)
            
        missing_keys = set(overrides.keys()) - used_keys
        if missing_keys:
            f_out.write("\n# --- Injected Overrides ---\n")
            for key in missing_keys:
                f_out.write(f"{key}  {overrides[key]}\n")

def generate_epf(master_config, run_dir):
    """Generates the EPF file for Stage 5 based strictly on YAML priors."""
    sys_pars = master_config.get('system', {})
    fit_ctrl = master_config.get('fitting_control', {})
    eureka_cfg = fit_ctrl.get('eureka', {})
    free_params = fit_ctrl.get('free_parameters', [])
    target = master_config['meta']['target_name']
    
    ld_config = master_config.get('limb_darkening', {})
    ld_law = ld_config.get('model', 'quadratic')
    
    s5_epf_path = os.path.join(run_dir, f"S5_{target}_fit_par.epf")
    with open(s5_epf_path, 'w') as f:
        f.write("# Parameter Name, Value, Free/Fixed, Prior1, Prior2, Prior Type\n")
        
        # 1. Physical Parameters
        for p_name in ["rp", "t0", "inc", "a", "per", "ecc", "w"]:
            if p_name in sys_pars:
                data = sys_pars[p_name]
                state = 'free' if p_name in free_params else 'fixed'
                f.write(f"{p_name} {data['val']} '{state}' {data['prior1']} {data['prior2']} '{data['prior_type']}'\n")
                
        # 2. Limb Darkening 
        f.write(f"limb_dark '{ld_law}' 'independent'\n")
        if ld_law == 'quadratic':
            f.write("u1 0.2 'free' 0.0 0.1 'N'\n")
            f.write("u2 0.2 'free' 0.0 0.1 'N'\n")
        elif ld_law == 'linear':
            f.write("u1 0.2 'free' 0.0 1.0 'U'\n")
            
        # 3. Systematics & Polynomial Baseline
        f.write("\n# Polynomial Baseline\n")
        f.write("c0 1.0 'free' 0.9 1.1 'U'\n")
        f.write("c1 0.0 'free' -0.1 0.1 'U'\n")
        f.write("scatter_mult 1.1 'free' 0.8 10 'U'\n")

        # 4. --- Custom Systematics & GP Parameters ---
        systematics = eureka_cfg.get('systematics', {})
        if systematics:
            f.write("\n# Decorrelation & GP Parameters\n")
            for p_name, data in systematics.items():
                # We default these to 'free' since you explicitly added them to the YAML
                f.write(f"{p_name} {data['val']} 'free' {data['prior1']} {data['prior2']} '{data['prior_type']}'\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--stages", type=int, nargs='+', required=True) 
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        master_cfg = yaml.safe_load(f)

    target = master_cfg['meta']['target_name']
    top_dir = os.path.abspath(master_cfg['io']['top_dir'])
    target_out_root = os.path.abspath(os.path.join(top_dir, master_cfg['io']['output_dir'], "eureka", target)) + "/"
    
    template_dir = os.path.join(top_dir, "configs", "default_eureka")
    run_dir = os.path.join(top_dir, "configs", "eureka_runs", target)
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Executing Eureka for stages {args.stages}...", flush=True)

    for stage in args.stages:
        template_name = get_eureka_template_name(stage, master_cfg['meta']['instrument'], master_cfg['meta']['observing_mode'])
        template_file = os.path.join(template_dir, template_name)
        run_file = os.path.join(run_dir, f"S{stage}_{target}.ecf")
        
        if not os.path.exists(template_file):
            print(f"    [!] Error: Missing {template_file}. Please add default ECFs to configs/default_eureka/")
            sys.exit(1)

        edit_and_save_ecf(master_cfg, template_file, run_file, stage, top_dir, target_out_root)
        if stage == 5:
            generate_epf(master_cfg, run_dir)
        
        import matplotlib
        import eureka.lib.plots
        matplotlib.rcParams['font.family'] = 'sans-serif' 
        eureka.lib.plots.set_rc(style='custom')

        if stage == 1:
            import eureka.S1_detector_processing.s1_process as s1
            s1.rampfitJWST(target, run_dir)
        elif stage == 2:
            import eureka.S2_calibrations.s2_calibrate as s2
            s2.calibrateJWST(target, run_dir)
        elif stage == 3:
            import eureka.S3_data_reduction.s3_reduce as s3
            s3.reduce(target, run_dir)
        elif stage == 4:
            import eureka.S4_generate_lightcurves.s4_genLC as s4
            s4.genlc(target, run_dir)
        elif stage == 5:
            import eureka.S5_lightcurve_fitting.s5_fit as s5
            s5.fitlc(target, run_dir)
        elif stage == 6:
            import eureka.S6_planet_spectra.s6_spectra as s6
            s6.plot_spectra(target, run_dir)
            
        print(f"\n=== STAGE {stage} COMPLETE ===\n", flush=True)