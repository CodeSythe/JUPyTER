import os
import glob
import yaml
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def load_config(config_path="configs/master_run_config.yaml"):
    """Loads the master YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def find_newest_file(pattern):
    """Finds the most recently modified file matching a wildcard pattern."""
    files = glob.glob(pattern, recursive=True)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_eureka_data(filepath):
    """Loads Eureka! text outputs and converts fractional depth to percent."""
    df = pd.read_csv(filepath, delimiter=' ', comment='#')
    
    wave = df['wavelength'].values
    wave_err = (df['bin_width']).values
    
    # Eureka outputs absolute fractional depth. Multiply by 100 for percent (%)
    depth_pct = df['rp^2_value'].values * 100.0
    
    # Average the asymmetric errors and convert to percent
    err_pct = ((df['rp^2_errorneg'] + df['rp^2_errorpos']) / 2.0).values * 100.0
    
    return wave, wave_err, depth_pct, err_pct

def load_exotedrf_data(filepath):
    """Loads exoTEDRF CSV outputs and converts ppm to percent."""
    df = pd.read_csv(filepath, comment='#')
    
    wave = df['wave'].values
    wave_err = df['wave_err'].values
    
    # exoTEDRF outputs in ppm. Divide by 10,000 to get percent (%)
    depth_pct = df['dppm'].values / 10000.0
    err_pct = df['dppm_err'].values / 10000.0
    
    return wave, wave_err, depth_pct, err_pct

def main():
    parser = argparse.ArgumentParser(description="Compare Eureka! and exoTEDRF Spectra")
    parser.add_argument("--residuals", action="store_true", help="Plot a residual panel (Eureka - exoTEDRF)")
    args = parser.parse_args()

    config = load_config()
    target = config['meta']['target_name']
    out_dir = config['io']['output_dir']

    # Dynamic path matching to grab the newest runs
    eureka_pattern = os.path.join(out_dir, "eureka", target, "Stage6", f"S6_*_{target}_run*", "*", f"S6_{target}_*_rp^2_Table_Save.txt")
    exotedrf_pattern = os.path.join(out_dir, "exotedrf", target, "pipeline_outputs_directory_*", "Stage4", f"{target}_*_transmission_spectrum*.csv")

    eureka_file = find_newest_file(eureka_pattern)
    exotedrf_file = find_newest_file(exotedrf_pattern)

    if not eureka_file or not exotedrf_file:
        print("[!] Missing output files. Ensure both Eureka! Stage 6 and exoTEDRF Stage 4 are complete.")
        return

    print(f"[+] Loading Eureka! file: {os.path.basename(eureka_file)}")
    print(f"[+] Loading exoTEDRF file: {os.path.basename(exotedrf_file)}")

    # Load and process data
    try:
        eu_wave, eu_wave_err, eu_depth_pct, eu_err_pct = load_eureka_data(eureka_file)
    except Exception as e:
        print(f"[!] Error reading Eureka file: {e}")
        return

    try:
        exo_wave, exo_wave_err, exo_depth_pct, exo_err_pct = load_exotedrf_data(exotedrf_file)
    except Exception as e:
        print(f"[!] Error reading exoTEDRF file: {e}")
        return

    # ==========================================
    # PLOTTING
    # ==========================================
    plt.style.use('default')
    
    # Adjust figure layout based on whether residuals are requested
    if args.residuals:
        fig, (ax_main, ax_res) = plt.subplots(2, 1, figsize=(12, 8), dpi=150, 
                                              gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
    else:
        fig, ax_main = plt.subplots(figsize=(12, 6), dpi=150)
        ax_res = None

    color_eu = '#0072B2'   # Blue
    color_exo = '#D55E00'  # red-orange

    # --- Main Plot ---
    ax_main.errorbar(eu_wave, eu_depth_pct, xerr=eu_wave_err, yerr=eu_err_pct, 
                     fmt='o', markersize=7, capsize=0, elinewidth=1.5,
                     label='Eureka! (v1.3)', alpha=0.85, color=color_eu,
                     markeredgecolor='white', markeredgewidth=0.5)
    
    ax_main.errorbar(exo_wave, exo_depth_pct, xerr=exo_wave_err, yerr=exo_err_pct, 
                     fmt='s', markersize=7, capsize=0, elinewidth=1.5,
                     label='exoTEDRF', alpha=0.85, color=color_exo,
                     markeredgecolor='white', markeredgewidth=0.5)

    ax_main.set_ylabel("Transit Depth (%)", fontsize=14, fontweight='bold', labelpad=10)
    ax_main.set_title(f"Transmission Spectrum Pipeline Comparison: {target}", fontsize=16, fontweight='bold', pad=15)
    ax_main.legend(fontsize=12, loc='upper right', framealpha=0.9, edgecolor='grey', borderpad=0.8)
    
    # --- Residuals Plot ---
    if ax_res is not None:
        # Check if wavelength grids match well enough to subtract
        if len(eu_wave) == len(exo_wave):
            # Calculate residuals (Eureka - exoTEDRF)
            residuals = eu_depth_pct - exo_depth_pct
            
            # Error propagation for subtraction: sqrt(err1^2 + err2^2)
            res_err = np.sqrt(eu_err_pct**2 + exo_err_pct**2)
            
            ax_res.errorbar(eu_wave, residuals, xerr=eu_wave_err, yerr=res_err,
                            fmt='D', markersize=5, capsize=0, elinewidth=1.2,
                            color='black', alpha=0.7, markeredgecolor='white', markeredgewidth=0.5)
            
            ax_res.axhline(0, color='grey', linestyle='--', alpha=0.7)
            ax_res.set_ylabel("Δ Depth (%)", fontsize=12, fontweight='bold')
        else:
            print("[!] Warning: Wavelength grids do not match. Cannot compute direct residuals.")
            ax_res.text(0.5, 0.5, 'Wavelength Grids Mismatch', ha='center', va='center', transform=ax_res.transAxes)

    # --- Formatting for the bottom-most axis ---
    bottom_ax = ax_res if ax_res is not None else ax_main
    bottom_ax.set_xlabel("Wavelength (µm)", fontsize=14, fontweight='bold', labelpad=10)

    # Clean grids for all axes
    for ax in [ax_main, ax_res] if ax_res is not None else [ax_main]:
        ax.tick_params(axis='both', which='major', labelsize=12, length=6, width=1.5)
        ax.tick_params(axis='both', which='minor', length=3, width=1)
        ax.grid(True, which='major', linestyle='-', alpha=0.3, color='grey')
        ax.grid(True, which='minor', linestyle=':', alpha=0.2, color='grey')
        ax.minorticks_on()
    
    plt.tight_layout()

    # Save the plot
    res_suffix = "_with_residuals" if args.residuals else ""
    plot_path = os.path.join(out_dir, f"{target}_pipeline_comparison{res_suffix}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {plot_path}")
    
    plt.show()

if __name__ == "__main__":
    main()