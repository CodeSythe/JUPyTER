import os
import glob
import yaml
import argparse
import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, Whisker
from bokeh.transform import factor_cmap

def convert_to_unit(absolute_value, target_unit):
    """Converts absolute transit depth (Rp/R*)^2 to the requested unit."""
    if target_unit == "percent":
        return absolute_value * 100.0
    elif target_unit == "ppm":
        return absolute_value * 1e6
    return absolute_value

def main():
    parser = argparse.ArgumentParser(description="Compile and Plot Interactive Transmission Spectra")
    parser.add_argument("--config", type=str, default="configs/master_run_config.yaml", help="Path to master config")
    parser.add_argument("--unit", type=str, choices=["percent", "ppm", "absolute"], default="percent", help="Units for transit depth")
    args = parser.parse_args()

    # 1. Read Master Config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    target = cfg['meta']['target_name']
    out_dir = cfg['io']['output_dir']
    comp_dir = os.path.join(out_dir, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    all_data = []

    # 2. Extract Eureka! Data
    eureka_pattern = os.path.join(out_dir, "eureka", target, "Stage6", f"S6*_{target}_run1", "S6*Table_Save.txt")
    eureka_files = glob.glob(eureka_pattern)
    if eureka_files:
        try:
            # Eureka format: wave, bin_width, rp2, err_neg, err_pos (absolute units)
            data = np.loadtxt(eureka_files[0], comments="#")
            if data.ndim > 1:
                for row in data:
                    all_data.append({
                        "Pipeline": "Eureka!",
                        "Wave": row[0],
                        "Wave_Err": row[1] / 2.0,
                        "Depth": convert_to_unit(row[2], args.unit),
                        "Err_Neg": convert_to_unit(row[3], args.unit),
                        "Err_Pos": convert_to_unit(row[4], args.unit)
                    })
            print(f"[+] Loaded Eureka! spectrum")
        except Exception as e:
            print(f"[!] Error reading Eureka file: {e}")

    # 3. Extract exoTEDRF Data
    exotedrf_pattern = os.path.join(out_dir, "exotedrf", target, "stage4", "*transmission_spectrum*prebin.csv")
    exotedrf_files = glob.glob(exotedrf_pattern)
    if exotedrf_files:
        try:
            # exoTEDRF format: wave, wave_err, dppm, dppm_err (ppm units)
            data = pd.read_csv(exotedrf_files[0], comment='#')
            for _, row in data.iterrows():
                abs_depth = row['dppm'] / 1e6
                abs_err = row['dppm_err'] / 1e6
                all_data.append({
                    "Pipeline": "exoTEDRF",
                    "Wave": row['wave'],
                    "Wave_Err": row['wave_err'],
                    "Depth": convert_to_unit(abs_depth, args.unit),
                    "Err_Neg": convert_to_unit(abs_err, args.unit),
                    "Err_Pos": convert_to_unit(abs_err, args.unit)
                })
            print(f"[+] Loaded exoTEDRF spectrum")
        except Exception as e:
            print(f"[!] Error reading exoTEDRF file: {e}")

    # 4. Extract SPARTA Data
    sparta_bins = sorted(glob.glob(os.path.join(out_dir, "sparta", target, "mcmc_chain_bin_*", "white_light_result.txt")))
    if sparta_bins:
        try:
            for f in sparta_bins:
                data = np.loadtxt(f, comments="#")
                if data.size > 0 and data.ndim == 1:
                    # SPARTA format: min_wave, max_wave, rp2, err_neg, err_pos (absolute units)
                    w_min, w_max, d_med, d_neg, d_pos = data[0], data[1], data[2], data[3], data[4]
                    all_data.append({
                        "Pipeline": "SPARTA",
                        "Wave": (w_min + w_max) / 2.0,
                        "Wave_Err": (w_max - w_min) / 2.0,
                        "Depth": convert_to_unit(d_med, args.unit),
                        "Err_Neg": convert_to_unit(d_neg, args.unit),
                        "Err_Pos": convert_to_unit(d_pos, args.unit)
                    })
            print(f"[+] Loaded SPARTA spectrum")
        except Exception as e:
            print(f"[!] Error reading SPARTA files: {e}")

    if not all_data:
        print("[!] No transmission spectra found for any pipeline.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Calculate upper and lower bounds for the error bars (Whiskers)
    df['Upper'] = df['Depth'] + df['Err_Pos']
    df['Lower'] = df['Depth'] - df['Err_Neg']

    # Apply slight x-axis offsets to prevent error bars from completely eclipsing each other
    offset_map = {'Eureka!': -0.003, 'exoTEDRF': 0.0, 'SPARTA': 0.003}
    df['Wave_Plot'] = df.apply(lambda row: row['Wave'] + offset_map.get(row['Pipeline'], 0), axis=1)

    # 5. Save Unified Data to CSV
    csv_path = os.path.join(comp_dir, f"{target}_unified_spectrum_{args.unit}.csv")
    df.drop(columns=['Upper', 'Lower', 'Wave_Plot']).to_csv(csv_path, index=False)
    print(f"[+] Saved Unified Dataset to: {csv_path}")

    # 6. Interactive Plotting with Bokeh
    unit_labels = {"percent": "Transit Depth (%)", "ppm": "Transit Depth (ppm)", "absolute": "Transit Depth (Rp/R*)²"}
    
    p = figure(
        title=f"Interactive Transmission Spectrum Comparison: {target}",
        x_axis_label="Wavelength (µm)",
        y_axis_label=unit_labels[args.unit],
        width=1000, height=600,
        tools="pan,wheel_zoom,box_zoom,reset,save",
        active_scroll="wheel_zoom"
    )

    colors = {'Eureka!': '#1f77b4', 'exoTEDRF': '#2ca02c', 'SPARTA': '#d62728'}
    markers = {'Eureka!': 'circle', 'exoTEDRF': 'square', 'SPARTA': 'triangle'}

    for pipeline in df['Pipeline'].unique():
        sub_df = df[df['Pipeline'] == pipeline]
        source = ColumnDataSource(sub_df)
        
        # Add Error Bars (Whiskers)
        whisker = Whisker(source=source, base="Wave_Plot", upper="Upper", lower="Lower", line_color=colors[pipeline], line_width=1.5, line_alpha=0.7)
        whisker.upper_head.line_color = colors[pipeline]
        whisker.lower_head.line_color = colors[pipeline]
        p.add_layout(whisker)

        # Add Scatter Points
        scatter = p.scatter(
            x='Wave_Plot', y='Depth', source=source,
            size=9, color=colors[pipeline], marker=markers[pipeline],
            legend_label=pipeline, alpha=0.9, line_color="black"
        )
        
        # Add Hover Tool for exactly this pipeline
        hover = HoverTool(renderers=[scatter], tooltips=[
            ("Pipeline", "@Pipeline"),
            ("Wavelength", "@Wave{0.000} µm"),
            ("Depth", f"@Depth{{0.000}} (+@Err_Pos{{0.000}} / -@Err_Neg{{0.000}})"),
        ])
        p.add_tools(hover)

    # Styling
    p.legend.click_policy = "hide" # Click the legend to hide/show pipelines!
    p.legend.location = "top_right"
    p.xgrid.grid_line_alpha = 0.3
    p.ygrid.grid_line_alpha = 0.3

    # Output to HTML
    html_path = os.path.join(comp_dir, f"{target}_transmission_comparison_{args.unit}.html")
    output_file(html_path, title=f"{target} Spectra Comparison")
    save(p)
    print(f"[+] Saved Interactive Bokeh Plot to: {html_path}")

if __name__ == "__main__":
    main()