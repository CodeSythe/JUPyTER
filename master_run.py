# ==============================================================================
# JUPyTER: JWST Unified Pipeline Toolkit for Exoplanet Reduction
# Master Execution Script
# 
# Description: This is the main engine of JUPyTER. It reads the master config, 
#              determines which stages of which pipelines need to be run, manages
#              conda environments, and dispatches the work safely.
# ==============================================================================
# master_run.py

import os
import sys
import yaml
import argparse
import subprocess
import datetime
from pathlib import Path

def load_config(config_path):
    """
    Loads the master YAML configuration file.
    This file acts as the single source of truth for all reduction parameters.
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def check_smart_start(pipeline, target, output_dir, requested_start, requested_end):
    """
    The 'Smart Resume' Engine.
    
    Instead of blindly starting from Stage 1 every time, this function checks the 
    output directories to see what work has already been completed. If it finds 
    successful outputs from previous stages, it automatically advances the starting 
    stage to save compute time (unless the user explicitly uses --force-redo).
    """
    pipeline_out = Path(output_dir) / pipeline / target
    actual_start = requested_start

    # ==========================================
    # EUREKA! RESUME LOGIC
    # ==========================================
    if pipeline == "eureka":
        # Eureka cleanly separates its outputs into Stage-specific folders.
        stage_folders = {1: "Stage1", 2: "Stage2", 3: "Stage3", 4: "Stage4", 5: "Stage5", 6: "Stage6"}
        for stage in range(requested_start, requested_end + 1):
            folder = pipeline_out / stage_folders.get(stage, f"Stage{stage}")
            # If the folder exists and is not empty, assume the stage is complete.
            if folder.exists() and any(folder.iterdir()):
                actual_start = stage + 1
            else:
                break
                
    # ==========================================
    # EXOTEDRF RESUME LOGIC
    # ==========================================
    elif pipeline == "exotedrf":
        # exoTEDRF creates timestamped root directories. We must find the newest one.
        exo_roots = list(pipeline_out.glob("pipeline_outputs_directory*"))
        if exo_roots:
            exo_root = max(exo_roots, key=os.path.getmtime)
            # exoTEDRF runs Stages 1-3 as a single block. 
            # We check Stage 3 to see if the whole block finished.
            if requested_start <= 3:
                stage3_dir = exo_root / "Stage3"
                s3_complete = stage3_dir.exists() and (list(stage3_dir.glob("**/*_spectra_fullres.fits")) or list(stage3_dir.glob("**/*_x1d.fits")))
                actual_start = 4 if s3_complete else 1
            else:
                actual_start = requested_start
        else:
            actual_start = 1
            
    # ==========================================
    # SPARTA RESUME LOGIC
    # ==========================================
    elif pipeline == "sparta":
        # SPARTA does not use subfolders. It dumps everything directly into the target 
        # directory. We must check for specific file signatures to verify completion.
        for stage in range(requested_start, requested_end + 1):
            if stage == 1 and list(pipeline_out.glob("*rateints.fits")):
                actual_start = 2
            elif stage == 2 and list(pipeline_out.glob("cleaned_*rateints.fits")):
                actual_start = 3
            elif stage == 3 and list(pipeline_out.glob("optx1d_*.fits")):
                actual_start = 4
            elif stage == 4 and list(pipeline_out.glob("chain*.h5")): 
                actual_start = 5
            else:
                break

    return None if actual_start > requested_end else actual_start

def dispatch_worker(pipeline_name, config_path, env_name, stages, config_dict, log_dir):
    """
    The Worker Dispatcher.
    
    This function constructs the command-line arguments to trigger the specific 
    pipeline wrapper (e.g., workers/call_eureka.py). It handles the environment 
    variables required by STScI (like the CRDS cache) and manages logging.
    """
    worker_script = os.path.join("workers", f"call_{pipeline_name}.py")
    crds_path = os.path.abspath(config_dict['io']['crds_cache'])
    live_output = config_dict.get('common_parameters', {}).get('live_terminal_output', True)
    
    # Create a unique timestamped log file for this specific run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{pipeline_name}_S{stages[0]}-{stages[-1]}_{timestamp}.log")
    
    # Set up the strict environment variables required by the JWST pipeline
    env = os.environ.copy()
    env["CRDS_PATH"] = crds_path
    env["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"
    env["CRDS_MODE"] = "auto"
    env["PYTHONUNBUFFERED"] = "1" 

    print(f"\n[Logging {pipeline_name.upper()} output to: {log_file}]")

    # Group Stages 1-3 for exoTEDRF because its architecture requires them to run sequentially in one command
    if pipeline_name == 'exotedrf':
        run_groups = []
        red_stages = [s for s in stages if s in [1, 2, 3]]
        if red_stages: run_groups.append([1, 2, 3])
        if 4 in stages: run_groups.append([4])
    else:
        run_groups = [[s] for s in stages]

    for group in run_groups:
        group_str = ", ".join(map(str, group))
        print("\n" + "="*60)
        print(f" [{pipeline_name.upper()}] EXECUTING STAGE [{group_str}]")
        print("="*60, flush=True)

        if not live_output:
            print(f"    [Silent Mode] Processing... tracking progress in {os.path.basename(log_file)}", flush=True)

        # Construct the command to activate the correct Conda environment before running the worker
        cmd = f"conda run --no-capture-output -n {env_name} python {worker_script} --config {config_path} --stages {' '.join(map(str, group))}"
        
        with open(log_file, "a") as f:
            f.write(f"\n{'='*60}\n--- STARTING STAGE [{group_str}] ---\n{'='*60}\n")

            # Launch the worker process
            process = subprocess.Popen(
                cmd, shell=True, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, universal_newlines=True
            )
            
            # Stream the output live to the terminal (if enabled) and to the log file
            for line in process.stdout:
                if live_output:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                f.write(line)
                f.flush()
                
            process.wait()

            # Catch crashes immediately and stop the pipeline to prevent cascading errors
            if process.returncode != 0:
                error_msg = f"\n[!] Error: {pipeline_name.upper()} crashed during Stage(s) [{group_str}]."
                print(error_msg)
                f.write(error_msg + "\n")
                return False 
                
        print(f"\n      [{pipeline_name.upper()}] STAGE [{group_str}] COMPLETE", flush=True)
    return True

def main():
    """
    The Entry Point.
    Parses arguments, loads the configuration, and kicks off the reduction plan.
    """
    parser = argparse.ArgumentParser(description="JUPyTER: JWST Unified Pipeline Toolkit for Exoplanet Reduction")
    parser.add_argument("--config", type=str, default="configs/master_run_config.yaml", help="Path to your master configuration file.")
    parser.add_argument("--force-redo", action="store_true", help="Bypass the smart resume feature and force pipelines to overwrite existing data.")
    args = parser.parse_args()

    config = load_config(args.config)
    target = config['meta']['target_name']
    out_dir = config['io']['output_dir']
    
    # Ensure the central logging directory exists
    log_dir = os.path.join(config['io']['top_dir'], "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Ensure the base output directories exist for all active pipelines
    for pipe in ['eureka', 'exotedrf', 'sparta']:
        (Path(out_dir) / pipe / target).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print(f"  JUPyTER: JWST Unified Pipeline Wrapper")
    print(f"  Target: {target} | Mode: {config['meta']['observing_mode']}")
    print("="*60)
    
    run_plan = []
    total_stages = 0

    # Build the execution plan by checking the config and the smart-resume status
    for pipe_name, pipe_cfg in config['pipelines'].items():
        if pipe_cfg.get('run', False):
            start = pipe_cfg.get('start_stage', 1)
            end = pipe_cfg.get('end_stage', 6)
            
            if not args.force_redo:
                start = check_smart_start(pipe_name, target, out_dir, start, end)
                
            if start is not None and start <= end:
                stages_to_run = list(range(start, end + 1))
                total_stages += len(stages_to_run)
                run_plan.append((pipe_name, pipe_cfg['env_name'], stages_to_run))

    if total_stages == 0:
        print("[✓] All requested stages are already complete based on output directories. Use --force-redo to overwrite.")
        return

    print(f"\n[+] Processing Pipelines...\n")
    results = {}
    
    # Execute the plan
    for pipe_name, env_name, stages in run_plan:
        success = dispatch_worker(pipe_name, args.config, env_name, stages, config, log_dir)
        results[pipe_name] = "SUCCESS" if success else "FAILED"
            
    # Print the final report
    print("\n" + "="*60)
    print("  JUPyTER REDUCTION SUMMARY")
    print("="*60)
    for pipe, status in results.items():
        print(f"  {pipe.upper().ljust(10)} : {status}")
    print(f"\n[✓] Outputs saved to {out_dir}")
    print(f"[✓] Full terminal logs saved to {log_dir}")

if __name__ == "__main__":
    main()