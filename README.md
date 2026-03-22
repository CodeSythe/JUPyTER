# JUPyTER: JWST Unified Pipeline Toolkit for Exoplanet Reduction

JUPyTER is an automated, unified wrapper designed to run raw JWST time-series data through three pipelines simultaneously: **Eureka!**, **exoTEDRF**, and **SPARTA**.

Because this wrapper interfaces with three distinct, heavy-duty astrophysics pipelines, **you must install the backend pipelines and their Conda environments on your machine before running JUPyTER.**

---

## 🛠️ Phase 1: Environment Setup

You will need three separate Conda environments. JUPyTER uses these names to automatically switch environments during the reduction process.

### 1. Eureka!
The official [Eureka! Installation Instructions](https://eurekadocs.readthedocs.io/en/v1.3/installation.html).

Use Eureka! v1.3

```bash
# Create the Eureka! environment
conda create -n eureka python==3.13.0
conda activate eureka

# Clone repo
git clone -b v1.3 https://github.com/kevin218/Eureka.git
cd Eureka

# Install from source
pip install -e '.[jwst]'
```


### 2. exoTEDRF
Follow the official [exoTEDRF Installation Instructions](https://exotedrf.readthedocs.io/).

Use ExoTEDRF v2.4.1

```bash
# Create the ! environment
conda create -n exotedrf python<3.14
conda activate exotedrf

# Clone repo
git clone https://github.com/radicamc/exoTEDRF.git
cd exoTEDRF

# Install 
python setup.py install
```

Also intall some dependencies (if not installed with the above commands)

```bash
pip install exouprf==1.1.1 exotic_ld h5py
pip install webbpsf
```

### 3. SPARTA
Unlike the other two pipelines, SPARTA is executed via direct script calls. You must clone the SPARTA repository directly into your JUPyTER folder and build its environment.
```bash
# Clone JUPyTER first
git clone [https://github.com/CodeSythe/JUPyTER.git](https://github.com/CodeSythe/JUPyTER.git)
cd JUPyTER

# Clone SPARTA directly inside the JUPyTER directory
git clone [https://github.com/ideasrule/sparta.git](https://github.com/ideasrule/sparta.git)

# Create the SPARTA environment
conda create -n sparta python=3.11
conda activate sparta
pip install numpy scipy astropy matplotlib pyyaml emcee
conda deactivate
```

## 🚀 Phase 2: How to Use JUPyTER

### 1. Add your Raw Data
Download your uncalibrated JWST .fits files from MAST and place them into the inputs directory.

### 2. Set up your CRDS Cache
JWST requires calibration reference files. Create a folder on your machine to act as the cache 
```bash
mkdir crds_cache
```

### 3. Configure the Run
Open configs/master_run_config.yaml 
- Update the top_dir and crds_cache paths to match your machine's absolute paths.
- Set your planetary priors, target name, and instrument parameters.
- Toggle which pipelines you want to run (run: True or False).

### 4. Execute
```bash
python master_run.py
```

## 💡 Tips & Best Practices
- **Pipeline Roles**: I highly recommend using Eureka! exclusively to perform your broad White Light Curve (WLC) fits to derive and lock in your precise orbital parameters. Use Eureka!/exoTEDRF for your spectroscopic reduction and final transmission spectrum.
- **SPARTA Status**: ⚠️ Put SPARTA on hold for now. The current SPARTA source code contains several bugs related to MIRI rotation and optimal extraction. Keep run: False for SPARTA in the master config until patched.
- **Advanced Configuration**: If you need to deeply customize pipeline parameters beyond what is available in the master config, you can edit the underlying pipeline template files directly inside the configs/ folder.
- Documentation & Parameters: * For questions about detector-level parameters (like jump detection or ramp fitting), consult the [Official JWST Pipeline ReadTheDocs](https://jwst-pipeline.readthedocs.io/en/latest/index.html).
    - More information about the instruments consult [JWST User Documentation](https://jwst-docs.stsci.edu/)
    - For step specific logic, consult [Eureka! Docs](https://eurekadocs.readthedocs.io/en/v1.3/stages.html) and [exoTEDRF Docs](https://exotedrf.readthedocs.io/en/latest/content/usage.html)