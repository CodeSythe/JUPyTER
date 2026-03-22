# JUPyTER: JWST Unified Pipeline Toolkit for Exoplanet Reduction

JUPyTER is an automated, unified wrapper designed to run raw JWST time-series data through three pipelines simultaneously: **Eureka!**, **exoTEDRF**, and **SPARTA**.

Because this wrapper interfaces with three distinct, heavy-duty astrophysics pipelines, **you must install the backend pipelines and their Conda environments on your machine before running JUPyTER.**

---

## 🛠️ Phase 1: Environment Setup

You will need three separate Conda environments. JUPyTER uses these names to automatically switch environments during the reduction process.

### 1. Eureka!
The official [Eureka! Installation Instructions](https://eurekadocs.readthedocs.io/en/latest/installation.html).

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
* Ensure your conda environment is named exactly: `exotedrf`

### 3. SPARTA
Unlike the other two pipelines, SPARTA is executed via direct script calls. You must clone the SPARTA repository directly into your JUPyTER folder and build its environment.
```bash
# Clone JUPyTER first
git clone [https://github.com/YOUR_USERNAME/JUPyTER.git](https://github.com/YOUR_USERNAME/JUPyTER.git)
cd JUPyTER

# Clone SPARTA directly inside the JUPyTER directory
git clone [https://github.com/ideasrule/sparta.git](https://github.com/ideasrule/sparta.git)

# Create the SPARTA environment
conda create -n sparta python=3.11
conda activate sparta
pip install numpy scipy astropy matplotlib pyyaml emcee
conda deactivate
```