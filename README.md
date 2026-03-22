# JUPyTER: JWST Unified Pipeline Toolkit for Exoplanet Reduction
*Maintained by the NEXOTRANS Team*

JUPyTER is an automated, unified wrapper designed to run raw JWST time-series data through three state-of-the-art pipelines simultaneously: **Eureka!**, **exoTEDRF**, and **SPARTA**.

Because this wrapper interfaces with three distinct, heavy-duty astrophysics pipelines, **you must install the backend pipelines and their Conda environments on your machine before running JUPyTER.**

---

## 🛠️ Phase 1: Environment Setup

You will need three separate Conda environments. JUPyTER uses these names to automatically switch environments during the reduction process.

### 1. Eureka!
Follow the official [Eureka! Installation Instructions](https://eurekadocs.readthedocs.io/en/latest/installation.html).
* Ensure your conda environment is named exactly: `eureka`

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