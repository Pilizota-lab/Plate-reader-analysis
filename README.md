# Introduction

This repository contains a set of graphical user interfaces (GUIs) designed to help with the post-processing, visualisation, and analysis of bacterial growth and fluorescence data collected through plate reader experiments. The tools are intended to be user-friendly, and no prior programming experience is required.

We provide the following tools:
1. _Data visualisation GUI_ - visualise and inspect raw data exported from plate reader software
2. _Growth rate estimator_ - compute growth rates of exponentially growing cells
3. _Fluorescence calculator_ - computes fluorescence normalised by optical density and allows user to select region of interest
4. _Data plotting GUI (experimental)_ - generate custom plots for growth data and collate multiple growth curves across different datasets

Users with no Python experience can follow the beginner-friendly setup guide at the end of this page. 

# Requirements 
## File format

Input data files should be uploaded in the .csv format. The data table should start from row 7 and should contain:
- well names in the first column
- sample details or conditions in the second column (this will allow for grouping of replicates)
- time series measurements in the remaining columns
This format is common with plate reader software such as MARS (BMG Labtech) and Tecan. Please refer to sample datasets provided in this repository. If your exported files have a slightly different structure, the GUIs can be adapted by modifying the load_data() functions. 

## Packages
Basic requirements: Python 3, Numpy, Seaborn, Matplotlib, Pandas, Scipy

Tested on: Python 3.11.9 on 64-bit Windows.

# Data visualisation GUI
This GUI allows the user to load the data and plot growth curves in selected wells, or selected groups of replicates. Useful for identifying any strange curves before further analysis. 

To run:
```bash
python platereader_data_visualisation_GUI.py
```

# Growth rate estimator
Iterates through every growth curve found in the dataset and computes the growth rate with a customisable running window. It allows the user to inspect how growth rate changes over time, adjust the fitting parameters to their particular dataset, and select the region of exponential, steady-state growth. This GUI helps distinguish true exponential phase from noise and artefacts. 

To run:
```bash
python growth_rate_estimator.py
```

Outputs:
- a results file containing, for every well: the global growth rate and its uncertainty (standard error of the slope), the time-point bounds of the chosen exponential interval, the filter and smoothing parameters used, and (for the exponential interval only), the timepoints and ODs, plus the "instantaneous" growth rates (computed with a running window), their time points, and their 95% confidence intervals
- a folder containing the lnOD vs time and growth rate vs time plots

# Fluorescence calculator
Allows users to compute OD-normalised fluorescence from paired fluorescence and OD reads. Users can load fluorescence and OD datasets, inspect the data, and select the region of interest based on which mean fluorescence will be calculated. 

To run:
```bash
python fluorescence_calculator.py
```
# Data plotting GUI
Similar to the Data visualisation GUI, it allows the user to explore plate reader data, but also customise plots before saving to disk. The user can also upload multiple data files, but this feature is limited to "group analysis" (i.e., individual growth curves cannot be selected, the entire group will be plotted instead). Please note that this GUI is still under development and has not yet been fully tested. Some features may not behave as expected.

To run:
```bash
python platereader_plotting_GUI_experimental.py
```
# Quick start guide for complete beginners to Python

All the GUIs above are written in a user-friendly way and require no previous coding experience. However, you will still need to install Python, as well as the required packages on your computer. Below is a step-by-step guide on how to do so.

1. Install Anaconda <br>
Download and install Anaconda from the official Anaconda website. Anaconda is an official package manager which allows you to install Python and any required dependencies safely, and ensures that packages do not clash across different software running Python on your computer.

2. Create a conda environment <br>

In Anaconda Prompt, create a new environment by running:

```bash
conda create -n platereader_env
```
When prompted, type "y" and press Enter. This will create a stand-alone location on your computer with everything you need to run any of the GUIs above. The GUIs will be run accessing this location. Even if you have other applications that use python installed on your computer, this will not affect that. Then, activate the environment with:
```bash
conda activate platereader_env
```
3. Install the required packages
With the environment activated, install the required packages by running:
```bash
pip install numpy pandas scipy matplotlib seaborn
```
4. Clone this repository
Download this repository from GitHub by clicking:
**Code > Download ZIP**
Then unzip the folder on your computer.
In Anaconda Prompt, move into the folder containing the downloaded GUIs.
For example: cd path/to/your/downloaded/folder
5. Run the GUIs
Once you have activated your environment and navigated to your repository folder through Anaconda prompt, simply type in the commands outlined above, for each GUI you wish to use. 
