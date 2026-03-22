# Introduction
These user interfaces are designed to aid with post-processing of plate reader data for bacterial growth. We provide the following easy-to-use UIs:
1. Data visualisation GUI - allows the user to quickly visualise and inspect raw data exported from plate reader softwares;
2. Data plotting GUI (experimental) - custom-made plots of growth data; also allows for collation of multiple curves across different dataset files;
3. Growth rate estimator - allows the user to compute growth rates of exponentially growing cells;
4. Fluorescence analysis GUI - allows the user to normalise fluorescence data w.r.t. the OD, and, optionally, to subtract autofluorescence (provided that a control is included in the experiment).

If you have very little coding experience or have never coded in Python before, follow the step by step instructions at the end of this page to set up everything needed for these GUIs.

# Requirements 
## A. File format
Data files need to be uploaded in the .csv format, and data table should start from row 7. Data table should contain the well names + sample details in the first 2 columns, and time series in the subsequent columns (see example below). We found that this is the format of most commonly exported data from plate reader softwares such as MARS (BMG Labtech) and Tecan; however, the GUIs can easily be edited to accommodate for slightly different file structure and type, or use the settings of the platereader software to change the exported file structure.


## B. Packages
Basic requirements: Python 3, Numpy, Seaborn, Matplotlib, Pandas, Scipy

Tested on: Python 3.11.9 on 64-bit Windows.


# Data visualisation GUI


# Data plotting GUI
Similar to the above, it allows the user to explore plate reader data, and customise plots before saving to disk. The user can also upload multiple data files, but this feature is limited to "group analysis" (i.e., individual growth curves cannot be selected, the entire group will be plotted instead). <br>
Please note that this GUI is still under development and has not yet been fully tested. Some features may not behave as expected. For known issues of this GUI, see the Issues page.

# Growth rate estimator
- allows user to inspect behaviour of growth rate over time and interact with fitting parameters such as window sizes
- especially good for fast growing cells, where rolling window size has to be relatively small (due to only having a few kinetic points that are above the instrument detection limit, and before coming out of exponential phase).
- also good at allowing users to select what is true exp phase rather than artefact/noise

# Fluorescence analysis GUI

in progress...

# Quick start guide for complete beginners to Python

All the GUIs above are written in a user-friendly way and require no previous coding experience. Despite this, you will still need to install Python, as well as the required packages on your computer. Below is a step-by-step guide on how to do so.

1. Install Anaconda <br>
Anaconda is an official package manager which allows you to install and use everything you need quickly, and ensure packaages don't clash across different softwares running python on your computer. Follow the download link below from the official developer [link]
2. Create a conda environment
This will create a stand-alone location on your computer with everything you need to run any of the GUIs above. The GUIs will be run accessing this location. Even if you have other applications that use python installed on your computer, this will not affect that.
To create the environment, follow the steps below:

3. Download the GUIs from this page

4. Run these codes
- open Anaconda Prompt (just installed)
- activate the environment writing the following line (this way you will be accessing all packages just installed in your new environment)
- run the GUI you're interested in







*Tell us you're using our GUIs*
*Leave feedback*
*To cite:*
