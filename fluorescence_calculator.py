#%%

# Contact: Diana Coroiu
# Affiliation (at the time of writing): University of Edinburgh
# Contact:
#   - personal: didicoroiu@yahoo.com 
#   - institutional: d.coroiu@sms.ed.ac.uk
# Last updated: 31/03/2026

# Description:

# This GUI lets the user interact with fluorescence data and select fluorescence during exponential phase.
# The fluorescence for each sample will be computed as the average of the normalised fluorescence points selected as exponential


# The raw data has to be in the .csv format and follow a given structure:
# (1): table headings in row 8, data start in row 9
# (2): time to be given in minutes (otherwise time scale will not make sense)
# (3): axis 0 (rows) to contain different wells, axis 1 (columns) to contain different time points
# (4): wells labelled as "A01" and not "A1"


# Intended for academic use; feel free to adapt for other plate-reader data file formats.

# Full instructions and details on GitHub: https://github.com/Pilizota-lab/Plate-reader-analysis/

# to be run from the terminal

# start code
# import modules
from tkinter import *
from tkinter import ttk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
from tkinter import filedialog as fd
from tkinter import messagebox
import math
import os
import sys
from scipy.signal import medfilt
from fluorescence_calculator_functions import * # custom functions - ensure functions file is in current working folder

# default parameters -> not modified by user so that one can always reset to defaults
default_params = {
    # time point boundaries if experiment ran for too long (in hours)
    "time_lower": 0,
    "time_upper": 35,
    # windows and filters - DEFAULTS AS BELOW, user choices saved to results for each curve
    "outlier_sigmas": 0.5, # number of stdevs outside of which the point is excluded
    "outlier_window": 9,  
    "smoothing_window": 1,  # smoothing with a hanning window (if not needed, set window to 1)
}
# live parameters -> updated every time user presses "apply to current well" (and will be carried over to next well unless reset by user)
live_params = default_params.copy() # initialise to defaults. Can be updated by user during run

# This will point to "recompute current well" function
recompute_hook = None

### START PROGRAM ###

# FLUORESCENCE DATA
messagebox.showinfo("Please choose a .csv file", "This file will be used to extract FLUORESCENCE data")
df_fluorescence, path = load_data()
# there's photobleaching in the media, so it should not be blanked in the same way as OD (i.e., by taking the first few points). Instead, subtract the media only wells.
# messagebox.showinfo("Please choose a .csv file", "Choose the fail containing the FLUORESCENCE BLANKS for your media")
df_fluorescence = blank_fluorescence_2(df_fluorescence)

# blank fluorescence by subtracting the avg of the 5 smallest fluorescence value for current well (to account for media photobleaching at the beginning of expt)
# df_fluorescence['OD'] = df_fluorescence.groupby('well')['OD'].transform(
#     lambda x: x - np.sort(x.values.astype(float))[:5].mean()
# )
# ABSORBANCE DATA
messagebox.showinfo("Please choose a .csv file", "This file will be used to extract OPTICAL DENSITY data")
df_od, _ = load_data()
df_od = interactive_blanking_od(df_od)

# create folder to save plots and analysed results
try:
    path_to_save = os.path.normpath(path[:-4] + r"_analysed")
    os.mkdir(path_to_save)
except:
    path_to_save = os.path.normpath(path[:-4] + r"_analysed")
    print("Warning: Path to results has already been created, old results might be overwritten.")



# containers for results
selections_to_save = {} # contains final data, as confirmed by user
selection_buffer = []     # Temporary click buffer (not well-specific)
exit_requested = {"value": False}

# MAIN SEQUENCE
parameter_window = create_parameter_window(live_params, default_params)

# try:
unique_wells = df_od['well'].unique()
print(f"{len(unique_wells)} wells found. Starting processing:")

for well in unique_wells:
    print(f"\n--- Processing well {well} ---")
    process_well_interactive(well, df_od, df_fluorescence, live_params, selections_to_save, selection_buffer, exit_requested, path_to_save)
    if exit_requested['value']:
        break

# Convert results to DataFrame
results_df = pd.DataFrame(selections_to_save.values())
    
# except NameError as e:
#     print(f"Error: Dataframes not defined or missing columns. Check raw file structure. Details: {e}")
# except SystemExit as e:
#     print(e)
#     # Save partial results before exiting
#     if selections_to_save:
#         results_df = pd.DataFrame(selections_to_save.values())
# except Exception as e:
#     print(f"An error occurred: {e}")


# Clean up
plt.close('all')

# END

# get results summary:
print(f"\n Results summary: \n")

summary_per_conc = (
    results_df.groupby("sample")
    .agg(
        fluorescence_avg = ("exponential_fluorescence", "mean"),
        fluorescence_std = ("exponential_fluorescence", "std"),
        n_replicates=       ("exponential_fluorescence",     "count"),  # count of successfully computed replicates
    )
    .reset_index()
)

print(summary_per_conc)


# save results to CSV (same path/file name + growth_rates at the end) - if user wants to do so
parameter_window.destroy()
saving_popup = Tk()
saving_popup.title('Do you want to write results to disk?')
def save_results():
    results_df.to_csv((path_to_save + r'\analysed_fluorescence_values.csv'), index=False)
    saving_popup.destroy()
    print('Data saved to disk in results folder.')
    return()

def move_on():
    saving_popup.destroy()

Button(saving_popup, text='Save', command = save_results).grid(row=1, column=1)
Button(saving_popup, text='Forget', command=move_on).grid(row=1, column=2)

saving_popup.protocol("WM_DELETE_WINDOW", move_on) #if user closes window instead
saving_popup.mainloop()

