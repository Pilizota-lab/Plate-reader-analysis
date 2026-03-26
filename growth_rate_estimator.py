#%%

# Contact: Diana Coroiu
# Affiliation (at the time of writing): University of Edinburgh
# Contact:
#   - personal: didicoroiu@yahoo.com 
#   - institutional: d.coroiu@sms.ed.ac.uk
# Last updated: 26.03.2026

## Acknowledgements
# This GUI was initially based on an early draft written by James Broughton. 

# Description:

# This code allows to estimate growth rates from plate reader data
# Software iterates through wells and plot data that is likely to be representative of exponential phase (i.e., between ODs 0.005 and 0.6).


# The raw data has to be in the .csv format and follow a given structure:
# (1): table headings in row 8, data start in row 9
# (2): time to be given in minutes (otherwise time scale will not make sense)
# (3): axis 0 (rows) to contain different wells, axis 1 (columns) to contain different time points
# (4): wells labelled as "A01" and not "A1"


# Intended for academic use; feel free to adapt for other plate-reader data file formats.

# Full instructions and details on GitHub: https://github.com/Pilizota-lab/Plate-reader-analysis/


# final working GUI for esimating growth rates from plate reader data - last updated: March 2026
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
from growth_rate_estimator_functions import * # custom functions - ensure functions file is in current working folder

# default parameters -> not modified by user so that one can always reset to defaults
default_params = {
    # OD boundaries
    "od_lower": 0.05,
    "od_upper": 1.5,
    # time point boundaries if experiment ran for too long (in hours)
    "time_lower": 0,
    "time_upper": 35,
    # windows and filters - DEFAULTS AS BELOW, user choices saved to results for each curve
    "running_window": 7, # number of point on which each slope is fitted to generate the growth rate against time plot; should be at least 3
    "outlier_sigmas": 0.5, # number of stdevs outside of which the point is excluded
    "outlier_window": 9,  
    "smoothing_window": 1,  # smoothing with a hanning window (if not needed, set window to 1)
    # parameters required to compute lag time
    "OD_frozen_stock": None, 
    "dilution_of_frozen_stock": None,
    "path_length_correction": None
}
# live parameters -> updated every time user presses "apply to current well" (and will be carried over to next well unless reset by user)
live_params = default_params.copy() # initialise to defaults. Can be updated by user during run

# This will point to "recompute current well" function
recompute_hook = None

### START PROGRAM ###

root = Tk()
root.withdraw()

# ask for path length correction value
correction_window = Toplevel()

ttk.Label(correction_window, text='Please provide the path length correction').pack()
entry = ttk.Entry(correction_window)
entry.pack()

error_label = ttk.Label(correction_window, text='')
error_label.pack()
pathlength_correction = None

def get_pathlength_correction():
    global pathlength_correction
    try:
        pathlength_correction = float(entry.get())
        if pathlength_correction>0 and pathlength_correction<=1: # set to 1 if user wants to stick to reader scale
            print(f'path length correction confirmed: {pathlength_correction}')
            correction_window.destroy()
        else:
            ttk.Label(correction_window, text='Please select a value between 0 and 1.').pack()
    except:
        ttk.Label(correction_window, text='Please enter a number.').pack()

ttk.Button(correction_window, text="Confirm", command = get_pathlength_correction).pack()
correction_window.wait_window()


df, path = load_data(pathlength_correction)

# BLANK
initial_window = Toplevel()
initial_window.geometry("350x100")
initial_window.title("Would you like to blank your data?")


def yes_blank():
    global df
    df = interactive_blanking(df) # overwrite raw data
    initial_window.destroy()
    return

def no_blank():
    messagebox.showwarning("Warning", "Your data has not been blanked.")
    initial_window.destroy()
    return

yes_button = Button(initial_window, text="Yes", command = yes_blank)
yes_button.place(x=100, y=40)
no_button = Button(initial_window, text ="No", command = no_blank)
no_button.place(x=200, y=40)
initial_window.protocol("WM_DELETE_WINDOW", no_blank)
initial_window.wait_window()

# containers for results
selections_to_save = {}  # stores selection-related vars for each well: {'time': (t1,t2), 'growth_rate': float, 'indices': (i1,i2)}
selection_buffer = []     # Temporary click buffer (not well-specific)
exit_requested = {"value": False}

# MAIN SEQUENCE
parameter_window = create_parameter_window(live_params, default_params)

try:
    unique_wells = df['well'].unique()
    print(f"{len(unique_wells)} wells found. Starting processing:")
    
    for well in unique_wells:
        print(f"\n--- Processing well {well} ---")
        process_well_interactive(well, df, live_params, selections_to_save, selection_buffer, exit_requested)
        if exit_requested['value']:
            break
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(selections_to_save.values())
    
except NameError as e:
    print(f"Error: DataFrame df not defined or missing columns. Check raw file structure. Details: {e}")
except SystemExit as e:
    print(e)
    # Save partial results before exiting
    if selections_to_save:
        results_df = pd.DataFrame(selections_to_save.values())
except Exception as e:
    print(f"An error occurred: {e}")

# Clean up
plt.close('all')


# get results summary:
print(f"\n Results summary: \n")

summary_per_conc = (
    results_df.groupby("sample")
    .agg(
        growth_rate_avg=   ("growth_rate_1/h",     "mean"),
        growth_rate_std =   ("growth_rate_1/h",   "std"),
        dbl_time_avg= ("doubling_time_min",   "mean"),
        lag_time_avg=      ("lag_time_h",          "mean"),
        lag_time_stdev=       ("lag_time_h",          "std"), 
        n_replicates=       ("growth_rate_1/h",     "count"),  # count of successfully computed replicates
    )
    .reset_index()
)

print(summary_per_conc)


# save results to CSV (same path/file name + growth_rates at the end) - if user wants to do so
parameter_window.destroy()
saving_popup = Toplevel()
saving_popup.title('Do you want to write results to disk?')
def save_results():
    results_df.to_csv((path[:-4] + '_growth_rates.csv'), index=False)
    saving_popup.destroy()
    print('data saved to disk next to the raw data.')
    root.quit()

def move_on():
    saving_popup.destroy()
    root.quit()

Button(saving_popup, text='Save', command = save_results).grid(row=1, column=1)
Button(saving_popup, text='Forget', command=move_on).grid(row=1, column=2)

saving_popup.protocol("WM_DELETE_WINDOW", move_on) #if user closes window instead

root.mainloop()

import os
os._exit(0)
