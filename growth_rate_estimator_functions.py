#### this file contains processing and interactive functions used by growth rate estimator GUI

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from scipy.stats import linregress
from scipy.stats import t
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from matplotlib.widgets import Button as MplButton



# 1. PROCESSING FUNCTIONS     (filters, smoothing, regression and computation of growth rate/lag time)

# Hampel filter -> get rid of outliters (removing them entirely)
def hampel_filter(timepts, data, window_size, n_sigmas):
    n = len(data)
    keep_mask = np.ones(n, dtype=bool)
    half_window = window_size // 2
    for i in range(half_window, n - half_window):
        # boundaries of current window
        start = i - half_window
        end = i + half_window + 1
        window = data[start:end]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        threshold = n_sigmas * mad
        if abs(data[i] - median) > threshold:
            keep_mask[i] = False # update mask to then remove
    filtered_data = np.array(data)[keep_mask]
    filtered_timepts =  np.array(timepts)[keep_mask]
    return filtered_timepts, filtered_data



# Smoothing (w/ Hanning window)
def hanning_smoothing(data, window_size):
    # data to be input as list or values
    # window size should be >=1 and odd
    if window_size>1:
        if window_size % 2 != 0:
            window_size += 1  #ensure window size is odd
        window = np.hanning(window_size)   #create weights for window
        smooth_data = np.convolve(data, window, mode='same')/sum(window)
        return smooth_data
    else: # window size not a valid number (not an natural non-0 number)
        return data

# Linear regression (using ordinary least squares)
def get_linear_fit(x, y):
    """
    Uses scipy.stats.linregress (ordinary least squares) to get slope and its standard error.
    The confidence intervals are computed based on t-distributions
    Bear in mind that lnOD data is heteroscedastic, so the error in slope is just an estimate and should be interpreted/compared with care. 
    """ 
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    results = linregress(x, y)
    slope = results.slope
    slope_err = results.stderr
    # compute 95% confidence intervals with t-critical
    tcrit = t.ppf(0.975, df=len(x)-2)
    ci = tcrit * slope_err # for plotting purposes
    return slope, slope_err, ci



def calculate_growth_parameters(times, ln_optical_densities, time_above_lod, od_above_lod, live_params):
    global_rate, rate_err, _ = get_linear_fit(times, ln_optical_densities)
    # calculate lag
    try:
        od_0 = live_params['OD_frozen_stock']
        dilution_factor = live_params['dilution_of_frozen_stock']
        path_correction = live_params['path_length_correction']
        lag_time = time_above_lod - (1/global_rate)*np.log(od_above_lod*dilution_factor/(path_correction*od_0))
    except:
        print("Parameters required to calculate the lag have not been given. Setting to NaN.")
        lag_time = np.nan
    return global_rate, rate_err, lag_time

def empty_result(well, df): # help clear data that user does not want to save
    return {
        'well':                      well,
        'sample':                    df.loc[df['well']==well, 'content'].tolist()[0],
        'exponential_points_bounds': (np.nan, np.nan),
        'growth_rate_1/h':           np.nan,
        'growth_rate_error':         np.nan,
        'doubling_time_min':         np.nan,
        'lag_time_h':                np.nan,
        'running_window':            np.nan,
        'hanning_window':            np.nan,
        'hampel_window':             np.nan,
        'hampel_sigmas':             np.nan,
    }


# 2. INTERACTIVE FUNCTIONS

def load_data():
    #get path interactively here
    path = fd.askopenfilename() # user selects file from file explorer GUI
    data = pd.read_csv(path, skiprows = 7).dropna()
    data.columns.astype(str).str.strip() != "" # solve issue with commas at the end of each rows
    data_tr = data.transpose()
    print(data)
    global time_pts, time_hours, wells, content
    time_pts = [float(i) for i in data.columns[2:]]
    time_hours = [time/60 for time in time_pts]
    wells = list(data_tr.iloc[0])
    content = list(data_tr.iloc[1])
    heads = ["content", "well", "time", "time_h", "OD"]
    df = pd.DataFrame(columns=heads)
    for wl in range(len(wells)):
        cont_list = []
        well_list = []
        for i in range(len(time_pts)):
            cont_list.append(content[wl])
            well_list.append(wells[wl])
        to_add = pd.DataFrame({
            "content": cont_list,
            "well": well_list,
            "time": time_pts,
            "time_h": time_hours,
            "OD": list(data_tr[wl])[2:],
            "lnOD": [np.log(a) for a in list(data_tr[wl])[2:]]
        })
        df = pd.concat([df, to_add], ignore_index=True)
    return(df, path) # raw data

def interactive_blanking(df):
    master = Tk()
    master.title("Select region which you want to use as blank")
    fig, ax = plt.subplots()
    for well in df["well"].unique():
        data_subset = df[df["well"] == well]
        ax.plot(data_subset["time_h"], data_subset["OD"]) 
    ax.set_xlabel("time (h)")   
    ax.set_ylabel("OD")
    ax.legend(df["well"].unique())
    ax.set_title("Raw growth curves \n Please select the region that should be considered the blank")
    ax.autoscale(enable=True)
    ax.margins(x=0)

    bounds = {"xmin": None, "xmax":None}

    class SelectBlanks:
        def __init__(self):
            self.xinit = None
            self.yinit = None
            self.xfin = None
            self.yfin = None
            self.rect = None
            self.goon = False #decide whether to continue changing the size of the rectangle or not

        def on_press(self, event):
            self.goon = True
            self.xinit, self.yinit = event.xdata, event.ydata
            bounds['xmin'] = self.xinit
            
        def on_motion(self, event):
            if self.goon:
                if self.rect:
                    self.rect.remove()
                self.xfin, self.yfin = event.xdata, event.ydata
                #calculate width and height
                width = self.xfin - self.xinit
                height = self.yfin - self.yinit
                self.rect = patches.Rectangle((self.xinit, self.yinit), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(self.rect)
                fig.canvas.draw_idle()

        def on_release(self, event):
            self.goon = False
            if self.rect:
                self.rect.remove()
            self.xfin, self.yfin = event.xdata, event.ydata
            width = self.xfin - self.xinit
            height = self.yfin - self.yinit
            bounds['xmax'] = self.xfin
            self.rect = patches.Rectangle((self.xinit, self.yinit), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(self.rect)
            fig.canvas.draw_idle()

    def confirmation():
        plt.close()
        master.quit()
        master.destroy()
        return()
            

    plot_blnk = SelectBlanks()
    fig.canvas.mpl_connect("button_press_event", plot_blnk.on_press)
    fig.canvas.mpl_connect("motion_notify_event", plot_blnk.on_motion)
    fig.canvas.mpl_connect("button_release_event", plot_blnk.on_release)
    Button(master, text="confirm selection", command = confirmation).grid(row = 1)
    
    plt.show(block=False) 
    master.mainloop()
    xmin = bounds['xmin']
    xmax = bounds['xmax']

    # compute blanked data based on xmin and xmax
    df_blanked = pd.DataFrame()
    for wl in range(len(wells)):
        well = wells[wl]
        #calculate average of points used for blanking
        df_with_blanks = df.query("well == @well and time_h>@xmin and time_h<@xmax")
        blank = np.average(df_with_blanks["OD"].to_list())
        df_to_blank_data = df.query("well == @well")
        list_ods = df_to_blank_data["OD"].to_list()
        list_ods_blanked = [round((a-blank), 3) for a in list_ods]
        list_ods_blanked_ln = [np.log(a) if a>0 else np.nan for a in list_ods_blanked]
        #create new df (df_blanked) similar to original df but with blanked values instead of the raw values
        cont_list = []
        well_list = []
        for i in range(len(time_pts)):
            cont_list.append(content[wl])
            well_list.append(wells[wl])
        to_add = pd.DataFrame({
            "content": cont_list,
            "well": well_list,
            "time": time_pts,
            "time_h": time_hours,
            "OD": list_ods_blanked,
            "lnOD": list_ods_blanked_ln
        })
        df_blanked = pd.concat([df_blanked, to_add], ignore_index = True)
    return df_blanked # contains blank data




def create_parameter_window(live_params, default_params):
    """
    Create the interactive window that will allow the user to input parameters related to current growth curve;
    Once confirmed, all parameters will be carried over to the next curve, unless reset.
    """

    win = Tk()
    win.title("Processing parameters")

    variables = {k: StringVar(value=str(v)) for k, v in live_params.items()} # create a dictionary to be linked to the tkinter window

    def row(r, label, key):
        ttk.Label(win, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
        e = ttk.Entry(win, textvariable=variables[key], width=10)
        e.grid(row=r, column=1, sticky="w", padx=6, pady=4)
        e.bind("<Return>", lambda _=None: do_apply())


    row(0, "OD lower bound", "od_lower")
    row(1, "OD upper bound", "od_upper")
    row(2, "Time lower bound (h)", "time_lower")
    row(3, "Time upper bound (h)", "time_upper")
    row(4, "Running window size", "running_window")
    row(5, "Outliers (number of sigmas)", "outlier_sigmas")
    row(6, "Outliers (window size)", "outlier_window")
    row(7, "Smoothing (window size)", "smoothing_window")
    row(8, "Frozen stock OD", "OD_frozen_stock")
    row(9, "Frozen stock dilution", "dilution_of_frozen_stock")
    row(10,"Path length correction factor", "path_length_correction")

    def do_apply():
        # update global params from the entries (minimal validation)
        try:
            live_params["od_lower"]        = float(variables["od_lower"].get())
            live_params["od_upper"]        = float(variables["od_upper"].get())
            live_params["time_lower"]      = float(variables["time_lower"].get())
            live_params["time_upper"]      = float(variables["time_upper"].get())
            live_params["running_window"]  = max(3, int(variables["running_window"].get()))
            live_params["outlier_sigmas"]  = float(variables["outlier_sigmas"].get())
            ow = max(1, int(variables["outlier_window"].get()))
            if ow % 2 == 0:
                ow += 1
            live_params["outlier_window"]  = ow
            live_params["smoothing_window"] = max(1, int(variables["smoothing_window"].get()))
        except ValueError:
            print("Invalid parameter, please enter numbers.")
            return

        try: # culture setup parameters are optional, but if not given, lag time will be NaN
            live_params["OD_frozen_stock"]        = float(variables["OD_frozen_stock"].get())
            live_params["dilution_of_frozen_stock"]        = float(variables["dilution_of_frozen_stock"].get())
            live_params["path_length_correction"]      = float(variables["path_length_correction"].get())
        except ValueError:
            print("Lag time cannot be computed with current parameters.")

        # call current recompute hook if set
        global recompute_hook
        if recompute_hook:
            recompute_hook()

    ttk.Button(win, text="Apply to current well", command=do_apply)\
        .grid(row=11, column=0, columnspan=2, padx=6, pady=(8, 4), sticky="ew")

    def do_reset():
        # reset entry fields to defaults
        for k, v in default_params.items():
            variables[k].set(str(v))
        # also update the live params + replot
        do_apply()

    ttk.Button(win, text="Reset to defaults", command=do_reset)\
        .grid(row=12, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="ew")

    # keep window alive unless the user closes it on purpose
    # (will also terminate when there are no more wells to plot - see main)
    win.protocol("WM_DELETE_WINDOW", win.destroy)
    return win



# Click handler for selecting the bounds of exponential phase (2-point selection)
def on_point_click(event, state, ax1, ax2, well, df, live_params, selection_buffer, selections_to_save):
    if event.inaxes != ax2:
        return
    x_click, y_click = event.xdata, event.ydata
    if x_click is None or y_click is None: #if user clicks outside of the axes area
        return

    # get rid of any potential NaN values
    growth_rates = state['growth_rate_series']

    nan_mask = (np.isnan(growth_rates))
    time_series_for_growth_rates = state['time_series_for_growth_rates']
    growth_rates_valid = growth_rates[~nan_mask]
    time_series_valid = time_series_for_growth_rates[~nan_mask]

    # compute distances from user click to each point and find the closest one
    distances = np.sqrt((time_series_valid - x_click)**2 + (growth_rates_valid - y_click)**2)
    closest_point_id_valid = np.argmin(distances) # this is the ID in the valid series only
    closest_point_id = np.where(~nan_mask)[0][closest_point_id_valid] # point in the real array
    
    # buffer to hold current selections
    if len(selection_buffer) >= 2:
        selection_buffer.clear() # empty buffer
        ax1.scatter(state['time_series_filtered_sliced'], state['lnOD_data_filtered_sliced'], s=10, color='tab:blue', label='Exponential phase') # get rid of highlighted points in previous selection

    selection_buffer.append(closest_point_id) #IDs of exp bounds in the time_series_for_growth_rates

    if len(selection_buffer) == 2:
        # get exp bounds in time_series and lnOD_data
        rw = live_params['running_window']
        # real_indices = (min(selection_buffer)-rw//2, max(selection_buffer)+rw//2 -1) - this is wrong, they're the indices
        real_indices = (min(selection_buffer), max(selection_buffer)+rw-1)
        exponential_times = state['time_series_filtered_sliced'][real_indices[0]:real_indices[1]]
        exponential_lnODs = state['lnOD_data_filtered_sliced'][real_indices[0]:real_indices[1]]

        # compute growth time, error and lag time
        global_rate, rate_err, lag_time = calculate_growth_parameters(
            exponential_times, 
            exponential_lnODs,
            exponential_times[0],
            np.exp(exponential_lnODs[0]),
            live_params
        )
        

        # append to dict of final selections
        selections_to_save[well] = {
            'well':                 well,
            'sample':               df.loc[df['well']==well, 'content'].tolist()[0], # returns a string
            'exponential_points_bounds': (exponential_times[0], exponential_times[-1]), # contains time point bounds for selected exponential
            'growth_rate_1/h':      global_rate if global_rate is not None else np.nan,
            'growth_rate_error':    rate_err if rate_err is not None else np.nan,
            'doubling_time_min':    np.log(2) / global_rate * 60 if (global_rate and global_rate > 0) else np.nan,
            'lag_time_h':           lag_time if lag_time is not None else np.nan,
            'running_window':       live_params['running_window'],
            'hanning_window':       live_params['smoothing_window'],
            'hampel_window':        live_params['outlier_window'],
            'hampel_sigmas':        live_params['outlier_sigmas']
        }

    # redraw RHS plot to update with newly selected point
    ax2.clear()
    ax2.errorbar(time_series_for_growth_rates, growth_rates, yerr=state['growth_rate_cis'], picker=5, fmt='o', linestyle=None, capsize=5, label='Instantaneous growth rates', alpha=0.6)


    if len(selection_buffer) == 1: # only draw line
        t1  = time_series_for_growth_rates[selection_buffer[0]]
        gr1 = growth_rates[selection_buffer[0]]
        ax2.scatter([t1], [gr1], color='red', s=100, marker='x', label='Selected') # highlight selected point
        ax2.axvline(x=t1, color='green', linestyle='--', alpha=0.5)


    elif len(selection_buffer) == 2: #draw both lines
        start, end = min(selection_buffer), max(selection_buffer)
        t_start = time_series_for_growth_rates[start]
        t_end   = time_series_for_growth_rates[end]
        exponential_cis = state['growth_rate_cis']
        # highlight the points that are selected as exponential 
        ax2.errorbar(time_series_for_growth_rates[start:end+1], growth_rates[start: end+1], yerr = exponential_cis[start: end+1], picker=5, fmt='o', linestyle=None, capsize=5, label='Exponential growth rates', alpha=0.6, color = 'red')
        ax2.axvline(x=t_start, color='green', linestyle='--', alpha=0.5)
        ax2.axvline(x=t_end,   color='green', linestyle='--', alpha=0.5)
        ax2.axvspan(t_start, t_end, alpha=0.15, color='green', label='selected exponential')
        ax2.axhline(y = global_rate, color = 'red', label = f'global growth rate ({global_rate:.2f}/h)')

        # replot points that correspond to the picked exponential phase
        ax1.scatter(exponential_times, exponential_lnODs, s=10, color='purple', label='Exponential phase')
        ax1.figure.canvas.draw_idle()
    ax2.set_title(f"Growth rate for well {well} \n ({df.loc[df['well']==well, 'content'].tolist()[0]})")
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Growth rate (1/h)")
    ax2.set_ylim(-0.05, 3)
    ax2.legend()
    plt.draw()


               
# Button handlers for user decision before moving on to next well

def on_keep_click(event, fig, well, df, selections_to_save, selection_buffer):
    if len(selection_buffer) < 2:
        print(f"No selection made for well {well}. Saving as NaN.")
        selections_to_save[well] = empty_result(well, df)
        selection_buffer.clear()
    else:
        print(f"Confirmed selection for well {well}.")
    global proceed_flag
    proceed_flag = True
    plt.close(fig)

def on_skip_click(event, fig, well, df, selections_to_save, selection_buffer):
    print(f"Skipping well {well}.")
    selections_to_save[well] = empty_result(well, df)
    selection_buffer.clear()
    global proceed_flag
    proceed_flag = True
    plt.close(fig)


def on_exit_click(event, fig, well, df, selections_to_save, selection_buffer, exit_requested):
    print("Exit button clicked. Terminating script...")
    # clear last well since it was never confirmed, so should not be saved
    selections_to_save[well] = empty_result(well, df)
    selection_buffer.clear()
    exit_requested['value'] = True
    global proceed_flag
    proceed_flag = True #unblock while in main
    plt.close(fig)


# Define function that will take the data for current well and allow user to interact with it and get the growth rate
def process_well_interactive(well, df, live_params, selections_to_save, selection_buffer, exit_requested):
    
    """
    For current well, it will do the following, each time it gets "recomputed" (due to a new set of processing parameters being applied by user):
    1. get live parameters and preselected exponential bounds (if available)
    2. create array with time data and lnOD data sliced around the time bounds
    3. plot this data on LHS plot (light blue points, static)
    4. apply the 2 filters on the data
    5. plot this data on LHS plot (bright orange, static)
    6. highlight selected exponential (if available) on LHS plot in purple
    7. plot growth rate against time on RHS plot based on filtered data (interactive)
    8. on RHS plot, draw a line of slope 0 at y = (slope calculated on LHS)
    9. legend on RHS should contain slope and error in slope fit from LHS (i.e., the info on growth rate for current curve)
    10. plots title to include info on current well (i.e., what is included in medium)

    """

    # initialise bag with latest RHS plot data that allows click handler to see the latest computed values
    state ={
        "time_series_filtered_sliced": None,
        "lnOD_data_filtered_sliced": None,
        "time_series_for_growth_rates": None,
        "growth_rate_series": None,
        "growth_rate_cis": None

    }

    # Create figure only once for current curve
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(bottom=0.15, wspace=0.3)

    def recompute(): # recomputes growth rate against time with new parameters; choice of exponential is unaffected
        # 1. read live parameters for slicing, filter parameters and params required to calculate the lag
        t_lower = live_params['time_lower']
        t_upper = live_params['time_upper']
        od_lower = live_params['od_lower']
        od_upper = live_params['od_upper']

        running_window = live_params["running_window"]
        hampel_sigmas  = live_params["outlier_sigmas"]
        hampel_window  = live_params["outlier_window"]
        hanning_window = live_params["smoothing_window"]


        # 2. slice at time bounds
        time_series = df.loc[(df['well'] == well) & (df['time_h'] > t_lower) & (df['time_h'] < t_upper), "time_h"].tolist()
        lnOD_data = df.loc[(df['well'] == well) & (df['time_h'] > t_lower) & (df['time_h'] < t_upper), "lnOD"].tolist()

        if not time_series:
            print(f"There's no data for well {well} between {t_lower}h and {t_upper}h")
            result = empty_result(well, df)
            return result

        # 3. filter and smooth
        time_series_filtered, lnOD_data_filtered = hampel_filter(time_series, lnOD_data, hampel_window, hampel_sigmas) # to fill this in once function is defined 
        lnOD_data_filtered = hanning_smoothing(lnOD_data_filtered, window_size = hanning_window)

        # 4. slice by OD bounds - done after filtering to avoid averrant early spikes
        time_series_filtered_sliced = np.array(time_series_filtered)
        lnOD_data_filtered_sliced = np.array(lnOD_data_filtered)
        mask_od = (lnOD_data_filtered_sliced >= np.log(od_lower)) & (lnOD_data_filtered_sliced <= np.log(od_upper))  # mask checking in which positions criteria are not met
        time_series_filtered_sliced = time_series_filtered_sliced[mask_od] #remove data points at which lnOD is not within the defined bounds
        lnOD_data_filtered_sliced = lnOD_data_filtered_sliced[mask_od]

        # 5. compute slope with rolling window
        growth_rates_series = [] #initialise
        growth_rate_cis = []
        for i in range(len(lnOD_data_filtered_sliced) - (running_window-1)):
            slope, slope_err, ci = get_linear_fit(
                time_series_filtered_sliced[i : i+running_window], 
                lnOD_data_filtered_sliced[i : i+running_window]
            )
            growth_rates_series.append(slope)
            growth_rate_cis.append(ci)

        # 6. redraw for current data
        axs[0].clear()
        axs[1].clear()
        # LHS plot
        axs[0].scatter(time_series, lnOD_data, label="lnOD (raw)", s=10, color='gray', alpha=0.4)
        axs[0].scatter(time_series_filtered, lnOD_data_filtered, label="lnOD (filtered)", s=10, color='tab:blue')
        axs[0].axhline(np.log(od_lower), color='mediumorchid', linestyle='--', alpha=0.7, label=f'OD>{od_lower}')
        axs[0].axhline(np.log(od_upper),   color='mediumorchid', linestyle='--', alpha=0.7, label=f'OD<{od_upper}')        
        axs[0].set_xlabel("Time (h)")
        axs[0].set_ylabel("lnOD")
        axs[0].set_title(f"Growth curve for well {well} \n ({df.loc[df['well'] == well, 'content'].tolist()[0]})")
        axs[0].legend()
        #axs[0].set_ylim(-5.5,0)
        # create empty twin plot such that actual OD values are shown on the on RHS of the LHS plot
        twinax = axs[0].twinx()
        ylims = axs[0].get_ylim(); twinax.set_ylim(np.exp(ylims))
        twinax.set_ylabel("OD"); #twinax.set_yscale('log')

        # RHS plot
        tps_to_plot = time_series_filtered_sliced[running_window//2 : (len(time_series_filtered_sliced) - (running_window+1)//2 + 1)] #center around middle time point in each window
        axs[1].errorbar(tps_to_plot, growth_rates_series, yerr=growth_rate_cis, picker=5, fmt='o', linestyle=None, capsize=5, label='Instantaneous growth rates', alpha=0.6)
        axs[1].set_xlabel("Time (h)")
        axs[1].set_ylabel("Growth rate (1/h)")
        axs[1].set_ylim(-0.05, 3)
        axs[1].set_title(f"Growth rate for well {well} \n ({df.loc[df['well'] == well, 'content'].tolist()[0]})")
        axs[1].legend()

        fig.canvas.draw_idle()

        # 7. update state with current interactive points on RHS
        state['time_series_filtered_sliced'] = np.array(time_series_filtered_sliced)
        state['growth_rate_series'] = np.array(growth_rates_series)
        state['lnOD_data_filtered_sliced'] = np.array(lnOD_data_filtered_sliced)
        state['time_series_for_growth_rates'] = np.array(tps_to_plot)
        state['growth_rate_cis'] = np.array(growth_rate_cis)


    global recompute_hook 
    recompute_hook = recompute # make this function visible to on point click with current params
    recompute() # run

    # Click handler reads from state, which recompute() keeps up to date
    fig.canvas.mpl_connect(
        'button_press_event',
        lambda event: (
            state['time_series_filtered_sliced'] is not None
            and on_point_click(event, state, axs[0], axs[1], well, df, live_params, selection_buffer, selections_to_save)
        ),
    )

    # User decision buttons. Keep, discard or abort program -> change coordinates to make it look ok!!!
    button_ax_keep = plt.axes([0.65, 0.002, 0.1, 0.06])
    confirm_button = MplButton(button_ax_keep, 'keep', color='springgreen', hovercolor='thistle')
    button_ax_skip = plt.axes([0.775, 0.002, 0.1, 0.06])
    discard_button = MplButton(button_ax_skip, 'skip', color='tomato', hovercolor='thistle')
    button_ax_exit = plt.axes([0.9, 0.002, 0.1, 0.06])
    exit_button = MplButton(button_ax_exit, 'exit', color='firebrick', hovercolor='thistle')

    global proceed_flag
    proceed_flag = False
    confirm_button.on_clicked(lambda event: on_keep_click(event, fig, well, df, selections_to_save, selection_buffer))
    discard_button.on_clicked(lambda event: on_skip_click(event, fig, well, df, selections_to_save, selection_buffer))
    exit_button.on_clicked(lambda event: on_exit_click(event, fig, well, df, selections_to_save, selection_buffer, exit_requested))

    plt.show(block=True)

    # Wait for user to decide what to do next
    while not proceed_flag:
        plt.pause(0.1)
    return 
    

