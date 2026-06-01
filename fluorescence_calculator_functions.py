#### this file contains processing and interactive functions used by fluorescence calculator GUI

import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd
import seaborn as sns
from scipy.stats import sem
from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
from matplotlib.widgets import Button as MplButton
import os



# 1. PROCESSING FUNCTIONS     (filters, smoothing, regression and computation of growth rate/lag time)

# Hampel filter -> get rid of outliters (removing them entirely)
def hampel_filter(timepts_min, timepts_h, data, window_size, n_sigmas):
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
    filtered_timepts_min =  np.array(timepts_min)[keep_mask]
    filtered_timepts_h =  np.array(timepts_h)[keep_mask]
    return filtered_timepts_min, filtered_timepts_h, filtered_data, keep_mask



# Smoothing (w/ Hanning window)
def hanning_smoothing(data, window_size):
    # data to be input as list or values
    window = np.hanning(window_size)   #create weights for window
    smooth_data = np.convolve(data, window, mode='same')/sum(window)
    return smooth_data



def normalise(well, df_fluorescence, df_od, live_params):
    '''
    Only normalise data for current well
    '''
    # get live parameters related to smoothing and filtering
    filter_win = live_params['outlier_window']
    filter_sigma = live_params['outlier_sigmas']
    smooth_win = live_params['smoothing_window']
    df_normalised = pd.DataFrame()
    # apply filter and smoothing on OD - some points will be dropped from time points and data, using the returned mask to decide what corresponding fluorescence points to remove
    timepts_min = df_od.loc[df_od['well']==well, 'time'].to_numpy()
    timepts_h = df_od.loc[df_od['well']==well, 'time_h'].to_numpy()
    od_vals =  df_od.loc[df_od['well']==well, 'OD'].to_numpy()
    filtered_time_mins, filtered_time_h, filtered_od, mask = hampel_filter(timepts_min, timepts_h, od_vals, filter_win, filter_sigma)
    try:
        fluo_vals = df_fluorescence.loc[df_fluorescence['well']==well, 'OD'].to_numpy()
        filtered_fluorescence = np.array(fluo_vals)[mask]
    except:
        print(f"No fluorescence data found for well {well}.")
        # save empty result for current well
        return pd.DataFrame(columns = ['content', 'time', 'time_h', 'normalised_fluorescence', 'raw_fluorescence', 'OD'])

    smooth_filtered_od = hanning_smoothing(filtered_od, smooth_win)
    df_normalised = pd.DataFrame({
        "well": well,
        "content": df_od.loc[df_od['well']==well, 'content'].tolist()[0],
        "time": filtered_time_mins, #this is time in minutes as recorded in the fluorescence file; only required to save to output so that user can retrieve previously selected exponential
        "time_h": filtered_time_h,
        "normalised_fluorescence": filtered_fluorescence/smooth_filtered_od,
        "raw_fluorescence": filtered_fluorescence,
        "OD": smooth_filtered_od
    })

    return df_normalised


def empty_result(well, df_od): # help clear data that user does not want to save
    return {
        'well':                      well,
        'sample':                    df_od.loc[df_od['well']==well, 'content'].tolist()[0],
        'exponential_points_bounds': (np.nan, np.nan),
        'exponential_fluorescence':  np.nan,
        'exponential_fluorescence_err': np.nan,
        # parameters used to get rid of noise in OD
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
    print(f"Analysing data from {path}")
    # global time_pts, time_hours, wells, content
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

def interactive_blanking_od(df):
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
    wells = df['well'].unique().tolist()


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
        to_add = pd.DataFrame({
            "content": df.loc[df['well']==well, 'content'].tolist(),
            "well": well,
            "time":  df.loc[df['well']==well, 'time'].tolist(),
            "time_h":  df.loc[df['well']==well, 'time_h'].tolist(),
            "OD": list_ods_blanked,
            "lnOD": list_ods_blanked_ln
        })
        df_blanked = pd.concat([df_blanked, to_add], ignore_index = True)

    return df_blanked # contains blank data



def blank_fluorescence(df_to_blank):
    # this requires a special function due to media photobleaching
    df_background, _ = load_data() # user selects file from file explorer GUI
    wells = df_to_blank['well'].unique().tolist()

    # compute blanked data by subtracting the average background at each time point
    df_blanked = pd.DataFrame()


    for wl in range(len(wells)):
        well = wells[wl]
        #calculate average of points used for blanking at each time point
        average_bkgr = df_background.groupby("time_h")['OD'].mean().tolist()
        raw_vals = df_to_blank.loc[df_to_blank['well']==well, 'OD'].tolist()
        average_bkgr = average_bkgr[:len(raw_vals)]
        blanked_vals = np.array(raw_vals) - np.array(average_bkgr)
        blanked_vals = [round(a, 0) for a in blanked_vals]
        
        #create new df (df_blanked) similar to original df but with blanked values instead of the raw values
        to_add = pd.DataFrame({
            "content": df_to_blank.loc[df_to_blank['well']==well, 'content'].tolist(),
            "well": well,
            "time":  df_to_blank.loc[df_to_blank['well']==well, 'time'].tolist(),
            "time_h":  df_to_blank.loc[df_to_blank['well']==well, 'time_h'].tolist(),
            "OD": blanked_vals,  # this is actually "blanked fluorescence" but kept as OD for consistency with the other blanking method
        })
        df_blanked = pd.concat([df_blanked, to_add], ignore_index = True)
    
    print(df_blanked)

    return df_blanked # contains blank data


def blank_fluorescence_2(df_to_blank):
    # this is another way to blank fluorescence, by subtracting the average of the 5 lowest fluorescence points from each well.
    wells = df_to_blank['well'].unique().tolist()
    # compute blanked data by subtracting the average background at each time point
    df_blanked = pd.DataFrame()

    for wl in range(len(wells)): # wl is well idx
        well = wells[wl]
        # calculate average background to be subtracted from current well 
        average_bkgr = df_to_blank.query("well==@well")['OD'].astype(float).nsmallest(5).mean()
        raw_vals = df_to_blank.loc[df_to_blank['well']==well, 'OD'].tolist()
        blanked_vals = [a - average_bkgr for a in raw_vals]
        blanked_vals = [round(a, 0) for a in blanked_vals]
        
        #create new df (df_blanked) similar to original df but with blanked values instead of the raw values
        to_add = pd.DataFrame({
            "content": df_to_blank.loc[df_to_blank['well']==well, 'content'].tolist(),
            "well": well,
            "time":  df_to_blank.loc[df_to_blank['well']==well, 'time'].tolist(),
            "time_h":  df_to_blank.loc[df_to_blank['well']==well, 'time_h'].tolist(),
            "OD": blanked_vals,  # this is actually "blanked fluorescence" but kept as OD for consistency with the other blanking method
        })
        df_blanked = pd.concat([df_blanked, to_add], ignore_index = True)
    
    print(df_blanked)

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


    row(0, "Time lower bound (h)", "time_lower")
    row(1, "Time upper bound (h)", "time_upper")
    row(2, "Outliers (number of sigmas)", "outlier_sigmas")
    row(3, "Outliers (window size)", "outlier_window")
    row(4, "Smoothing (window size)", "smoothing_window")


    def do_apply():
        # update live params from the entries (minimal validation)
        try:
            live_params["time_lower"]      = float(variables["time_lower"].get())
            live_params["time_upper"]      = float(variables["time_upper"].get())
            live_params["outlier_sigmas"]  = float(variables["outlier_sigmas"].get())

            ow = max(1, int(variables["outlier_window"].get()))
            if ow % 2 == 0:
                ow += 1
            live_params["outlier_window"]  = ow
            variables["outlier_window"].set(ow)

            sw = max(1, int(variables["smoothing_window"].get()))
            if sw % 2 == 0:
                sw += 1
            live_params["smoothing_window"] = max(1, int(variables["smoothing_window"].get()))
            variables["smoothing_window"].set(sw)
        except ValueError:
            print("Invalid parameter, please enter numbers.")
            return

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


def replot_all(axs, df_od, df_normalised, current_well, live_params): # when new well is displayed, or when data is "recomputed" or user selection of exponential is changed
    '''
    Adds all elements required by the plots (labels, limits, titles etc).
    axs is an array of 3 axes
    '''
    xlims = (live_params['time_lower'], live_params['time_upper'])
    sample = df_normalised.loc[df_normalised['well']==current_well, 'content'].tolist()[0]
    axs[0].scatter(df_normalised.loc[df_normalised['well']==current_well, 'time_h'], df_normalised.loc[df_normalised['well']==current_well, 'normalised_fluorescence'], s=5, picker = 5, color = 'green', label='all data')
    axs[0].set_xlabel('Time (h)')
    axs[0].set_ylabel('Normalised fluorescence (A.U.)')
    axs[0].set_title(f'Normalised fluorescence for well {current_well} \n ({sample})')
    axs[0].set_xlim(xlims[0], xlims[1])
    axs[0].set_ylim(bottom=-100)

    axs[1].scatter(df_normalised.loc[df_normalised['well']==current_well, 'time_h'], df_normalised.loc[df_normalised['well']==current_well, 'OD'], color = 'blue', s=5, label='filtered')
    axs[1].scatter(df_od.loc[df_od['well']==current_well, 'time_h'], df_od.loc[df_od['well']==current_well, 'OD'], color = 'gray', label='raw points', alpha = 0.3, s=5, zorder=0)
    axs[1].set_xlabel('Time (h)')
    axs[1].set_ylabel('Optical Density')
    axs[1].set_yscale('log')
    axs[1].set_title(f'OD for well {current_well} ({sample})')
    axs[1].set_xlim(xlims[0], xlims[1])
    axs[1].set_ylim(bottom=0.01)

    axs[2].scatter(df_normalised.loc[df_normalised['well']==current_well, 'time_h'], df_normalised.loc[df_normalised['well']==current_well, 'raw_fluorescence'], s=5, color = 'palegreen', label='all data')
    axs[2].set_xlabel('Time (h)')
    axs[2].set_ylabel('Raw fluorescence (A.U.)')
    axs[2].set_title(f'Raw fluorescence for well {current_well} \n ({sample})')
    axs[2].set_xlim(xlims[0], xlims[1])


# Click handler for selecting the bounds of exponential phase (2-point selection)
def on_point_click(event, axs, well, df_od, df_normalised, live_params, selection_buffer, selections_to_save):
    # axs = array of 3 axes
    if event.inaxes != axs[0]:
        return
    x_click, y_click = event.xdata, event.ydata
    if x_click is None or y_click is None: #if user clicks outside of the axes area
        return

    # convert all relevant series to numpy arrays (otherwise indexing issues are painful to solve)
    times = df_normalised['time_h'].to_numpy()
    norm_fluo = df_normalised['normalised_fluorescence'].to_numpy()
    od = df_normalised['OD'].to_numpy()
    raw_fluo = df_normalised['raw_fluorescence'].to_numpy()


    # compute distances from user click to each point and find the closest one
    # distances = np.sqrt((times - x_click)**2 + (norm_fluo - y_click)**2)
    # closest_point_id = np.argmin(distances) # this is the ID in the valid series only
    x_ax_range = axs[0].get_xlim()[1] - axs[0].get_xlim()[0] 
    y_ax_range = axs[0].get_ylim()[1] - axs[0].get_ylim()[0]
    distances = np.sqrt(((times - x_click)/x_ax_range)**2 + ((norm_fluo - y_click)/y_ax_range)**2)
    closest_point_id = np.nanargmin(distances)
    print(f"clicked at x={x_click:.2f}, y={y_click:.2f}")
    print(f"closest point: index={closest_point_id}, x={times[closest_point_id]:.2f}, y={norm_fluo[closest_point_id]:.2f}")

    # buffer to hold current selections
    if len(selection_buffer) >= 2: # reset
        selection_buffer.clear() # empty buffer
        axs[0].clear(); axs[1].clear(); axs[2].clear() # clear plots to get rid of selections 
        replot_all(axs, df_od, df_normalised, well, live_params) # replot all
        axs[0].legend(); axs[1].legend(); axs[2].legend()

    selection_buffer.append(closest_point_id) # ID within the filtered data (coming from df_normalised)

    if len(selection_buffer) == 2:
        # get exp points and replot in red
        lo = min(selection_buffer[0], selection_buffer[1])
        hi = max(selection_buffer[0], selection_buffer[1]) + 1
        exponential_times = times[lo:hi]
        exponential_norm_fluo = norm_fluo[lo:hi]
        exponential_ODs = od[lo:hi]
        exponential_raw_fluo = raw_fluo[lo:hi]

        axs[0].scatter(exponential_times, exponential_norm_fluo, s=5, color = 'red', label = 'exponential')
        axs[1].scatter(exponential_times, exponential_ODs, s=5, color = 'red', label = 'exponential')
        axs[2].scatter(exponential_times, exponential_raw_fluo, s=5, color = 'red', label = 'exponential')

        # calculate mean fluorescence in current selection
        exponential_fluorescence = np.average(exponential_norm_fluo)
        exponential_fluo_err = sem(exponential_norm_fluo)
        # draw a flat line at y = exponential_fluorescence as well as highlighting exponential span
        axs[0].plot(exponential_times, [exponential_fluorescence for a in exponential_times], color = 'darkred', label = f"avg fluo = {round(exponential_fluorescence,0)} +/ {round(exponential_fluo_err, 0)}")
        axs[0].axvline(x=exponential_times[0], color='red', linestyle='--')
        axs[1].axvline(x=exponential_times[0], color='red', linestyle='--')
        axs[2].axvline(x=exponential_times[0], color='red', linestyle='--')
        axs[0].axvline(x=exponential_times[-1], color='red', linestyle='--')
        axs[1].axvline(x=exponential_times[-1], color='red', linestyle='--')
        axs[2].axvline(x=exponential_times[-1], color='red', linestyle='--')

        axs[0].axvspan(exponential_times[0], exponential_times[-1], alpha=0.15, color='red')
        axs[0].legend(); axs[1].legend(); axs[2].legend()
        # append to dict of final selections
        selections_to_save[well] = {
            'well':                 well,
            'sample':               df_normalised.loc[df_normalised['well']==well, 'content'].tolist()[0], # returns a string
            'exponential_points_bounds': (df_normalised.loc[df_normalised['time_h']==exponential_times[0], 'time'].tolist()[0], df_normalised.loc[df_normalised['time_h']==exponential_times[-1], 'time'].tolist()[0]), # contains time point bounds for selected exponential (in mins)
            'exponential_fluorescence': exponential_fluorescence,
            'exponential_fluorescence_err': exponential_fluo_err,
            'hanning_window':       live_params['smoothing_window'],
            'hampel_window':        live_params['outlier_window'],
            'hampel_sigmas':        live_params['outlier_sigmas']
        }


    if len(selection_buffer) == 1: # draw x=current selection
        axs[0].axvline(x=times[closest_point_id], color='red', linestyle='--')
        axs[1].axvline(x=times[closest_point_id], color='red', linestyle='--')
        axs[2].axvline(x=times[closest_point_id], color='red', linestyle='--')

    plt.draw()


               
# Button handlers for user decision before moving on to next well

def on_keep_click(event, fig, well, df_od, selections_to_save, selection_buffer, path_to_save, df_normalised):
    os.makedirs(os.path.join(path_to_save, "saved plots"), exist_ok = True)
    if len(selection_buffer) < 2:
        print(f"No selection made for well {well}. Saving as NaN.")
        selections_to_save[well] = empty_result(well, df_od)
        selection_buffer.clear()
        plt.savefig(os.path.join(path_to_save, f"saved plots", f"well {well}_({df_normalised.loc[df_normalised['well']==well, 'content'].tolist()[0]})_discarded.jpg"))
    else:
        print(f"Confirmed selection for well {well}.")
        plt.savefig(os.path.join(path_to_save, f"saved plots", f"well {well}_({df_normalised.loc[df_normalised['well']==well, 'content'].tolist()[0]}).jpg"))
    global proceed_flag
    proceed_flag = True
    plt.close(fig)

def on_skip_click(event, fig, well, df_od, selections_to_save, selection_buffer, path_to_save, df_normalised):
    print(f"Skipping well {well}.")
    selections_to_save[well] = empty_result(well, df_od)
    selection_buffer.clear()
    os.makedirs(os.path.join(path_to_save, "saved plots"), exist_ok = True)
    plt.savefig(os.path.join(path_to_save, f"saved plots", f"well {well}_({df_normalised.loc[df_normalised['well']==well, 'content'].tolist()[0]})_discarded.jpg"))
    global proceed_flag
    proceed_flag = True
    plt.close(fig)


def on_exit_click(event, fig, well, df_od, selections_to_save, selection_buffer, exit_requested):
    print("Exit button clicked. Terminating script...")
    # clear last well since it was never confirmed, so should not be saved
    selections_to_save[well] = empty_result(well, df_od)
    selection_buffer.clear()
    exit_requested['value'] = True
    global proceed_flag
    proceed_flag = True #unblock while in main
    plt.close(fig)



# Define function that will take the data for current well and allow user to interact with it and get the growth rate
def process_well_interactive(well, df_od, df_fluorescence, live_params, selections_to_save, selection_buffer, exit_requested, path_to_save):
    # normalise data for current well:
    df_normalised = normalise(well, df_fluorescence, df_od, live_params)

    # Create figure only once for current well
    fig, axs = plt.subplots(3, figsize=(6, 9))
    fig.subplots_adjust(top=0.88, hspace=0.45)
    def recompute(): # re-normalise when input parameters change
        nonlocal df_normalised
        df_normalised = normalise(well, df_fluorescence, df_od, live_params)
        if df_normalised.empty:
            print(f"There's no data for well {well} with selected parameters.")
            result = empty_result(well, df_od)
            return result
        axs[0].clear(); axs[1].clear(); axs[2].clear() 
        replot_all(axs, df_od, df_normalised, well, live_params)

        fig.canvas.draw_idle()

    global recompute_hook 
    recompute_hook = recompute # make this function visible to on point click with current params
    recompute() # run

    # Click handler reads from state, which recompute() keeps up to date
    fig.canvas.mpl_connect(
        'button_press_event',
        lambda event: (
            on_point_click(event, axs, well, df_od, df_normalised, live_params, selection_buffer, selections_to_save)
        ),
    )

    # User decision buttons. Keep, discard or abort program -> change coordinates to make it look ok!!!
    button_ax_keep = plt.axes([0.15, 0.93, 0.2, 0.04])
    button_ax_skip = plt.axes([0.40, 0.93, 0.2, 0.04])
    button_ax_exit = plt.axes([0.65, 0.93, 0.2, 0.04])
    confirm_button = MplButton(button_ax_keep, 'keep', color='springgreen', hovercolor='thistle')
    discard_button = MplButton(button_ax_skip, 'skip', color='tomato', hovercolor='thistle')
    exit_button = MplButton(button_ax_exit, 'exit', color='firebrick', hovercolor='thistle')

    global proceed_flag
    proceed_flag = False
    confirm_button.on_clicked(lambda event: on_keep_click(event, fig, well, df_od, selections_to_save, selection_buffer, path_to_save, df_normalised))
    discard_button.on_clicked(lambda event: on_skip_click(event, fig, well, df_od, selections_to_save, selection_buffer, path_to_save, df_normalised))
    exit_button.on_clicked(lambda event: on_exit_click(event, fig, well, df_od, selections_to_save, selection_buffer, exit_requested))

    plt.show(block=True)

    # Wait for user to decide what to do next
    while not proceed_flag:
        plt.pause(0.1)
    return
    

