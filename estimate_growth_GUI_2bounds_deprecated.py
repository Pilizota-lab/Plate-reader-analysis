# final working GUI for esimating growth rates from plate reader data
# to be run from the terminal
#%% 
# import modules
# Diana's modules
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
import math
import os
import sys
# Jimi's extra modules
from matplotlib.widgets import Button as MplButton
from scipy.signal import medfilt

# TODO parameters below should have a default, but let the user change them and see what works best for their condition
# processing parameters - these are the parameters that will be used for estimating growth rates (users can adjust if needed)
# growth rate is only processed and plotted between these OD's
greater_than = np.log(0.005)
less_than = np.log(0.6)
# time point boundaries if experiment ran for too long
xcut = 0
xcut_upper = 35
# windows and filters - DEFAULTS AS BELOW, will need to be saved to the output file
fitting_window = 5  # running window size # TODO make adjustable for each well?
hampel_sigma = 2  # number of stdevs outside of which the point is excluded
hampel_window = 9
hanning_window = 1 # smoothing with a hanning window (if not needed, set window to 1)


# --- default params ---
default_params = {
    "xcut": 0.0,
    "xcut_upper": 35.0,
    "fitting_window": 5,
    "hampel_sigma": 2,
    "hampel_window": 9,  
    "hanning_window": 1, # >=1
}
# --- live params ---
params = {
    "xcut": 0.0,
    "xcut_upper": 35.0,
    "fitting_window": 5,
    "hampel_sigma": 2,
    "hampel_window": 9,  
    "hanning_window": 1, # >=1
}
# This will point to "recompute current well" function
recompute_hook = None




# start program
path = fd.askopenfilename() # user selects file from file explorer GUI

# 1. define dataframe with raw data
data = pd.read_csv(path, skiprows = 7).dropna()
data_tr = data.transpose()
print(data)
time_pts = [float(i) for i in data.columns[2:]]
time_hours = [time/60 for time in time_pts]
time_increment = time_hours[1]-time_hours[0]
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
df

# 2. blank data
#select blank region by creating an interactive graph
#no parameter input needed here
master = Tk()
master.title("Select region which you want to use as blank")

def plot_for_blanks():
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
            global xmin
            xmin = self.xinit
            
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
            global xmax
            xmax = self.xfin
            self.rect = patches.Rectangle((self.xinit, self.yinit), width, height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(self.rect)
            fig.canvas.draw_idle()

    def confirmation():
        plt.close()
        master.quit()
        master.destroy()
        exit
        return()
            

    plot_blnk = SelectBlanks()
    fig.canvas.mpl_connect("button_press_event", plot_blnk.on_press)
    fig.canvas.mpl_connect("motion_notify_event", plot_blnk.on_motion)
    fig.canvas.mpl_connect("button_release_event", plot_blnk.on_release)
    Button(master, text="confirm selection", command = confirmation).grid(row = 1)
    plt.show()
    return()

Button(master, text="select blanks", command = plot_for_blanks).grid(row=0)
# if window is closed manually by user, move on 
def on_closing():
    master.quit()
    master.destroy()
master.protocol("WM_DELETE_WINDOW", on_closing)
master.mainloop()

print (xmin, xmax)

#blank data based on users choice of "lag" phase
#no extra parameters (apart from coordinates from user input)
df_blanked = pd.DataFrame(columns=heads)
for wl in range(len(wells)):
    well = wells[wl]
    #calculate average of points used for blanking
    df_with_blanks = df.query("well == @well and time_h>@xmin and time_h<@xmax")
    blank = np.average(df_with_blanks["OD"].to_list())
    df_to_blank_data = df.query("well == @well")
    list_ods = df_to_blank_data["OD"].to_list()
    list_ods_blanked = [a-blank for a in list_ods]
    list_ods_blanked_ln = [np.log(a) for a in list_ods_blanked]
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
df_blanked #this is cointains the blank data to work with from now on


# 3. define filters

# Hampel filter function with a running window
def hampel_filter(data, window_size, n_sigmas):
    n = len(data)
    filtered_data = data.copy()
    half_window = window_size // 2
    for i in range(n):
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        window = data[start:end]
        median = np.median(window)
        mad = np.median(np.abs(window - median))
        scale = 1
        threshold = n_sigmas * scale * mad
        if abs(data[i] - median) > threshold:
            filtered_data[i] = median
    return filtered_data

def hanning_smoothing(data, window_size):
    # data to be input as list or values
    # make sure window size is odd and >=1
    if window_size>1:
        # create weights for window
        window = np.hanning(window_size)
        smooth_data = np.convolve(data, window, mode='same')/sum(window)
        return smooth_data
    else:
        return data


# 4a create window with parameters - include reset and confirm buttons
def open_params_window():
    win = Tk()
    win.title("Processing parameters")

    # simple entries bound to StringVars
    vars_ = {k: StringVar(value=str(v)) for k, v in params.items()}

    def row(r, label, key):
        ttk.Label(win, text=label).grid(row=r, column=0, sticky="w", padx=6, pady=4)
        e = ttk.Entry(win, textvariable=vars_[key], width=10)
        e.grid(row=r, column=1, sticky="w", padx=6, pady=4)
        e.bind("<Return>", lambda _=None: do_apply())

    row(0, "xcut (h)",            "xcut")
    row(1, "xcut_upper (h)",      "xcut_upper")
    row(2, "fitting_window (pts)","fitting_window")
    row(3, "hampel_sigma",        "hampel_sigma")
    row(4, "hampel_window (odd)", "hampel_window")
    row(5, "hanning_window",      "hanning_window")

    def do_apply():
        # update global params from the entries (minimal validation)
        try:
            params["xcut"]           = float(vars_["xcut"].get())
            params["xcut_upper"]     = float(vars_["xcut_upper"].get())
            params["fitting_window"] = max(2, int(vars_["fitting_window"].get()))
            params["hampel_sigma"]   = float(vars_["hampel_sigma"].get())
            hw = max(1, int(vars_["hampel_window"].get()))
            if hw % 2 == 0: hw += 1
            params["hampel_window"]  = hw
            params["hanning_window"] = max(1, int(vars_["hanning_window"].get()))
        except ValueError:
            print("Invalid parameter; please enter numbers.")
            return

        # call current recompute hook if set
        global recompute_hook
        if recompute_hook:
            recompute_hook()

    ttk.Button(win, text="Apply to current well", command=do_apply)\
        .grid(row=6, column=0, columnspan=2, padx=6, pady=8, sticky="ew")
    ttk.Button(win, text="Apply to current well", command=do_apply)\
        .grid(row=6, column=0, columnspan=2, padx=6, pady=(8, 4), sticky="ew")

    def do_reset():
        # reset entry fields to defaults
        for k, v in default_params.items():
            vars_[k].set(str(v))
        # also update the live params + replot
        do_apply()

    ttk.Button(win, text="Reset to defaults", command=do_reset)\
        .grid(row=7, column=0, columnspan=2, padx=6, pady=(0, 8), sticky="ew")



    # don’t block your script; keep this window alive
    win.protocol("WM_DELETE_WINDOW", win.destroy)
    return win

param_win = open_params_window()

# 4. define interactive functions

# Dictionary to store selections
selections = {}            # Final stored selection per well: {'time': (t1,t2) or float, 'growth_rate': float, 'indices': (i1,i2)}
selections_multi = {}      # Temporary click buffer per well: list of selected gr-indices
confirmed_wells = set()    # Track confirmed wells
selected_fit_handles = {}  # Store handles for selected fit line and points per well

# "Click" handler for selecting growth rate points (now supports two-point selection)
def on_point_click(event, time_data, growth_rates, ax1, ax2, well, query_df, fitting_window_local, filtered_df):
    if event.inaxes != ax2:
        return
    x_click, y_click = event.xdata, event.ydata
    if x_click is None or y_click is None:
        return

    # Find the closest valid point (ignore NaNs)
    valid_mask = ~np.isnan(growth_rates)
    if not np.any(valid_mask):
        return
    valid_time = time_data[valid_mask]
    valid_gr = growth_rates[valid_mask]
    distances = np.sqrt((valid_time - x_click)**2 + (valid_gr - y_click)**2)
    closest_on_valid = np.argmin(distances)
    selected_idx = np.where(valid_mask)[0][closest_on_valid]  # index into growth_rates / time_data

    # Initialize buffer for this well if needed
    buf = selections_multi.get(well, [])
    # If buffer already has two selections, start over with the new first selection
    if len(buf) >= 2:
        buf = []
    buf.append(selected_idx)
    selections_multi[well] = buf

    # Update second subplot (ax2) visual: redraw points and selected markers/lines
    ax2.clear()
    ax2.scatter(time_data, growth_rates, picker=5, label='Growth rates', alpha=0.6)
    # If one point selected: show it
    if len(buf) == 1:
        idx1 = buf[0]
        t1 = time_data[idx1]
        gr1 = growth_rates[idx1]
        ax2.scatter([t1], [gr1], color='red', s=100, marker='x', label='Selected point 1')
        ax2.axvline(x=t1, color='green', linestyle='--', alpha=0.5, label='Selected time 1')
        ax2.text(0.05, 0.95, f"Selected 1: Time = {t1:.2f} h, GR = {gr1:.3f}/h",
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    # If two points selected: compute average of growth rates for the two selected points (each estimated with fitting_window_local)
    elif len(buf) == 2:
        i1, i2 = buf[0], buf[1]
        start_idx, end_idx = min(i1, i2), max(i1, i2)
        t_start, t_end = time_data[start_idx], time_data[end_idx]

        # Remove previous selected fit if it exists
        if well in selected_fit_handles:
            for handle in selected_fit_handles[well]:
                try:
                    handle.remove()
                except Exception:
                    pass

        slopes = []
        handles = []
        # For each index from start_idx to end_idx inclusive, compute the slope using the fitting window starting at that index
        for sel_idx in range(start_idx, end_idx + 1):
            color = 'purple' if sel_idx == start_idx else 'magenta' if sel_idx == end_idx else 'blue'  # Differentiate start/end, others blue
            label_suffix = f'idx{sel_idx}'
            win_start = sel_idx
            win_end = sel_idx + fitting_window_local
            if win_end <= len(query_df):
                window_df = query_df.iloc[win_start:win_end]
                x_win = window_df['time_h'].values
                y_win = window_df['lnOD_filtered'].values
                # only compute fit if we have at least 2 points and no nan that prevents sizing
                if len(x_win) >= 2 and not np.any(np.isnan(y_win)):
                    slope, intercept = np.polyfit(x_win, y_win, 1)
                    slopes.append(slope)
                    # plot points and fit line on ax1
                    p = ax1.scatter(x_win, y_win, color=color, s=20, label=f'Selected window {label_suffix}')
                    line_x = np.array([x_win.min(), x_win.max()])
                    line_y = slope * line_x + intercept
                    l, = ax1.plot(line_x, line_y, color=color, linestyle='-', label=f'Fit {label_suffix}')
                    handles.extend([p, l])
                else:
                    slopes.append(np.nan)
            else:
                # If window would exceed available rows, append NaN
                slopes.append(np.nan)

        # Average all the slopes (ignoring nan)
        avg_gr = np.nanmean(slopes) if len(slopes) > 0 else np.nan

        # Save final selection entry for this well
        selections[well] = {
            'time': (float(t_start), float(t_end)),
            'growth_rate': float(avg_gr) if not np.isnan(avg_gr) else np.nan,
            'indices': (int(start_idx), int(end_idx))
        }

        # Visual markers on ax2 for the two selected points
        ax2.scatter([time_data[i1], time_data[i2]], [growth_rates[i1], growth_rates[i2]],
                    color='red', s=100, marker='x', label='Selected points')
        ax2.axvline(x=time_data[i1], color='green', linestyle='--', alpha=0.5, label='Start time')
        ax2.axvline(x=time_data[i2], color='green', linestyle='--', alpha=0.5, label='End time')

        ax2.text(0.05, 0.95, f"Points: {time_data[i1]:.2f}, {time_data[i2]:.2f} h\nAvg GR = {avg_gr:.4f}/h",
                 transform=ax2.transAxes, fontsize=9, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

        # Save handles for later removal
        selected_fit_handles[well] = handles

    # Redraw ax2 labels and legend
    ax2.set_xlabel("Time (h)")
    ax2.set_ylabel("Growth rate (1/h)")
    ax2.set_title(f"Rolling growth rate for well {well} (Click twice to select range, then Confirm or Discard)")
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        unique = dict(zip(labels, handles))
        ax2.legend(unique.values(), unique.keys())
    plt.draw()

# "Confirm" button handler
def on_confirm_click(event, fig, well, proceed_flags):
    print(f"Confirmed selection for well {well}. Moving to next well...")
    # If user confirmed but only single-point buffer present, convert to single selection (previous behaviour)
    buf = selections_multi.get(well, [])
    if well not in selections:
        if len(buf) == 1:
            idx = buf[0]
            # time_data and growth_rates not available here; rely on query created in process_well_interactive
            # Fallback: mark selection as single index and leave growth_rate NaN — process_well_interactive will handle if needed
            selections[well] = {'time': (idx, idx), 'growth_rate': np.nan, 'indices': (idx, idx)}
    confirmed_wells.add(well)
    proceed_flags[well] = True
    plt.close(fig)

# "Discard" button handler
def on_discard_click(event, fig, well, proceed_flags):
    print(f"Discarded selection for well {well}. Setting growth rate to NaN and moving to next well...")
    selections[well] = {'time': (np.nan, np.nan), 'growth_rate': np.nan, 'indices': (None, None)}
    confirmed_wells.add(well)
    proceed_flags[well] = True
    plt.close(fig)

# "Exit" button handler
def on_exit_click(event, fig):
    print("Exit button clicked. Terminating script...")
    plt.close(fig)
    raise SystemExit("Script terminated by user")


# 5. iterate through every well and process data: apply filters, get growth rate with running window and let user select what is exponential

# Function to process a single well interactively
def process_well_interactive(well, df):
    
    """
    Same behaviour as your original function, but reads parameters from the global `params`
    and exposes a `recompute()` that the parameter window can call via `recompute_hook`.
    """
    global recompute_hook

    # For click handler + result, we always use these latest arrays/dfs
    latest = {"time_data": None, "growth_rates": None, "filtered_df": None, "query_df": None}

    # Create figure once
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    fig.subplots_adjust(bottom=0.8)

    def recompute():
        # read live parameters
        x0   = params["xcut"]
        x1   = params["xcut_upper"]
        fw   = params["fitting_window"]
        hs   = params["hampel_sigma"]
        hw   = params["hampel_window"]
        hanw = params["hanning_window"]

        # 1) time slice
        df_well_raw = df[df['well'] == well].copy()
        df_well_raw = df_well_raw[(df_well_raw['time_h'] > x0) & (df_well_raw['time_h'] < x1)]
        if df_well_raw.empty:
            print(f"No data for well {well} after time filtering")
            return

        # 2) filters
        filtered_df = df_well_raw.copy()
        filtered_df['lnOD_filtered'] = hampel_filter(
            filtered_df['lnOD'].values, window_size=hw, n_sigmas=hs
        )
        filtered_df['lnOD_filtered'] = hanning_smoothing(
            filtered_df['lnOD_filtered'].values, window_size=hanw
        )

        # 3) OD limits
        query_df = filtered_df.query(
            'lnOD_filtered > @greater_than and lnOD_filtered < @less_than and lnOD_filtered != 0'
        )
        if query_df.empty:
            print(f"No data for well {well} after OD thresholding")
            return

        # 4) rolling slopes with current fw
        gr = []
        for i in range(len(query_df) - (fw - 1)):
            w = query_df.iloc[i:i+fw]
            slope, intercept = np.polyfit(w['time_h'].values, w['lnOD_filtered'].values, 1)
            gr.append(slope)
        gr_plot = [g if g >= 0.01 else np.nan for g in gr]
        time_data = query_df.time_h.iloc[:len(gr_plot)].values
        growth_rates_data = np.array(gr_plot)

        # 5) redraw (simple: clear + scatter)
        axs[0].clear(); axs[1].clear()
        axs[0].scatter(df_well_raw.time_h, df_well_raw.lnOD, label="lnOD (raw)", s=10, color='tab:blue', alpha=0.5)
        axs[0].scatter(filtered_df.time_h, filtered_df.lnOD_filtered, label="lnOD (filtered)", s=10, color='tab:orange')
        axs[0].axhline(greater_than, color='red', linestyle='--', alpha=0.7, label=f'OD > {np.exp(greater_than)}')
        axs[0].axhline(less_than,   color='blue', linestyle='--', alpha=0.7, label=f'OD < {np.exp(less_than)}')
        axs[0].set_xlabel("Time (h)"); axs[0].set_ylabel("lnOD"); axs[0].set_title(f"lnOD for well {well}")
        axs[0].legend(); axs[0].set_ylim(-5.5,0)

        axs[1].scatter(time_data, growth_rates_data, picker=5, label='Growth rates', alpha=0.6)
        axs[1].set_xlabel("Time (h)"); axs[1].set_ylabel("Growth rate (1/h)")
        axs[1].set_title(f"Rolling growth rate for well {well} (fw={fw}, Hampel={hw}/{hs})")
        axs[1].legend(); axs[1].set_ylim(0, 2)

        fig.canvas.draw_idle()

        # 6) update for click handler and result
        latest.update(time_data=time_data, growth_rates=growth_rates_data,
                      filtered_df=filtered_df, query_df=query_df)

        # (optional) clear temporary selection overlays when params change
        selections_multi[well] = []
        if well in selected_fit_handles:
            for h in selected_fit_handles[well]:
                try: h.remove()
                except Exception: pass
            selected_fit_handles.pop(well, None)

    # expose recompute to the param window
    recompute_hook = recompute

    # first draw
    recompute()

    # connect click handler – always uses the latest arrays and current fw
    fig.canvas.mpl_connect(
        'button_press_event',
        lambda event: (latest["time_data"] is not None) and on_point_click(
            event, latest["time_data"], latest["growth_rates"],
            axs[0], axs[1], well, latest["query_df"],
            params["fitting_window"], latest["filtered_df"]
        )
    )

    # Add your existing Confirm/Discard/Exit buttons (unchanged)
    button_ax_confirm = plt.axes([0.65, 0.02, 0.1, 0.075])
    confirm_button = MplButton(button_ax_confirm, 'Confirm', color='lightgreen', hovercolor='lightcoral')
    button_ax_discard = plt.axes([0.775, 0.02, 0.1, 0.075])
    discard_button = MplButton(button_ax_discard, 'Discard', color='lightyellow', hovercolor='lightblue')
    button_ax_exit = plt.axes([0.9, 0.02, 0.1, 0.075])
    exit_button = MplButton(button_ax_exit, 'Exit', color='lightcoral', hovercolor='lightblue')

    proceed_flags = {well: False}
    confirm_button.on_clicked(lambda event: on_confirm_click(event, fig, well, proceed_flags))
    discard_button.on_clicked(lambda event: on_discard_click(event, fig, well, proceed_flags))
    exit_button.on_clicked(lambda event: on_exit_click(event, fig))

    plt.tight_layout()
    plt.show(block=True)

    # Wait for confirm/discard/exit (same as before)
    while not proceed_flags.get(well, False):
        plt.pause(0.1)

    # --- BUILD AND RETURN RESULT (this is your missing return) ---
    gr_val = selections[well]['growth_rate'] if well in selections else np.nan
    result = {
        "well": well,
        "medium": (latest["filtered_df"]['content'].iloc[0]
                   if (latest["filtered_df"] is not None and 'content' in latest["filtered_df"].columns)
                   else None),
        "selected_time_h": selections[well]['time'] if well in selections else (np.nan, np.nan),
        "growth_rate_1ph": gr_val if (gr_val is not None) else np.nan,
        "doubling_time_min": (np.log(2) / gr_val * 60) if (gr_val and gr_val > 0) else np.nan,
        "hanning": params['hanning_window'],
        "fitting window": params['fitting_window'],
        "hampel_size": params['hampel_window'],
        "hampel_sigmas": params['hampel_sigma']
    }
    return result


    # old version:
    # if well in confirmed_wells: # not sure how we'd get to this scenario so might not be needed? 
    #     return None  # Skip if already confirmed
    
    # # Filter data (clean, stage-named dataframes so you can easily comment out a stage)
    # # filtered_df will contain filtered data for current well that will be plotted on the RHS straight away
    # global filtered_df  # used by on_point_click for plotting OD

    # # Raw per-well slice and time window
    # df_well_raw = df[df['well'] == well].copy() # raw (but blanked)
    # df_well_raw = df_well_raw[(df_well_raw['time_h'] > xcut) & (df_well_raw['time_h'] < xcut_upper)]  # only work with the data that is between the beginning and 35h 
    # if df_well_raw.empty: # error handling
    #     print(f"No data for well {well} after time filtering")
    #     return None

    # filtered_df = df_well_raw.copy()
    # # will apply filters on the lnOD values
    # filtered_df['lnOD_filtered'] = filtered_df['lnOD'].values

    # # Filter 1 - Hampel spike removal (optional)
    # filtered_df['lnOD_filtered'] = hampel_filter(
    #     filtered_df['lnOD_filtered'].values,
    #     window_size=hampel_window,
    #     n_sigmas=hampel_sigma
    # )

    # # # Filter 2 - Smoothed-std filtering to drop high-variance regions (optional)
    # # filtered_df['std_filtering'] = filtered_df['lnOD_filtered'].rolling(5, center=True).std()
    # # filtered_df = filtered_df[filtered_df['std_filtering'] < 0.25].copy() # only keeps points within noise bounds

    # # Filter 3 - Keep only non-decreasing OD points (optional)
    # # filtered_df = filtered_df[filtered_df['OD_filtered'].values <= filtered_df['OD_filtered'].shift(-1).values].copy()

    # # Smoothing (optional) - use hanning window (make as small as possible)
    # filtered_df['lnOD_filtered'] = hanning_smoothing(
    #     filtered_df['lnOD_filtered'].values,
    #     window_size= hanning_window
    # )

    # # Add Log-transform - THIS SHOULD BE DONE BEFORE FILTERS
    # # filtered_df['lnOD_filtered'] = np.log(filtered_df['OD_filtered'])

    # # Create query_df which is used for fitting and apply hard limits on OD
    # query_df = filtered_df.query('lnOD_filtered > @greater_than and lnOD_filtered < @less_than and lnOD_filtered != 0')
    # if query_df.empty:
    #     print(f"No data for well {well} after OD thresholding")
    #     return None
   
    # gr = []
    # for x in range(len(query_df) - (fitting_window_local - 1)):
    #     window_data = query_df.iloc[x:x + fitting_window_local]
    #     x_temp = window_data['time_h'].values
    #     y_temp = window_data['lnOD_filtered'].values
    #     slope, intercept = np.polyfit(x_temp, y_temp, 1)
    #     gr.append(slope)
    
    # if not gr:
    #     print(f"No growth rates calculated for well {well}")
    #     return None
       
    # # Create figure with space for buttons
    # fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # fig.subplots_adjust(bottom=0.8)  # Increased space for buttons
    
    # # Subplot 1: lnOD (no initial max growth fit)
    # axs[0].scatter(df_well_raw.time_h, df_well_raw.lnOD, 
    #                label="lnOD (raw)", s=10, color='tab:blue', alpha=0.5)
    # axs[0].scatter(filtered_df.time_h, filtered_df.lnOD_filtered, 
    #                label="lnOD (filtered)", s=10, color='tab:orange')
    # axs[0].axhline(greater_than, color='red', linestyle='--', 
    #                alpha=0.7, label=f'OD > {np.exp(greater_than)}')
    # axs[0].axhline(less_than, color='blue', linestyle='--', 
    #                alpha=0.7, label=f'OD < {np.exp(less_than)}')
    # axs[0].set_xlabel("Time (h)")
    # axs[0].set_ylabel("lnOD")
    # axs[0].set_title(f"lnOD for well {well}")
    # axs[0].legend()
    # axs[0].set_ylim(-5.5,0)
    
    # # Subplot 2: Rolling growth rate
    # gr_plot = [g if g >= 0.01 else np.nan for g in gr]
    # time_data = query_df.time_h.iloc[:len(gr)].values
    # growth_rates_data = np.array(gr_plot)
    # scat = axs[1].scatter(time_data, growth_rates_data, picker=5, label='Growth rates', alpha=0.6)
    # axs[1].set_xlabel("Time (h)")
    # axs[1].set_ylabel("Growth rate (1/h)")
    # axs[1].set_title(f"Rolling growth rate for well {well} (Click twice to select range, then Confirm or Discard)")
    # axs[1].legend()
    # axs[1].set_ylim(bottom=0)  # Ensure y-axis starts at 0
    
    # # Reset any temporary selection buffer for this well
    # selections_multi[well] = []
    # # Connect point click event
    # fig.canvas.mpl_connect('button_press_event', 
    #                        lambda event: on_point_click(event, time_data, growth_rates_data, axs[0], axs[1], well, query_df, fitting_window_local, filtered_df))
    
    # # Add Confirm, Discard, and Exit buttons (lowered)
    # button_ax_confirm = plt.axes([0.65, 0.02, 0.1, 0.075])
    # confirm_button = MplButton(button_ax_confirm, 'Confirm', color='lightgreen', hovercolor='lightcoral')
    # button_ax_discard = plt.axes([0.775, 0.02, 0.1, 0.075])
    # discard_button = MplButton(button_ax_discard, 'Discard', color='lightyellow', hovercolor='lightblue')
    # button_ax_exit = plt.axes([0.9, 0.02, 0.1, 0.075])
    # exit_button = MplButton(button_ax_exit, 'Exit', color='lightcoral', hovercolor='lightblue')
    
    # # Proceed flag
    # proceed_flags = {well: False}
    
    # # Connect button callbacks
    # confirm_button.on_clicked(lambda event: on_confirm_click(event, fig, well, proceed_flags))
    # discard_button.on_clicked(lambda event: on_discard_click(event, fig, well, proceed_flags))
    # exit_button.on_clicked(lambda event: on_exit_click(event, fig))
    
    # plt.tight_layout()
    # plt.show(block=True)
    
    # # Wait for confirm, discard, or exit
    # while not proceed_flags.get(well, False):
    #     plt.pause(0.1)
    
    # result = {
    #     "well": well,
    #     "medium": filtered_df['content'].iloc[0] if 'content' in filtered_df.columns else None,
    #     "selected_time_h": selections[well]['time'],
    #     "growth_rate_1ph": selections[well]['growth_rate'],
    #     "doubling_time_min": np.log(2) / selections[well]['growth_rate'] * 60 if selections[well]['growth_rate'] and selections[well]['growth_rate'] > 0 else np.nan
    # }
    # return result

# main program execution (integrating all interactive functions defined above)

growth_rates = {}
results = []

try:
    unique_wells = df_blanked['well'].unique()
    print(f"Processing {len(unique_wells)} unique wells one by one...")
    
    for well in unique_wells:
        print(f"\n--- Processing well {well} ---")
        result = process_well_interactive(well, df_blanked)
        if result is not None:
            results.append(result)
            growth_rates[well] = round(result["growth_rate_1ph"], 2)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    print("\nFinal Results Table:")
    print(results_df)
    
except NameError as e:
    print(f"Error: DataFrame 'df_blanked' not defined or missing columns. Please ensure 'df_blanked' exists with 'well', 'time_h', 'OD' columns. Details: {e}")
except SystemExit as e:
    print(e)
    # Save partial results before exiting
    if results:
        results_df = pd.DataFrame(results)
        print("\nPartial Results Table (before exit):")
except Exception as e:
    print(f"An error occurred: {e}")

# Clean up
plt.close('all')


# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)
print(results_df)

# Get stats per medium
avg_by_medium = (
    results_df
    .groupby('medium', as_index=False)
    .agg(
        mean_growth_rate=('growth_rate_1ph', 'mean'),
        std_growth_rate=('growth_rate_1ph', 'std'),
        n=('growth_rate_1ph', 'count'),
        mean_doubling_min=('doubling_time_min', 'mean')
    )
    .sort_values('mean_growth_rate', ascending=False)
)

avg_by_medium

# save results to CSV (same path/file name + growth_rates at the end) - if user wants to do so
saving_popup = Tk()
saving_popup.title('Do you want to write results to disk?')
def save_results():
    results_df.to_csv((path[:-4] + '_growth_rates.csv'), index=False)
    saving_popup.destroy()
    print('data saved to disk next to the raw data.')
    return()

def move_on():
    saving_popup.destroy()

Button(saving_popup, text='Save', command = save_results).grid(row=1, column=1)
Button(saving_popup, text='Forget', command=move_on).grid(row=1, column=2)

saving_popup.protocol("WM_DELETE_WINDOW", move_on) #if user closes window instead
saving_popup.mainloop()


# TODO compute STDEV of individual fits (which can then be propagated into ste over replicates)
# kill kernel at the end to reset values and avoid GUI functions doing funny stuff
sys.exit(0)

# %%
