"""
Plate Reader Analysis Tool
-------------------------
This application provides a graphical interface for analyzing plate reader data.
It supports single and multiple file analysis, data blanking, and various visualization options.

Key Features:
- Interactive data visualization
- Well-based and condition-based analysis
- Support for multiple datasets
- Customizable plot options and colors
- Data blanking capabilities

Author: [Your Name]
Version: 4.0
Last Updated: [Date]
"""

# ===============================
# IMPORTS AND SETUP
# ===============================
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
import os
import sys
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.colors as mcolors
from tkinter import colorchooser
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ===============================
# MODULE-LEVEL VARIABLES
# ===============================
"""Global variables used throughout the application"""
df = None
df_list = []
file_paths = []
tabControl = None
left_frame = None
right_frame = None
well_vars = []
well_log = None
edit_title = None
to_plot = None
indiv = None
ci_95 = None
cond_log = None
edit_title_contents = None
color_vars = {}
initial_window = None
xmin = None
xmax = None
master = None
all_conditions = []

# ===============================
# DATA PROCESSING FUNCTIONS
# ===============================
"""Core functions for data handling and processing"""

def read_and_process_file(path):
    """
    Reads and processes a CSV file containing plate reader data.
    
    Args:
        path (str): Path to the CSV file
        
    Returns:
        pandas.DataFrame: Processed data with standardized columns
    """
    data = pd.read_csv(path, skiprows=7)
    data_tr = data.transpose()
    time_pts = [float(i) for i in data.columns[2:]]
    time_hours = [time/60 for time in time_pts]
    wells = list(data_tr.iloc[0])
    content = list(data_tr.iloc[1])
    heads = ["content", "well", "time", "time_h", "OD", "lnOD", "dataset"]
    temp_df = pd.DataFrame(columns=heads)
    
    dataset_name = os.path.splitext(os.path.basename(path))[0]
    
    for wl in range(len(wells)):
        cont_list = []
        well_list = []
        dataset_list = []
        for i in range(len(time_pts)):
            cont_list.append(content[wl])
            well_list.append(wells[wl])
            dataset_list.append(dataset_name)
        to_add = pd.DataFrame({
            "content": cont_list,
            "well": well_list,
            "time": time_pts,
            "time_h": time_hours,
            "OD": list(data_tr[wl])[2:],
            "lnOD": [np.log(a) for a in list(data_tr[wl])[2:]],
            "dataset": dataset_list
        }, columns=heads)
        
        temp_df = pd.concat([temp_df, to_add], ignore_index=True, sort=False)
    return temp_df

# ===============================
# UI COMPONENT FUNCTIONS
# ===============================
"""Functions for creating and managing UI elements"""

def add_title_controls(fig, ax, default_title):
    """
    Creates a window for editing graph titles.
    
    Args:
        fig (matplotlib.figure.Figure): The figure object
        ax (matplotlib.axes.Axes): The axes object
        default_title (str): Default title for the plot
    
    Returns:
        tk.Toplevel: Window containing title controls
    """
    title_window = Toplevel()
    title_window.title("Edit Graph Title")
    title_window.geometry("400x100")
    
    title_var = StringVar(value=default_title)
    title_entry = Entry(title_window, textvariable=title_var, width=40)
    title_entry.pack(pady=10, padx=10)
    
    def update_title():
        new_title = title_var.get()
        ax.set_title(new_title)
        fig.canvas.draw_idle()
    
    update_button = Button(title_window, text="Update Title", command=update_title)
    update_button.pack(pady=5)
    
    try:
        fig_x = fig.canvas.manager.window.winfo_x()
        fig_y = fig.canvas.manager.window.winfo_y()
        title_window.geometry(f"+{fig_x + fig.canvas.get_width_height()[0] + 10}+{fig_y}")
    except:
        pass
    
    return title_window

def create_file_header(parent):
    """
    Creates a header showing current file information.
    
    Args:
        parent (tk.Widget): Parent widget to contain the header
        
    Returns:
        tk.Frame: Frame containing the file header
    """
    header_frame = Frame(parent, bg='white', relief='ridge', bd=1)
    header_frame.pack(fill=X, padx=10, pady=(10,5))
    
    files_text = "Current file: " + os.path.basename(file_paths[0]) if len(df_list) == 1 else \
                "Current files: " + ", ".join(os.path.basename(f) for f in file_paths)
    
    Label(header_frame, 
          text=files_text,
          font=('Arial', 10, 'italic'),
          bg='white').pack(pady=5)
    return header_frame

def style_button(button):
    """Applies consistent styling to buttons"""
    button.configure(
        bg='#2196F3',
        fg='white',
        font=('Arial', 9),
        relief='flat',
        padx=10,
        pady=5,
        cursor='hand2',
        activebackground='#1976D2',
        activeforeground='white'
    )

def create_movement_handler():
    """Creates a handler for plot panning functionality"""
    class MovementHandler:
        def __init__(self):
            self.xpress = None
            self.ypress = None
            self.keep_running = True
        
        def on_press(self, event):
            self.xpress, self.ypress = event.xdata, event.ydata
            self.keep_running = True
        
        def on_motion(self, event):
            if event.inaxes and self.keep_running:
                dx = event.xdata - self.xpress
                dy = event.ydata - self.ypress
                ax = event.inaxes
                l0, r0 = ax.get_xlim()
                ax.set_xlim(l0 - dx, r0 - dx)
                l0, r0 = ax.get_ylim()
                ax.set_ylim(l0 - dy, r0 - dy)
                ax.figure.canvas.draw_idle()

        def on_release(self, event):
            self.keep_running = False
            
    return MovementHandler()

def setup_plot_interactions(fig, ax):
    """Sets up zooming and panning interactions for plots"""
    def on_scroll(event):
        if event.button == 'up':
            k_x = 0.5
            k_y = 0.5
        else:  # 'down'
            k_x = 2
            k_y = 2
        xdata, ydata = event.xdata, event.ydata
        l0, r0 = ax.get_xlim()
        l1 = xdata - k_x*xdata + k_x*l0
        r1 = xdata - k_x*xdata + k_x*r0
        ax.set_xlim(l1, r1)
        l0, r0 = ax.get_ylim()
        l1 = ydata - k_y*ydata + k_y*l0
        r1 = ydata - k_y*ydata + k_y*r0
        ax.set_ylim(l1, r1)
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect('scroll_event', on_scroll)
    
    move_handle = create_movement_handler()
    fig.canvas.mpl_connect('button_press_event', move_handle.on_press)
    fig.canvas.mpl_connect('motion_notify_event', move_handle.on_motion)
    fig.canvas.mpl_connect('button_release_event', move_handle.on_release)

# ===============================
# VISUALIZATION FUNCTIONS
# ===============================
"""Functions for data plotting and visualization"""

def plot_wells():
    """
    Creates an interactive plot for selected wells.
    Features:
    - Individual well selection
    - Linear/log scale options
    - Customizable titles
    - Pan and zoom capabilities
    """
    fig, ax = plt.subplots()
    title_window = None
    df_mod_well = pd.DataFrame()
    cols = ["A","B", "C", "D", "E", "F", "G", "H"]
    
    for i in range(0, 8):
        for j in range(0, 12):
            if well_vars[i][j].get() == 1:
                if j < 9:
                    well_name = str(cols[i])+str(0)+str(j+1)
                else:
                    well_name = str(cols[i])+str(j+1)
                df_mod_well = pd.concat([df_mod_well, df.query("well==@well_name")], ignore_index=True)

    for well in df_mod_well['well'].unique():
        data_subset = df_mod_well[df_mod_well['well'] == well]
        if well_log.get() == 1:
            ax.plot(data_subset['time_h'], data_subset['lnOD'], label=well)
        else:
            ax.plot(data_subset['time_h'], data_subset['OD'], label=well)
            
    if well_log.get() == 1:
        ax.set_ylabel("lnOD")
    else:
        ax.set_ylabel("Optical Density")
    ax.set_xlabel('Time (h)')
    ax.legend()
    
    default_title = "Growth Curves in Different Wells"
    ax.set_title(default_title)
    ax.set_aspect("auto")
    ax.autoscale(enable=True)
    ax.margins(x=0)

    if edit_title.get() == 1:
        title_window = add_title_controls(fig, ax, default_title)

    setup_plot_interactions(fig, ax)

    def on_close(event):
        if title_window is not None:
            title_window.destroy()
    fig.canvas.mpl_connect('close_event', on_close)

    plt.show()

def plot_contents():
    """
    Creates an interactive plot for different conditions.
    Features:
    - Condition-based grouping
    - Individual/aggregated data views
    - Confidence intervals
    - Custom color schemes
    - Pan and zoom capabilities
    """
    global color_vars, to_plot, indiv, ci_95, cond_log, edit_title_contents, all_conditions
    
    fig, ax = plt.subplots(figsize=(8, 6))
    title_window = None
    df_mod_cond = pd.DataFrame()

    selected_conditions = []
    for i, (condition, dataset) in enumerate(all_conditions):
        if to_plot[i].get() == 1:
            dataset_condition_data = df[(df['content'] == condition) & (df['dataset'] == dataset)]
            df_mod_cond = pd.concat([df_mod_cond, dataset_condition_data], ignore_index=True)
            selected_conditions.append((condition, dataset))

    if not df_mod_cond.empty:
        palette = {cond: color_vars[f'group_{i}']['color'] 
                  for i, (cond, _) in enumerate(set((c, d) for c, d in selected_conditions))}
        
        if indiv.get() == 1:
            for condition, dataset in set((c, d) for c, d in selected_conditions):
                condition_data = df_mod_cond[
                    (df_mod_cond['content'] == condition) & 
                    (df_mod_cond['dataset'] == dataset)
                ]
                
                base_color = np.array(mcolors.to_rgb(palette[condition]))
                wells = condition_data['well'].unique()
                
                for i, well in enumerate(wells):
                    well_data = condition_data[condition_data['well'] == well]
                    
                    if len(wells) > 1:
                        variation = 0.15
                        factor = 1 + variation * (i / (len(wells) - 1) - 0.5)
                        varied_color = np.clip(base_color * factor, 0, 1)
                        color = mcolors.to_hex(varied_color)
                    else:
                        color = palette[condition]
                    
                    if cond_log.get() == 1:
                        ax.plot(well_data['time_h'], well_data['lnOD'], 
                               color=color, alpha=0.7,
                               label=f"{condition} - {well}")
                    else:
                        ax.plot(well_data['time_h'], well_data['OD'], 
                               color=color, alpha=0.7,
                               label=f"{condition} - {well}")
        else:
            if cond_log.get() == 1:
                if ci_95.get() == 1:
                    sns.lineplot(data=df_mod_cond, x="time_h", y="lnOD", 
                               hue="content", style="dataset", ci=95, ax=ax,
                               palette=palette)
                else:
                    sns.lineplot(data=df_mod_cond, x="time_h", y="lnOD", 
                               hue="content", style="dataset", ci=None, ax=ax,
                               palette=palette)
                ax.set_ylabel("ln(OD)")
            else:
                if ci_95.get() == 1:
                    sns.lineplot(data=df_mod_cond, x="time_h", y="OD", 
                               hue="content", style="dataset", ci=95, ax=ax,
                               palette=palette)
                else:
                    sns.lineplot(data=df_mod_cond, x="time_h", y="OD", 
                               hue="content", style="dataset", ci=None, ax=ax,
                               palette=palette)
                ax.set_ylabel("Optical Density")

        ax.set_xlabel("Time (h)")
        default_title = "Growth Curves in Different Conditions"
        ax.set_title(default_title)
        ax.set_aspect("auto")
        ax.autoscale(enable=True)
        ax.margins(x=0)

        if indiv.get() == 1:
            handles, labels = ax.get_legend_handles_labels()
            by_condition = {}
            for h, l in zip(handles, labels):
                condition = l.split(" - ")[0]
                if condition not in by_condition:
                    by_condition[condition] = (h, l)
            
            ax.legend(
                [h for h, l in by_condition.values()],
                [l.split(" - ")[0] for h, l in by_condition.values()]
            )

        if edit_title_contents.get() == 1:
            title_window = add_title_controls(fig, ax, default_title)

        setup_plot_interactions(fig, ax)

        def on_close(event):
            if title_window is not None:
                title_window.destroy()
        fig.canvas.mpl_connect('close_event', on_close)

        plt.show()

# ===============================
# EVENT HANDLERS
# ===============================
"""Functions handling user interactions and events"""

def update_condition_checkbuttons():
    """Updates the checkbuttons for different conditions"""
    global left_frame, right_frame, to_plot, indiv, ci_95, cond_log, edit_title_contents
    global color_vars, all_conditions
    
    # Clear existing widgets
    for widget in left_frame.winfo_children():
        widget.destroy()
    
    for widget in right_frame.winfo_children():
        widget.destroy()
    
    # Rebuild conditions and widgets

    all_conditions = []
    for single_df in df_list:
        conditions = list(single_df["content"].unique())
        datasets = list(single_df["dataset"].unique())
        all_conditions.extend([(cond, dataset) for cond, dataset in zip(conditions, [datasets[0]]*len(conditions))])
    
    to_plot = [IntVar() for _ in all_conditions]
    
    # Left side: select/deselect + checkboxes
    select_buttons_frame = Frame(left_frame, bg='white')
    select_buttons_frame.pack(fill=X, pady=5)
    
    def select_all_conditions():
        for var in to_plot:
            var.set(1)

    def deselect_all_conditions():
        for var in to_plot:
            var.set(0)

    select_all_button = Button(select_buttons_frame, text="Select All", command=select_all_conditions)
    select_all_button.pack(side=LEFT, padx=5)
    style_button(select_all_button)

    deselect_all_button = Button(select_buttons_frame, text="Deselect All", command=deselect_all_conditions)
    deselect_all_button.pack(side=LEFT, padx=5)
    style_button(deselect_all_button)

    ttk.Separator(left_frame, orient='horizontal').pack(fill=X, pady=5)
    Label(left_frame, text="Select conditions to plot:", font=('Arial', 10, 'bold')).pack(anchor=W, pady=(0,5))

    for i, (condition, dataset) in enumerate(all_conditions):
        name = f"{condition} ({dataset})"
        cb = Checkbutton(left_frame, text=name, variable=to_plot[i],
                         bg='white', activebackground='#e3f2fd',
                         font=('Arial', 9), cursor='hand2')
        cb.pack(anchor=W, pady=2)

    # Right side: extra options + color pickers
    controls_frame = Frame(right_frame, bg='white', relief='groove', bd=1)
    controls_frame.pack(fill=X, pady=5)
    options_frame = Frame(right_frame, bg='white', relief='groove', bd=1)
    options_frame.pack(fill=X, pady=5)

    # Create variables for checkboxes
    indiv = IntVar()
    ci_95 = IntVar()
    cond_log = IntVar()
    edit_title_contents = IntVar()
    
    # Add checkboxes with better organization
    Checkbutton(options_frame, text="Plot individual data", variable=indiv, bg='white').pack(anchor=W)
    Checkbutton(options_frame, text="Include 95% confidence intervals", variable=ci_95, bg='white').pack(anchor=W)
    Checkbutton(options_frame, text="Log scale", variable=cond_log, bg='white').pack(anchor=W)
    Checkbutton(options_frame, text="Edit title", variable=edit_title_contents, bg='white').pack(anchor=W)

    # Color control frame with dropdown
    color_frame = Frame(right_frame, bg='white', relief='groove', bd=1)
    color_frame.pack(fill=X, pady=10)

    # Create a container for the color controls that will be shown/hidden
    color_controls_container = Frame(color_frame, bg='white')
    
    def toggle_color_controls():
        if color_controls_container.winfo_ismapped():
            color_controls_container.pack_forget()
            toggle_button.configure(text="▼ Show Color Controls")
        else:
            color_controls_container.pack(fill=X, pady=5)
            toggle_button.configure(text="▲ Hide Color Controls")

    # Add toggle button
    toggle_button = Button(color_frame, text="▼ Show Color Controls",
                         command=toggle_color_controls,
                         bg='white', relief='flat',
                         font=('Arial', 10),
                         cursor='hand2')
    toggle_button.pack(fill=X, pady=5)

    def choose_color(condition_key):
        color = colorchooser.askcolor(title="Choose color")[1]
        if color:
            color_vars[condition_key]['color'] = color
            color_vars[condition_key]['button'].configure(bg=color)

    # Add color controls to the container (initially hidden)
    Label(color_controls_container, text="Group colors:", 
          font=('Arial', 10, 'bold')).pack(anchor=W, pady=(5,5))

    color_vars = {}
    unique_conditions = set(cond for cond, _ in all_conditions)
    for i, condition in enumerate(unique_conditions):
        condition_key = f'group_{i}'
        color_vars[condition_key] = {'color': list(mcolors.TABLEAU_COLORS.values())[i % len(mcolors.TABLEAU_COLORS)]}
        
        group_color_frame = Frame(color_controls_container, bg='white')
        group_color_frame.pack(anchor=W)
        Label(group_color_frame, text=f"Group {i+1}:", bg='white').pack(side=LEFT)
        
        color_button = Button(group_color_frame, text="Pick Color",
                            command=lambda k=condition_key: choose_color(k),
                            bg=color_vars[condition_key]['color'])
        color_button.pack(side=LEFT, padx=5)
        color_vars[condition_key]['button'] = color_button

def on_blanking_confirmation():
    """Sets up the main application window after blanking decision"""
    global initial_window, master, tabControl, well_vars, well_log, edit_title
    initial_window.quit()
    initial_window.destroy()

    master = Tk()
    master.title("Plate Reader Analysis")
    master.geometry("800x600")
    master.minsize(800, 600)
    
    style = ttk.Style()
    style.theme_use('clam')
    style.configure('Notebook', background='#eaeaea', padding=5)
    style.configure('Notebook.Tab', padding=[15, 5], font=('Arial', 10))
    style.configure('TFrame', background='#eaeaea')
    style.configure('Custom.TButton', padding=10, font=('Arial', 10))
    
    top_frame = Frame(master, bg='#eaeaea', height=20)
    top_frame.place(relx=0, rely=0, relwidth=1, relheight=0.1)

    center_frame = Frame(master, bg='#eaeaea')
    center_frame.place(relx=0, rely=0.05, relwidth=1, relheight=0.8)
    
    bottom_frame = Frame(master, bg='#eaeaea', height=5)
    bottom_frame.place(relx=0, rely=0.9, relwidth=1, relheight=0.1)
    
    


    # Configure the main scrollbar style
    style.configure("Custom.Vertical.TScrollbar",
        troughcolor='#FFFFFF',         # Background of the scrollbar area
        background='#E0E0E0',          # Color of the slider itself (light gray)
        arrowcolor='#E0E0E0',          # Color of the arrows
        bordercolor='#FFFFFF',         # Border color
        lightcolor='#E0E0E0',          # Pressed state color
        darkcolor='#E0E0E0'           # Pressed state darker shade
    )

    style.map("Custom.Vertical.TScrollbar",
        background=[('disabled', '#FFFFFF'), ('active', '#E0E0E0')],
        arrowcolor=[('disabled', '#FFFFFF'), ('active', '#E0E0E0')],
        troughcolor=[('disabled', '#FFFFFF')],
        lightcolor=[('disabled', '#FFFFFF')],
        darkcolor=[('disabled', '#FFFFFF')]
    )

    # Create a canvas with scrollbar for the main container
    canvas = Canvas(center_frame, bg='#eaeaea')
    scrollbar = ttk.Scrollbar(center_frame, orient=VERTICAL, command=canvas.yview, style="Custom.Vertical.TScrollbar")
    
    main_container = Frame(canvas, bg='#eaeaea', padx=20, pady=20)
    canvas_frame = canvas.create_window((0, 0), window=main_container, anchor=NW)
    
    # Configure canvas
    canvas.configure(yscrollcommand=scrollbar.set)
    
    # Pack scrollbar and canvas
    scrollbar.pack(side=RIGHT, fill=Y)
    canvas.pack(side=LEFT, fill=BOTH, expand=True)
    
    def configure_scroll_region(event):
        canvas.configure(scrollregion=canvas.bbox("all"))
    
    def configure_canvas_width(event):
        canvas.itemconfig(canvas_frame, width=event.width)
    
    main_container.bind('<Configure>', configure_scroll_region)
    canvas.bind('<Configure>', configure_canvas_width)
    
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")
    
    canvas.bind_all("<MouseWheel>", on_mousewheel)

    # Tab styling
    tabControl = ttk.Notebook(main_container, padding=10)

    if len(df_list) == 1:
        tab1 = ttk.Frame(tabControl, padding=10)
        tabControl.add(tab1, text="Well Analysis")
        
        # Add file header before wells container
        file_header = create_file_header(tab1)
        
        # Well analysis container styling
        wells_container = Frame(tab1, bg='white', padx=20, pady=20, relief='ridge', bd=1)
        wells_container.pack(fill=BOTH, expand=True, padx=10, pady=10)
        
        # Add title for well selection
        Label(wells_container, text="Select Wells for Analysis", 
              font=('Arial', 12, 'bold'), bg='white').pack(pady=(0,15))

        # Wells grid styling
        wells_grid = Frame(wells_container, bg='white')
        wells_grid.pack(fill=BOTH, expand=True)
        
        # Initialize well_vars and create grid
        well_vars = []
        cols = ["A","B", "C", "D", "E", "F", "G", "H"]
        wells_grid = Frame(wells_container, bg='white')
        wells_grid.pack(fill=BOTH, expand=True)
        
        for i in range(0, 8):
            well_vars.append([])
            for j in range(0, 12):
                well_vars[i].append(IntVar())
                name = str(cols[i])+str(j+1)
                cb = Checkbutton(wells_grid, text=name, variable=well_vars[i][j],
                               bg='white', activebackground='#e3f2fd',
                               font=('Arial', 9), cursor='hand2')
                cb.grid(row=i+1, column=j+1, sticky=W, padx=3, pady=3)

        # Controls frame styling
        controls_frame = Frame(wells_container, bg='white', relief='groove', bd=1)
        controls_frame.pack(fill=X, pady=15, padx=10)
        
        # Left side controls
        left_controls = Frame(controls_frame, bg='white')
        left_controls.pack(side=LEFT, padx=10)
        
        well_log = IntVar()
        Checkbutton(left_controls, text="Log scale", variable=well_log, bg='white').pack(side=LEFT, padx=5)
        
        edit_title = IntVar()
        Checkbutton(left_controls, text="Edit title", variable=edit_title, bg='white').pack(side=LEFT, padx=5)

        # Right side controls (empty frame for layout)
        right_controls = Frame(controls_frame, bg='white')
        right_controls.pack(side=RIGHT, padx=10)

        # Select all/deselect all frame
        select_frame = Frame(wells_container, bg='white')
        select_frame.pack(fill=X, pady=5)

        class select_all_across():
            def __init__(self, row_no):
                self.row_no = row_no

            def set_values(self):
                for a in range(0, 12):
                    well_vars[self.row_no][a].set(1)

        # Row selectors
        row_frame = Frame(wells_grid, bg='white')
        row_frame.grid(row=0, column=0, sticky=W)
        for i in range(0, 8):
            inst_across = select_all_across(i)
            Button(wells_grid, text="all", command=inst_across.set_values, bg='white').grid(row=i+1, column=0, sticky=W, pady=4)

        class select_all_down():
            def __init__(self, col_no):
                self.col_no = col_no

            def set_values(self):
                for a in range(0, 8):
                    well_vars[a][self.col_no].set(1)

        # Column selectors
        for i in range(0, 12):
            inst_down = select_all_down(i)
            Button(wells_grid, text="all", command=inst_down.set_values, bg='white').grid(row=0, column=i+1, sticky=W, pady=4)

        def select_all():
            for i in range(0, 12):
                for j in range(0, 8):
                    well_vars[j][i].set(1)

        def deselect_all():
            for i in range(0, 12):
                for j in range(0, 8):
                    well_vars[j][i].set(0)

        # Selection buttons frame
        selection_buttons_frame = Frame(wells_container, bg='white')
        selection_buttons_frame.pack(fill=X, pady=5)
        select_all_button = Button(selection_buttons_frame, text="Select All", command=select_all)
        select_all_button.pack(side=LEFT, padx=5)
        style_button(select_all_button)
        deselect_all_button = Button(selection_buttons_frame, text="Deselect All", command=deselect_all)
        deselect_all_button.pack(side=LEFT, padx=5)
        style_button(deselect_all_button)

    # Group Analysis tab
    tab2 = ttk.Frame(tabControl, padding=10)
    tabControl.add(tab2, text="Group Analysis")
    
    # Add file header before conditions container
    file_header = create_file_header(tab2)
    
    # Create main container for group analysis with fixed proportions
    group_container = Frame(tab2)
    group_container.pack(fill=BOTH, expand=True, padx=5, pady=5)
    
    # Configure grid weights
    group_container.grid_columnconfigure(0, weight=1)
    group_container.grid_columnconfigure(1, weight=1)
    group_container.grid_rowconfigure(0, weight=1)

    def create_scrollable_frame(parent, side):
        # Create container
        container = Frame(parent, bg='white', relief='ridge', bd=1)
        container.grid(row=0, column=0 if side == 'left' else 1, sticky='nsew', padx=5)
        
        # Create canvas and scrollbar
        canvas = Canvas(container, bg='white', highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=VERTICAL, command=canvas.yview, style="Custom.Vertical.TScrollbar")
        
        # Create frame inside canvas
        scrollable_frame = Frame(canvas, bg='white')
        scrollable_frame.bind('<Configure>', 
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Add frame to canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor='nw', width=canvas.winfo_reqwidth())
        
        # Configure canvas
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack elements
        scrollbar.pack(side=RIGHT, fill=Y)
        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        
        return scrollable_frame, canvas

    # Create scrollable frames for left and right content
    global left_frame, right_frame
    left_frame, left_canvas = create_scrollable_frame(group_container, 'left')
    right_frame, right_canvas = create_scrollable_frame(group_container, 'right')
    
    # Initialize condition checkbuttons
    update_condition_checkbuttons()
    
    # Pack the tab control
    tabControl.pack(fill=BOTH, expand=True)

    # Create button frame
    button_frame = Frame(master, bg='#f0f0f0')
    button_frame.pack(side=BOTTOM, fill=X, padx=10, pady=10)
    
    def add_file():
        global df, df_list, file_paths
        new_path = fd.askopenfilename()
        if new_path:
            try:
                new_df = read_and_process_file(new_path)
                df_list.append(new_df)
                file_paths.append(new_path)
                df = pd.concat(df_list, ignore_index=True, sort=False)
                
                # Remove well analysis tab if multiple files
                if len(df_list) > 1:
                    for index in range(tabControl.index("end")):
                        if tabControl.tab(index, "text") == "Well Analysis":
                            tabControl.forget(index)
                            break
                    tabControl.select(0)
                
                update_condition_checkbuttons()
                
                # Update file headers
                for tab in tabControl.winfo_children():
                    for widget in tab.winfo_children():
                        if isinstance(widget, Frame) and widget.winfo_children() and \
                           isinstance(widget.winfo_children()[0], Label):
                            files_text = "Current files: " + ", ".join(
                                os.path.basename(f) for f in file_paths)
                            widget.winfo_children()[0].config(text=files_text)
                
                # Force geometry update
                master.update_idletasks()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error loading file: {str(e)}")
                print(f"Detailed error: {str(e)}")
    
    def plot_based_on_tab():
        current_tab = tabControl.index(tabControl.select())
        if current_tab == 0 and tabControl.tab(0, "text") == "Well Analysis":
            plot_wells()
        else:
            plot_contents()

    # Add the buttons with better spacing
    reupload_button = Button(button_frame, text="Re-upload Data File", 
           command=lambda: os.execl(sys.executable, sys.executable, *sys.argv))
    reupload_button.pack(side=LEFT, padx=5)
    style_button(reupload_button)
    
    add_file_button = Button(button_frame, text="Add File", command=add_file)
    add_file_button.pack(side=LEFT, padx=5)
    style_button(add_file_button)

    # Add the plot button at the bottom
    plot_button = Button(button_frame, text="Plot", command=plot_based_on_tab)
    plot_button.pack(side=LEFT, padx=5)
    style_button(plot_button)

    master.mainloop()

def handle_blanking():
    """
    Handles the data blanking process.
    Features:
    - Interactive region selection
    - Real-time visual feedback
    - Automatic blank calculation
    - Multi-file support
    """
    global df, xmin, xmax
    fig, ax = plt.subplots()
    for well in df["well"].unique():
        data_subset = df[df["well"] == well]
        ax.plot(data_subset["time_h"], data_subset["OD"]) 
    ax.set_xlabel("time (h)")   
    ax.set_ylabel("OD")
    ax.legend(df["well"].unique())
    ax.set_title("Raw growth curves")
    ax.autoscale(enable=True)
    ax.margins(x=0)

    class SelectBlanks:
        def __init__(self):
            self.xinit = None
            self.yinit = None
            self.xfin = None
            self.yfin = None
            self.rect = None
            self.goon = False

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
                width = self.xfin - self.xinit
                height = self.yfin - self.yinit
                self.rect = patches.Rectangle((self.xinit, self.yinit), width, height, 
                                           linewidth=1, edgecolor='r', facecolor='none')
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
            self.rect = patches.Rectangle((self.xinit, self.yinit), width, height, 
                                       linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(self.rect)
            fig.canvas.draw_idle()

    plot_blnk = SelectBlanks()
    fig.canvas.mpl_connect("button_press_event", plot_blnk.on_press)
    fig.canvas.mpl_connect("motion_notify_event", plot_blnk.on_motion)
    fig.canvas.mpl_connect("button_release_event", plot_blnk.on_release)
    plt.show()

    df_blanked = pd.DataFrame()
    for single_df in df_list:
        df_blanked_single = pd.DataFrame(columns=["content", "well", "time", "time_h", "OD", "dataset"])
        wells = single_df["well"].unique()
        for well in wells:
            df_with_blanks = single_df.query("well == @well and time_h>@xmin and time_h<@xmax")
            blank = np.average(df_with_blanks["OD"].to_list())
            df_to_blank = single_df.query("well == @well")
            list_ods = df_to_blank["OD"].to_list()
            list_ods_blanked = [a-blank for a in list_ods]
            list_ods_blanked_ln = [np.log(a) for a in list_ods_blanked]
            
            to_add = pd.DataFrame({
                "content": df_to_blank["content"],
                "well": df_to_blank["well"],
                "time": df_to_blank["time"],
                "time_h": df_to_blank["time_h"],
                "OD": list_ods_blanked,
                "lnOD": list_ods_blanked_ln,
                "dataset": df_to_blank["dataset"]
            })
            df_blanked_single = pd.concat([df_blanked_single, to_add], ignore_index=True)
        
        df_blanked = pd.concat([df_blanked, df_blanked_single], ignore_index=True)

    df = df_blanked
    on_blanking_confirmation()

def skip_blanking():
    """Skips the blanking process"""
    on_blanking_confirmation()

def create_initial_window():
    """Creates and configures the initial data blanking window"""
    window = Tk()
    window.geometry("800x150")
    window.title("Data Blanking Options")
    
    main_frame = Frame(window, padx=20, pady=20)
    main_frame.pack(expand=True, fill=BOTH)
    
    Label(main_frame, text="Would you like to blank the data?", 
          font=('Arial', 12)).pack(pady=(0,20))
    
    button_frame = Frame(main_frame)
    button_frame.pack()
    
    yes_button = Button(button_frame, text="Yes", command=handle_blanking, width=10, padx=10)
    yes_button.pack(side=LEFT, padx=10)
    style_button(yes_button)
    
    no_button = Button(button_frame, text="No", command=skip_blanking, width=10, padx=10)
    no_button.pack(side=LEFT, padx=10)
    style_button(no_button)
    
    return window

# ===============================
# MAIN APPLICATION LOGIC
# ===============================
"""Core application setup and initialization"""

def main():
    """
    Main entry point for the application.
    Initializes the GUI and sets up the initial data loading process.
    """
    global df, df_list, file_paths, initial_window
    
    # Get initial data file
    path = fd.askopenfilename()
    if not path:  # User cancelled file selection
        sys.exit(0)
        
    df_list = [read_and_process_file(path)]
    df = df_list[0]
    file_paths = [path]
    
    # Create and show initial window
    initial_window = create_initial_window()
    initial_window.mainloop()

if __name__ == "__main__":
    main()