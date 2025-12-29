#%%

# Contact: Diana Coroiu
# Affiliation (at the time of writing): University of Edinburgh
# Contact:
#   - personal: didicoroiu@yahoo.com 
#   - institutional: d.coroiu@sms.ed.ac.uk
# Last updated: 12/2025

# Description:

# This code allows to interact with raw plate reader data
# The raw data has to be in the .csv format and follow a given structure:
# (1): table headings in row 8, data start in row 9
# (2): time to be given in minutes (otherwise time scale will not make sense)
# (3): axis 0 (rows) to contain different wells, axis 1 (columns) to contain different time points
# (4): wells labelled as "A01" and not "A1"

# Intended for academic use; feel free to adapt for other plate-reader data file formats.

# Full instructions and details on GitHub: https://github.com/Pilizota-lab/Plate-reader-analysis/

from tkinter import *
from tkinter import ttk
from tkinter import filedialog as fd
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import seaborn as sns

path = fd.askopenfilename()
#define dataframe
data = pd.read_csv(path, skiprows = 7)
# if csv file contains a comma after final value in each row (common), remove from df
data.columns.astype(str).str.strip() != ""
data_tr = data.transpose()
print(data)
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

#def what happens after Yes or No are pressed (plotting code)
def on_blanking_confirmation():
    #this function will be called when the blanking window is closed
    #first destoy old window
    initial_window.quit()
    initial_window.destroy()

    #create new window with plotting options
    master = Tk() #creates parent window (root)
    master.title("Plate reader analysis GUI")
    tabControl = ttk.Notebook(master)
    tab1 = ttk.Frame(tabControl) #create tab and put in tab control
    tabControl.add(tab1, text = "well analysis")
    tabControl.pack(expand=1, fill="both")
    tab2 = ttk.Frame(tabControl) #create tab and put in tab control
    tabControl.add(tab2, text = "group analysis")
    tabControl.pack(expand=1, fill="both")

    #code to plot per well
    #create matrix (as list of lists) to memorise whether boxes are ticked or not
    vars = []
    cols = ["A","B", "C", "D", "E", "F", "G", "H"]
    for i in range(0, 8):
        vars.append([])
        for j in range(0, 12):
            vars[i].append(IntVar()) #then do vars[x].get() to see whether box was ticked (1) or not (0)
            name = str(cols[i])+str(j+1)
            Checkbutton(tab1, text=name, variable = vars[i][j]).grid(row=i+1, column = j+1, sticky=W)
            
    well_log = IntVar()
    Checkbutton(tab1, text = "log", variable = well_log).grid(row=10, column = 0, sticky=W)

    #add select all buttons

    #create all buttons that select all wells across
    class select_all_across():
        def __init__(self, row_no):
            self.row_no = row_no

        def set_values(self):
            for a in range(0, 12):
                vars[self.row_no][a].set(1)
            
    for i in range(0, 8):
        inst_across = select_all_across(i)
        Button(tab1, text="all", command=inst_across.set_values).grid(row=i+1, column = 0, sticky = W, pady = 4)

        
    #create all buttons that select all wells down
    class select_all_down():
        def __init__(self, col_no):
            self.col_no = col_no

        def set_values(self):
            for a in range(0, 8):
                vars[a][self.col_no].set(1)

    for i in range(0, 12):
        inst_down = select_all_down(i)
        Button(tab1, text="all", command=inst_down.set_values).grid(row=0, column = i+1, sticky = W, pady = 4)


    #create buttons for select all and deselect all
    def select_all():
        for i in range(0, 12):
            for j in range(0, 8):
                vars[j][i].set(1)

    def deselect_all():
        for i in range(0, 12):
            for j in range(0, 8):
                vars[j][i].set(0)

    Button(tab1, text="select all", command=select_all).grid(row=11, column = 0, sticky = W, pady = 4)
    Button(tab1, text="deselect all", command=deselect_all).grid(row= 12, column = 0, sticky = W, pady = 4)

    def plot_wells():
        if well_log.get()==1: # will need 2 y-axes (one for logged data, the other for linear data but logged scale)
            twin = True
        else:
            twin = False
        fig, ax = plt.subplots()
        if twin:
            ax_twin = ax.twinx()
        cols = ["A","B", "C", "D", "E", "F", "G", "H"]
        df_mod_well = pd.DataFrame()
        for i in range(0, 8):
            for j in range(0, 12):
                if vars[i][j].get() == 1:
                    #get name of the well
                    if j<9:
                        well_name = str(cols[i])+str(0)+str(j+1)
                    else:
                        well_name = str(cols[i])+str(j+1)
                    df_mod_well = pd.concat([df_mod_well, df.query("well==@well_name")], ignore_index=True)

        for well in df_mod_well['well'].unique():
            data_subset = df_mod_well[df_mod_well['well'] == well]
            if well_log.get()==1:
                ax.plot(data_subset['time_h'], data_subset['lnOD'], label=well)
                # also plot linear OD data on a logged axis
                ax_twin.plot(data_subset['time_h'], data_subset['OD'], alpha = 0) # not visible
                ax_twin.set_yscale('log')
            else:
                ax.plot(data_subset['time_h'], data_subset['OD'], label=well)
        if well_log.get()==1:
            ax.set_ylabel("lnOD")
            ax_twin.set_ylabel('corresponding OD')
            ax_twin.set_zorder(ax.get_zorder() - 1)  # push twin behind main; otherwise events will belong to this axis - important!
        else:
            ax.set_ylabel("optical density")
        ax.set_xlabel('Time (h)')
        ax.legend()
        ax.set_title("Growth curves in different wells")
        ax.set_aspect("auto")
        ax.autoscale(enable=True)
        ax.margins(x=0)


        # Add interactive functionality
        def on_scroll(event):
            # set scroll gains (zoom in/out rates)
            # if gain>1 => zoom out; if gain<1 => zoom in
            k_x = 2
            k_y = 2
            if event.button == 'up': # Zoom in
                xdata, ydata = event.xdata, event.ydata #coordinates of the mouse cursor where the event occured - expand plot around location of mouse cursor
                #for x axis (zoomin)
                l0 = ax.get_xlim()[0] # getting old coords
                r0 = ax.get_xlim()[1]
                l1 = xdata - (1/k_x)*xdata + (1/k_x)*l0  # compute new coords according to scroll gain set above
                r1 = xdata - (1/k_x)*xdata + (1/k_x)*r0
                ax.set_xlim(l1,r1)
                #for y axis (zoomin)
                l0 = ax.get_ylim()[0]
                r0 = ax.get_ylim()[1]
                l1 = ydata - (1/k_y)*ydata + (1/k_y)*l0
                r1 = ydata - (1/k_y)*ydata + (1/k_y)*r0
                ax.set_ylim(l1,r1)
                if twin == True: # need to set new y axis limits on a log scale
                    linear_l1 = np.exp(l1) #e^(l1) of originally computed l1 for y-axis
                    linear_r1 = np.exp(r1)
                    ax_twin.set_ylim(linear_l1, linear_r1)

                
            #define zoom-out rate as above
            elif event.button == 'down':
                # Zoom out
                xdata, ydata = event.xdata, event.ydata
                # get old coordinates
                l0 = ax.get_xlim()[0]
                r0 = ax.get_xlim()[1]
                l1 = xdata - k_x*xdata + k_x*l0
                r1 = xdata - k_x*xdata + k_x*r0
                ax.set_xlim(l1,r1)
                #y axis zoomin
                l0 = ax.get_ylim()[0]
                r0 = ax.get_ylim()[1]
                l1 = ydata - k_y*ydata + k_y*l0
                r1 = ydata - k_y*ydata + k_y*r0
                ax.set_ylim(l1,r1)
                if twin==True: # need to set new y axis limits on a log scale
                    linear_l1 = np.exp(l1)
                    linear_r1 = np.exp(r1)
                    ax_twin.set_ylim(linear_l1, linear_r1)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('scroll_event', on_scroll)


        class MovementHandler:

            def __init__(self):
                self.xpress = None # x coordinate of mouse press event
                self.ypress = None # y coordinate of mouse press event
                self.xrelease = None # x coordinate of mouse release event
                self.yrelease = None # y coordinate of mouse release event
                self.keep_running = True
            
            def on_press(self, event):
                self.xpress, self.ypress = event.xdata, event.ydata
                self.keep_running = True
            
            def on_motion(self, event):
                if event.inaxes and self.keep_running==True: # check if the mouse is inside the plot area
                    self.xrelease, self.yrelease = event.xdata, event.ydata # get the current x and y coordinates
                    x_change = self.xrelease - self.xpress
                    y_change = self.yrelease - self.ypress
                    #get old coordinates
                    l0 = ax.get_xlim()[0]
                    r0 = ax.get_xlim()[1]     
                    l1 = l0 - x_change
                    r1 = r0 - x_change
                    ax.set_xlim(l1,r1)
                    l0 = ax.get_ylim()[0]
                    r0 = ax.get_ylim()[1]     
                    l1 = l0 - y_change
                    r1 = r0 - y_change
                    ax.set_ylim(l1,r1)
                    if twin==True: # need to set new y axis limits on a log scale
                        linear_l1 = np.exp(l1)
                        linear_r1 = np.exp(r1)
                        ax_twin.set_ylim(linear_l1, linear_r1)   
                    fig.canvas.draw_idle()

            #when release you have to stop moving
            def on_release(self, event):
                self.keep_running = False




        movm_handler = MovementHandler() # Create an instance of the ClickHandler class

        # Add the event handling functions to the plot
        fig.canvas.mpl_connect('button_press_event', movm_handler.on_press)
        #fig.canvas.mpl_connect('button_release_event', click_handler.on_release)
        fig.canvas.mpl_connect('motion_notify_event', movm_handler.on_motion)
        fig.canvas.mpl_connect('button_release_event', movm_handler.on_release)

        plt.show()
        return()

    #code to plot per condition
    #create list of condition IntVar()
    conditions = list(df["content"].unique())
    to_plot = [IntVar() for i in conditions]
    for i in range(len(to_plot)):
        name = conditions[i]
        Checkbutton(tab2, text=name, variable = to_plot[i]).grid(row=i, sticky=W)


    indiv =IntVar()
    Checkbutton(tab2, text="plot individual data (by default, means are plotted)", variable = indiv).grid(row=2, column=5, sticky=W)
    ci_95 = IntVar()
    Checkbutton(tab2, text="include 95 confidence intervals", variable = ci_95).grid(row=3, column = 5, sticky=W)
    cond_log = IntVar()
    Checkbutton(tab2, text="log", variable = cond_log).grid(row=4, column = 5, sticky=W)

    def plot_contents():
        fig, ax = plt.subplots()
        if cond_log.get()==1:
            twin = True
        else:
            twin = False
        if twin:
            ax_twin = ax.twinx()
        conditions = list(df["content"].unique())
        df_mod_cond = pd.DataFrame()

        for i in range(len(conditions)): # get data that needs plotting
            if to_plot[i].get() == 1:
                #get name of the condition
                cond = conditions[i]
                df_mod_cond = pd.concat([df_mod_cond, df.query("content==@cond")], ignore_index=True)

        if indiv.get()==1:
            #plot individual lines, not the average
            if cond_log.get()==1:
                sns.lineplot(data = df_mod_cond, x = "time_h", y = "lnOD", hue = "well", ci=None, ax = ax)
                ax.set_ylabel("ln(OD)")
                sns.lineplot(data = df_mod_cond, x = "time_h", y = "OD", hue = "well", ax = ax_twin, alpha=0)
                ax_twin.set_ylabel('corresponding OD')
                ax_twin.set_zorder(ax.get_zorder() - 1)
            else:
                sns.lineplot(data = df_mod_cond, x = "time_h", y = "OD", hue = "well", ci=None, ax = ax)
            
        else:
            if cond_log.get()==1:
                if ci_95.get()==1:
                    sns.lineplot(data = df_mod_cond, x = "time_h", y = "lnOD", hue = "content", ci=95, ax = ax)
                else:
                    sns.lineplot(data = df_mod_cond, x = "time_h", y = "lnOD", hue = "content", ci=None, ax = ax)
                ax.set_ylabel("ln(OD)")
                sns.lineplot(data = df_mod_cond, x = "time_h", y = "OD", hue = "well", ax = ax_twin, alpha=0)
                ax_twin.set_ylabel('corresponding OD')
                ax_twin.set_zorder(ax.get_zorder() - 1) 
            else:
                if ci_95.get()==1:
                    sns.lineplot(data = df_mod_cond, x = "time_h", y = "OD", hue = "content", ci=95, ax=ax)
                else:
                    sns.lineplot(data = df_mod_cond, x = "time_h", y = "OD", hue = "content", ci=None, ax=ax)
                ax.set_ylabel("optical density")  
        ax.set_xlabel("time (h)")
        ax.set_aspect("auto")
        # ax.autoscale(enable=True)
        ax.margins(x=0)
        ax.set_title("Growth curves in different conditions")

        # Add interactive functionality
        def on_scroll(event):
            # set scroll gains (zoom in/out rates)
            # if gain>1 => zoom out; if gain<1 => zoom in
            k_x = 2
            k_y = 2
            if event.button == 'up': # Zoom in
                xdata, ydata = event.xdata, event.ydata #coordinates of the mouse cursor where the event occured - expand plot around location of mouse cursor
                #for x axis (zoomin)
                l0 = ax.get_xlim()[0] # getting old coords
                r0 = ax.get_xlim()[1]
                l1 = xdata - (1/k_x)*xdata + (1/k_x)*l0  # compute new coords according to scroll gain set above
                r1 = xdata - (1/k_x)*xdata + (1/k_x)*r0
                ax.set_xlim(l1,r1)
                #for y axis (zoomin)
                l0 = ax.get_ylim()[0]
                r0 = ax.get_ylim()[1]
                l1 = ydata - (1/k_y)*ydata + (1/k_y)*l0
                r1 = ydata - (1/k_y)*ydata + (1/k_y)*r0
                ax.set_ylim(l1,r1)
                if twin == True: # need to set new y axis limits on a log scale
                    linear_l1 = np.exp(l1) #e^(l1) of originally computed l1 for y-axis
                    linear_r1 = np.exp(r1)
                    ax_twin.set_ylim(linear_l1, linear_r1)

                
            #define zoom-out rate as above
            elif event.button == 'down':
                # Zoom out
                xdata, ydata = event.xdata, event.ydata
                # get old coordinates
                l0 = ax.get_xlim()[0]
                r0 = ax.get_xlim()[1]
                l1 = xdata - k_x*xdata + k_x*l0
                r1 = xdata - k_x*xdata + k_x*r0
                ax.set_xlim(l1,r1)
                #y axis zoomin
                l0 = ax.get_ylim()[0]
                r0 = ax.get_ylim()[1]
                l1 = ydata - k_y*ydata + k_y*l0
                r1 = ydata - k_y*ydata + k_y*r0
                ax.set_ylim(l1,r1)
                if twin==True: # need to set new y axis limits on a log scale
                    linear_l1 = np.exp(l1)
                    linear_r1 = np.exp(r1)
                    ax_twin.set_ylim(linear_l1, linear_r1)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect('scroll_event', on_scroll)


        class MovementHandler:
            def __init__(self):
                self.xpress = None # x coordinate of mouse press event
                self.ypress = None # y coordinate of mouse press event
                self.xrelease = None # x coordinate of mouse release event
                self.yrelease = None # y coordinate of mouse release event
                self.keep_running = True
            
            def on_press(self, event):
                self.xpress, self.ypress = event.xdata, event.ydata
                self.keep_running = True
            
            def on_motion(self, event):
                if event.inaxes and self.keep_running==True: # check if the mouse is inside the plot area
                    self.xrelease, self.yrelease = event.xdata, event.ydata # get the current x and y coordinates
                    x_change = self.xrelease - self.xpress
                    y_change = self.yrelease - self.ypress
                    #get old coordinates
                    l0 = ax.get_xlim()[0]
                    r0 = ax.get_xlim()[1]     
                    l1 = l0 - x_change
                    r1 = r0 - x_change
                    ax.set_xlim(l1,r1)
                    l0 = ax.get_ylim()[0]
                    r0 = ax.get_ylim()[1]     
                    l1 = l0 - y_change
                    r1 = r0 - y_change
                    ax.set_ylim(l1,r1)   
                    fig.canvas.draw_idle()

            #when release you have to stop moving
            def on_release(self, event):
                self.keep_running = False


        move_handle = MovementHandler() # Create an instance of the ClickHandler class

        # Add the event handling functions to the plot
        fig.canvas.mpl_connect('button_press_event', move_handle.on_press)
        #fig.canvas.mpl_connect('button_release_event', click_handler.on_release)
        fig.canvas.mpl_connect('motion_notify_event', move_handle.on_motion)
        fig.canvas.mpl_connect('button_release_event', move_handle.on_release)


        plt.show()
        return()

    Button(tab1, text='Plot', command=plot_wells).grid(row=13, sticky=W, pady=4)
    Button(tab2, text='Plot', command=plot_contents).grid(row=6, column = 5, sticky=W, pady=4)
    master.mainloop()

#define Yes and No buttons for blanking
def blanking_needed():
    global df
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
            

    plot_blnk = SelectBlanks()
    fig.canvas.mpl_connect("button_press_event", plot_blnk.on_press)
    fig.canvas.mpl_connect("motion_notify_event", plot_blnk.on_motion)
    fig.canvas.mpl_connect("button_release_event", plot_blnk.on_release)
    plt.show()

    df_blanked = pd.DataFrame(columns=heads)
    for wl in range(len(wells)):
        well = wells[wl]
        #calculate average of points used for blanking
        df_with_blanks = df.query("well == @well and time_h>@xmin and time_h<@xmax")
        blank = round(np.average(df_with_blanks["OD"].to_list()), 3)
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
        df_blanked = pd.concat([df_blanked, to_add], ignore_index=True)
    df_blanked #this is cointains the blank data to work with from now on
    df = df_blanked
    on_blanking_confirmation()
    initial_window.mainloop()

def skip_blanking():
    on_blanking_confirmation()
    initial_window.mainloop()
    
initial_window = Tk()
initial_window.geometry("350x100")
initial_window.title("Would you like to blank the data?")
yes_button = Button(initial_window, text="Yes", command = blanking_needed)
yes_button.place(x=100, y=40)
no_button = Button(initial_window, text ="No", command = skip_blanking)
no_button.place(x=200, y=40)

mainloop() #this creates a loop of events that runs indefinitely until the application is closed 

#need to kill kernel at the end, can't run this interface a second time in the same kernel as it enters some sort of infinite loop
import os
os._exit(00)
