# Data visualisation GUI
This script allows users to quickly plot and analyse raw data from plate reader experiments. File formats accepted are .csv and .xlsx (i.e., depending on the type of software used by your plate reader, you might need to export the data to one of these formats)
The script can be run from the terminal or inside a code editor. The following steps are involved: <br>
1. **Choose data file:** a File window will open and the user can go to the location of the file
2. **Blanking the data option:** a window asking the user if they want to blank the data will open <br>
a. The user does not want the data blanked, in which case the programme proceeds to step 3 <br>
b. The user wants to blank the data, in which case, all data is plotted in a single window and the user is asked to select the part of the graph that represents background absorbance/fluorescence by drawing a rectangle around that region.
3. **The analysis GUI:** this window will allow the user to select the wells (under the "well analysis" tab) or groups (under the "group analysis" tab) from which the data is to be plotted, and whether the absorbance/fluorencence axis should be logged or linear. In the case of group analysis, individual replicates can be analysed when the "plot individual data" box is ticked, and 95% confidence intervals can be included when means are plotted if the "include 95 confidence intervals" box is ticked.
<br>
The programme stops when the analysis GUI is closed. Only one data file can be analysed in one run, but multiple plots can be plotted. Plots can be zoomed in and out, and moved around to focus on regions of interest.
