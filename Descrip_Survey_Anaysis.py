#Import libraries required for this project
import pandas as pd
import matplotlib.pyplot as plotter
from matplotlib.gridspec import GridSpec
import seaborn as sns
from statsmodels.sandbox.regression.try_treewalker import data2

#Access the data. There is one sample data set and another we will process and compare its results against the example.
df_example = pd.read_csv(r'kaggle_example.csv')
df_test = pd.read_csv(r'kaggle_desc_analysis_output.csv')


#Create a figure with 2 row 3 columns to compare graphics one against the other one
#fig, ([plot_hex3,plot_hex4,NULL]) = plotter.subplots(nrows=1,ncols=3, figsize=(14, 5))
#plotter.subplots_adjust(hspace=0.4,wspace=0.4)

fig = plotter.figure(figsize=(14,5))
grid = GridSpec(nrows=1,ncols= 3, width_ratios=[1,1,1.5])




# # draw hexbin plot
# hb1 = plot_hex1.hexbin(df_test['q3'], df_test['output_12'], gridsize=20, cmap='Greens', edgecolors='grey', linewidths=0.1)
# plot_hex1.set_title('Run 12: q3 vs Grieving Stage', fontsize=9)
# plot_hex1.set_xlabel('q3 Answer', fontsize=9)
# plot_hex1.set_ylabel('Grieving Stage', fontsize=9)
#
# hb2 = plot_hex2.hexbin(df_test['q3'], df_test['output_15'], gridsize=20, cmap='Greens', edgecolors='grey', linewidths=0.1)
# plot_hex2.set_title('Run 15: q3 vs Grieving Stage', fontsize=9)
# plot_hex2.set_xlabel('q3 Answer', fontsize=9)
# plot_hex2.set_ylabel('Grieving Stage', fontsize=9)
#
plot_hex3 = plotter.subplot(grid[0])
hb3 = plot_hex3.hexbin(df_test['q3'], df_test['output_20'], gridsize=20, cmap='Greens', edgecolors='grey', linewidths=0.1)
plot_hex3.set_title(label='Run 20: q3 vs Grieving Stage', fontsize=9)
plot_hex3.set_xlabel('q3 Answer', fontsize=9)
plot_hex3.set_ylabel('Grieving Stage', fontsize=9)
fig.colorbar(hb3, ax=plot_hex3)

plot_hex4 = plotter.subplot(grid[1])
hb4 = plot_hex4.hexbin(df_test['q3'], df_test['output_21'], gridsize=20, cmap='Greens', edgecolors='grey', linewidths=0.1)
plot_hex4.set_title('Run 21: q3 vs Grieving Stage', fontsize=9)
plot_hex4.set_xlabel('q3 Answer', fontsize=9)
plot_hex4.set_ylabel('Grieving Stage', fontsize=9)

# Display table on the right
df_summary = df_test.groupby('output_12').count
print(df_summary)

tabledata = plotter.subplot(grid[2])
tabledata.axis('tight')
tabledata.axis('off')
table = tabledata.table(cellText=df_test['output_12'].mean, colLabels=df_test['output_12'] ,cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(12)
fig.colorbar(hb4, ax=plot_hex4)

# hb_ex = plot_hex_test.hexbin(df_example['q3'], df_example['label'], gridsize=20, cmap='Greens', edgecolors='grey', linewidths=0.1)
# plot_hex_test.set_title('Test Sample: q3 vs Grieving Stage', fontsize=9)
# plot_hex_test.set_xlabel('q3 Answer', fontsize=9)
# plot_hex_test.set_ylabel('Grieving Stage', fontsize=9)

# Adding a color bar to show the color scale
# fig.colorbar(hb1, ax=plot_hex1)
# fig.colorbar(hb2, ax=plot_hex2)


# fig.colorbar(hb_ex, ax=plot_hex_test)


# Function to add text labels to hexbin cells
def add_labels(ax, hb):
    offsets = hb.get_offsets()
    values = hb.get_array()
    for offset, value in zip(offsets, values):
        if value > 0:
            ax.text(offset[0], offset[1], int(value), color='red', ha='center', va='center', fontsize=7)

# add_labels(plot_hex1,hb1)
# add_labels(plot_hex2,hb2)
add_labels(plot_hex3, hb3)
add_labels(plot_hex4, hb4)
# add_labels(plot_hex_test, hb_ex)




plotter.show()
