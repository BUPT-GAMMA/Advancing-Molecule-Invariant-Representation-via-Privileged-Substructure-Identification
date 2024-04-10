import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import MultipleLocator

# Set larger font sizes for the title and labels
title_fontsize = 28
label_fontsize = 28
ticks_fontsize = 25

inv_risk = [0.000220, 0.000207, 0.000661, 0.001391, 0.000775, 0.000721, 0.000699,
0.001442, 0.001377, 0.001095, 0.000832, 0.000954, 0.001106, 0.002535, 0.003783, 
0.001247, 0.002041, 0.001810, 0.001685, 0.002877, 0.003008, 0.002536, 0.003853, 
0.002812, 0.002740, 0.003598, 0.003937, 0.004642, 0.003509, 0.002428, 0.003914, 
0.003920, 0.003747, 0.004862, 0.003958, 0.004084, 0.004111, 0.004009, 0.005868,
0.004539, 0.005138, 0.004107, 0.006437, 0.004264, 0.005275, 0.004394, 0.004533, 
0.006669, 0.007672, 0.004676, 0.006831, 0.006732, 0.008926, 0.006292, 0.003789
]
auc = [77.81, 76.29, 78.61, 76.62, 77.95, 79.86, 79.30,
80.02, 78.97, 76.55, 75.03, 79.29, 79.64, 78.87, 76.72, 
77.38, 79.26, 79.67, 78.25, 76.39, 77.60, 77.54, 79.59, 
78.95, 80.94, 80.09, 79.66, 80.26, 77.46, 78.36, 77.59, 
82.47, 80.12, 79.46, 78.01, 80.73, 81.15, 82.03, 83.02, 
81.02, 80.75, 79.47, 79.69, 80.14, 78.54, 77.36, 81.67, 
79.66, 80.59, 81.62, 79.43, 78.76, 80.13, 80.02, 80.64, 
]

inv_risk = np.array(inv_risk, dtype = float)
auc = np.array(auc, dtype = float)

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize=(7, 6))
# Color codes for the lines
color_risk = '#0c84c6'
# Plot ACC
plt.plot(inv_risk, auc, 'o', markersize='5', color=color_risk, zorder=100)

# Adding titles and labels
plt.xlabel('Invariant Risk', fontsize=label_fontsize, weight='bold')
plt.ylabel('ROC-AUC (%)', fontsize=label_fontsize, weight='bold')

# Adjust the x-axis to only include the specified ticks
plt.xticks([0, 0.002, 0.004, 0.006, 0.008, 0.010], fontsize=ticks_fontsize, weight='bold')
plt.ylim(73, 85)
ax = plt.gca()
ax.yaxis.set_major_locator(MultipleLocator(2))
ax.yaxis.set_tick_params(labelsize=ticks_fontsize)  # Set y-axis tick label size
ax.set_yticklabels(ax.get_yticks(), weight='bold')
ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.5, zorder=0)
ax.ticklabel_format(style='sci', scilimits=(-3,-2), axis='x')

final_output_path = 'inv_risk_AUC.pdf'
plt.savefig(final_output_path, bbox_inches='tight')  # bbox_inches='tight' is used to fit the plot neatly
