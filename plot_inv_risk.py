import matplotlib.pyplot as plt

# Set larger font sizes for the title and labels
title_fontsize = 28
label_fontsize = 28
ticks_fontsize = 25

# Making further adjustments to the font sizes and formatting the x-axis labels to match the provided image
# Data from the image
# epochs = list(range(1, 41))
epochs = list(range(1, 61))
# risk = [0.000238327, 0.000357099, 0.000464593, 0.000376008, 0.000354203, 0.000476194, 0.001011688, 
# 0.000847932, 0.000850485, 0.000990125, 0.001169409, 0.001503014, 0.001061334, 0.001441571, 
# 0.002833712, 0.001066651, 0.001654243, 0.001231835, 0.001625630, 0.001802248, 0.002786371, 
# 0.002870717, 0.002016007, 0.001636404, 0.002061365, 0.001309483, 0.001999334, 0.003080640, 
# 0.003956227, 0.002403586, 0.002335685, 0.002769697, 0.005465588, 0.004012984, 0.002732816,
# 0.003385539, 0.006993030, 0.003585438, 0.005328962, 0.007797205]
risk = [0.000269558, 0.000412958, 0.000452106, 0.000577621, 0.000356485, 0.000321858, 0.000629163, 
0.000777793, 0.000709975, 0.000546402, 0.000902051, 0.000977744, 0.000761999, 0.000847584, 
0.001369590, 0.000944632, 0.001437078, 0.001722196, 0.001236474, 0.001318087, 0.001711259, 
0.001744428, 0.002104189, 0.001679974, 0.002214289, 0.002355347, 0.003249424, 0.002348231, 
0.002658498, 0.002348231, 0.003138743, 0.002774623, 0.004140058, 0.003441805, 0.003819214, 
0.003192336, 0.002758721, 0.003306549, 0.005384831, 0.004411790, 0.004211982, 0.007093997, 
0.006276169, 0.004648302, 0.005363879, 0.006165723, 0.006063597, 0.005179152, 0.006460291, 
0.006024914, 0.004529621, 0.005080859, 0.005334274, 0.006994905, 0.006415047, 0.006033985, 
0.005847896, 0.005376514, 0.006414601, 0.005526071
]

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize=(7, 6))
# Color codes for the lines
color_risk = '#0c84c6'
# Plot ACC
plt.plot(epochs, risk, 'o-', markersize='4', linewidth=1.7, color=color_risk, zorder=100)

# Adding titles and labels
plt.xlabel('Epoch', fontsize=label_fontsize, weight='bold')
plt.ylabel('Invariant Risk', fontsize=label_fontsize, weight='bold')

# Adjust the x-axis to only include the specified ticks
# plt.xticks([0, 10, 20, 30, 40], fontsize=ticks_fontsize, weight='bold')
plt.xticks([0, 15, 30, 45, 60], fontsize=ticks_fontsize, weight='bold')
plt.ylim(0, 0.01)
plt.yticks([0.002, 0.004, 0.006, 0.008, 0.010], fontsize=ticks_fontsize, weight='bold')
ax = plt.gca()
# y 轴用科学记数法
ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.5, zorder=0)
ax.ticklabel_format(style='sci', scilimits=(-3,-2), axis='y')

# final_output_path = 'BACE_inv_risk.pdf'
final_output_path = 'BACE_inv_risk_rebuttal.pdf'
plt.savefig(final_output_path, bbox_inches='tight')  # bbox_inches='tight' is used to fit the plot neatly
