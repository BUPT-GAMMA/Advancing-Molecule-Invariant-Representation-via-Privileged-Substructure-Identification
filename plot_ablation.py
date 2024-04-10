import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Set larger font sizes for the title and labels
title_fontsize = 18
label_fontsize = 18
ticks_fontsize = 15

# Making further adjustments to the font sizes and formatting the x-axis labels to match the provided image
# Data from the image
methods = ['MILI', 'w/o EO', 'w/o RW', 'w/o IO']
# accuracy = [73.52, 72.21, 72.87, 72.31]
accuracy = [87.90, 84.42, 84.19, 83.55]

# Correcting the color codes (removing a character from '0c84c6e' to make it a valid hex)
colors = ['#002c53', '#ffa510', '#0c84c6', '#ffbd66']

# Create the bar graph again with the new font sizes and format changes
fig, ax = plt.subplots(figsize=(6, 6))
ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.5, zorder=0)
bars = ax.bar(methods, accuracy, color=colors, zorder=100)
# Set font properties globally to Times New Roman, which should apply to all text unless overridden
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams["font.weight"] = "bold"
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# Adding the accuracy values on top of the bars with a font size similar to the x and y axis labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2 , yval + 0.2, round(yval, 2), ha='center', va='bottom', fontsize=ticks_fontsize)

# Set title and labels with specified font sizes
# ax.set_title('BBBP', fontsize=title_fontsize, weight='bold')
ax.set_title('BACE', fontsize=title_fontsize, weight='bold')
ax.set_xlabel('')
ax.set_ylabel('ROC-AUC (%)', fontsize=label_fontsize, weight='bold')

# Rotate x-axis labels to match the image and set the font size
# 设置x轴标签，旋转25度，向右对齐，并设置旋转锚点
ax.set_xticklabels(methods, fontsize=label_fontsize, weight='bold')
# ax.set_xticklabels(methods, rotation=25, ha='right', fontsize=label_fontsize, rotation_mode='anchor')
# ax.tick_params(axis='x', direction='out', pad=20)  # 增加pad值以向外移动标签
# ax.tick_params(axis='y', direction='out', pad=20)  # 增加pad值以向上移动标签

# Adjust y-axis to display ticks with the same intervals as in the provided image and set the font size
# ax.set_yticks(range(68, 74, 1))  # Assuming the image has ticks every 2 units from 56 to 70
# ax.set_ylim(70.75, 74.5)  # Set y-axis limit to match the provided image
# ax.yaxis.set_major_locator(MultipleLocator(0.5))
ax.set_ylim(81.5, 89)  # Set y-axis limit to match the provided image
ax.yaxis.set_major_locator(MultipleLocator(1))
ax.yaxis.set_tick_params(labelsize=ticks_fontsize)  # Set y-axis tick label size
ax.set_yticklabels(ax.get_yticks(), weight='bold')
plt.subplots_adjust(bottom=0.3, left=0.145)
# Save the graph with the latest adjustments
# final_output_path = 'BBBP_ablation.pdf'
final_output_path = 'BACE_ablation.pdf'
plt.savefig(final_output_path, bbox_inches='tight')  # bbox_inches='tight' is used to fit the plot neatly