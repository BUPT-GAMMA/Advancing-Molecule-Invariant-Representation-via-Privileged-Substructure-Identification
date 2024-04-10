import matplotlib.pyplot as plt

# Set larger font sizes for the title and labels
label_fontsize = 28
ticks_fontsize = 25

# Making further adjustments to the font sizes and formatting the x-axis labels to match the provided image
# Data from the image
# iterations = list(range(1, 7))
iterations = list(range(1, 9))
# aucs = [81.78, 82.21, 84.22, 83.52, 83.20, 85.48]
aucs = [79.51, 82.57, 81.88, 83.10, 85.86, 85.94, 87.89, 84.50]

plt.rcParams['font.sans-serif'] = ['Times New Roman']
fig, ax = plt.subplots(figsize=(10, 6))

plt.plot(iterations, aucs, color='#0c84c6', marker='.', markeredgecolor='#0c84c6', markersize='16', linewidth=2, zorder=100)

# plt.xticks(iterations, fontsize=ticks_fontsize, weight='bold')
plt.xticks(iterations, fontsize=ticks_fontsize, weight='bold')
ax.set_xticklabels([0.1,1,10,50,100,150,200,250], fontsize=ticks_fontsize, weight='bold')
# plt.xlabel('Iteration', fontsize=label_fontsize, weight='bold')
plt.xlabel('$\lambda$', fontsize=label_fontsize, weight='bold')
plt.ylabel('ROC-AUC (%)', fontsize=label_fontsize, weight='bold')
plt.ylim(77, 90)
plt.yticks([78, 80, 82, 84, 86, 88, 90], fontsize=ticks_fontsize, weight='bold')

ax = plt.gca()
ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.5, zorder=0)

# final_output_path = 'BACE_iter.pdf'
final_output_path = 'BACE_lambda.pdf'
plt.savefig(final_output_path, bbox_inches='tight')  # bbox_inches='tight' is used to fit the plot neatly
