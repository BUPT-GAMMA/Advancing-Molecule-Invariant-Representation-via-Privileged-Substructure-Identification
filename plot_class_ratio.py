import matplotlib.pyplot as plt

# Set larger font sizes for the title and labels
title_fontsize = 28
label_fontsize = 28
ticks_fontsize = 25

# Making further adjustments to the font sizes and formatting the x-axis labels to match the provided image
# Data from the image
epochs = list(range(1, 41))
e1 = [1.385223507, 1.484551906, 1.566297411, 1.495284438, 1.496701598, 1.482465744, 1.508043646, 1.485561966, 1.566736578, 1.467237472, 1.538535952, 1.477090239, 1.503615856, 1.478602766, 1.540892243, 1.587520956, 1.495397448, 1.429993987, 1.532924652, 1.378969669, 1.449372410, 1.487125992, 1.420347213, 1.47296392, 1.544180750, 1.546056747, 1.471893310, 1.594033598, 1.433618545, 1.492808818, 1.486650943, 1.478138923, 1.516334176, 1.477912902, 1.522801756, 1.512963414, 1.456483721, 1.556576251, 1.558608889, 1.644772887]
e2 = [1.978512644, 1.663659453, 1.699406027, 1.793271541, 1.656820893, 1.579101443, 1.594780325, 1.551622033, 1.631871342, 1.523615479, 1.625272274, 1.596112132, 1.658119559, 1.666875839, 1.551031470, 1.535757541, 1.599460959, 1.533766150, 1.593242526, 1.581737041, 1.796356201, 1.770003795, 1.597195267, 1.55971395, 1.785162806, 1.615466117, 1.600865244, 1.537579536, 1.596806883, 1.554331421, 1.620218753, 1.554629206, 1.718675017, 1.542415261, 1.507142901, 1.649378657, 1.640991568, 1.602630019, 1.589350104, 1.586214661]
# e1 = [0.544217169, 0.576298415, 0.469683617, 0.482205390, 0.506035089, 0.470145791, 0.495813727, 0.467951416, 0.442343950, 0.432982891, 0.474079966, 0.580718457, 0.553101241, 0.462694257, 0.460539489, 0.492550492, 0.492213875, 0.429612547, 0.445963472, 0.462250679, 0.465519756, 0.484115749, 0.462915897, 0.441412955, 0.408517807, 0.429387181, 0.479250043, 0.425820499, 0.475500583, 0.437567949, 0.416220545, 0.432449251, 0.467077404, 0.442693084, 0.371926069, 0.410103440, 0.397847324, 0.399531096, 0.343487977, 0.37688931]
# e2 = [1.027758121, 0.874482452, 0.744352340, 0.811773121, 0.631068229, 0.718346297, 0.734462261, 0.657579958, 0.683951675, 0.637753248, 0.642272114, 0.735592544, 0.622947275, 0.660808622, 0.566604673, 0.590415537, 0.606777966, 0.641559302, 0.607198178, 0.591829836, 0.621217668, 0.629417657, 0.570208847, 0.564490497, 0.584188878, 0.664867460, 0.550820052, 0.592548787, 0.629088819, 0.600200414, 0.575314700, 0.632426500, 0.701362311, 0.635639905, 0.612866878, 0.546741008, 0.554608881, 0.486788243, 0.462755709, 0.43255707]

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.figure(figsize=(8, 6))
# Color codes for the lines
color_e1 = '#0c84c6'
color_e2 = '#f74d4d'
# Plot ACC
plt.plot(epochs, e1, 'o-', markersize='4', linewidth=1.7, color=color_e1, zorder=100, label = 'e1')
plt.plot(epochs, e2, 'o-', markersize='4', linewidth=1.7, color=color_e2, zorder=100, label = 'e2')

# Adding titles and labels
plt.xlabel('Epoch', fontsize=label_fontsize, weight='bold')
plt.ylabel('Class Ratio', fontsize=label_fontsize, weight='bold')
# plt.ylabel('Reweighted Empirical Risk', fontsize=label_fontsize, weight='bold')

# Adjust the x-axis to only include the specified ticks
plt.xticks([0, 10, 20, 30, 40], fontsize=ticks_fontsize, weight='bold')
plt.ylim(1.25, 2)
plt.yticks([1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2], fontsize=ticks_fontsize, weight='bold')
# plt.ylim(0.1, 1.2)
# plt.yticks([0.2, 0.4, 0.6, 0.8, 1, 1.2], fontsize=ticks_fontsize, weight='bold')

ax = plt.gca()
ax.yaxis.grid(True, linestyle='-', which='major', color='grey', alpha=.5, zorder=0)
handles, labels = ax.get_legend_handles_labels()
order = [0, 1]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],loc="upper right",prop={'family' : 'Times New Roman', 'size': 25, 'weight':'bold'})

final_output_path = 'BACE_class_ratio.pdf'
# final_output_path = 'BACE_risk.pdf'
plt.savefig(final_output_path, bbox_inches='tight')  # bbox_inches='tight' is used to fit the plot neatly