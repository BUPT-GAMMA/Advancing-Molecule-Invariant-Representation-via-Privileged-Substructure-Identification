import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.ticker import FixedLocator

fig = plt.figure(figsize=(9, 11))
ax = fig.add_subplot(111, projection='3d')
# plt.style.use('seaborn-colorblind')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
tmp_planes = ax.zaxis._PLANES
ax.zaxis._PLANES = (tmp_planes[2], tmp_planes[3],
                    tmp_planes[0], tmp_planes[1],
                    tmp_planes[4], tmp_planes[5])
view_1 = (25, 25)  # 25, 25
init_view = view_1
ax.view_init(*init_view)
X = np.array([1, 2, 3, 4, 5])
Y = np.array([1, 2, 3, 4, 5])
X, Y = np.meshgrid(X, Y)

Z = np.array([
    [84.49, 79.69, 80.94, 81.22, 82.53],
    [83.62, 80.09, 80.14, 85.39, 81.31],
    [83.05, 81.21, 82.38, 86.89, 81.28],
    [84.18, 83.90, 87.89, 82.24, 83.64],
    [85.06, 83.63, 85.60, 86.19, 82.92]
])

plt.yticks([1, 2, 3, 4, 5], ["2", "3", "4", "5", "6"], weight='bold')
plt.xticks([1, 2, 3, 4, 5], ["2", "3", "4", "5", "6"], weight='bold')
ax.set_zlim(75, 88)

surf = ax.plot_surface(X, Y, Z, cstride=1, rstride=1, color=(1, 1, 1, 1), cmap=plt.get_cmap('rainbow'),
                       linewidth=0, vmin=80, vmax=86)
ax.contour(X, Y, Z, zdir = 'z', offset = 75, cmap = plt.get_cmap('rainbow'))

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

ax.zaxis.set_major_locator(LinearLocator(6))  # +1
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))

font1 = {
    'size': 28,
    'rotation': 90,
    'weight': 'bold'
}

cb = plt.colorbar(surf, shrink=1, fraction=0.03, aspect=15, pad=0.05)
for l in cb.ax.yaxis.get_ticklabels():
    l.set_family('Times New Roman')
    l.set_size(25)

plt.xlabel(r'$l_2$', labelpad=16, fontdict={'size': 28})
plt.ylabel(r'$l_1$', labelpad=16, fontdict={'size': 28})
plt.xticks(size=25)
plt.yticks(size=25)
ax.tick_params(axis='z', labelsize=25)

# 指定z轴的刻度位置
ax.zaxis.set_major_locator(FixedLocator([76, 78, 80, 82, 84, 86, 88]))

# 设置z轴的刻度标签
ax.set_zticklabels([76, 78, 80, 82, 84, 86, 88], fontsize=25, weight='bold')
ax.zaxis.set_rotate_label(False)
ax.set_zlabel('ROC-AUC (%)', font1, labelpad=12)

final_output_path = 'BACE_layers.pdf'
plt.savefig(final_output_path, bbox_inches='tight')  # bbox_inches='tight' is used to fit the plot neatly