import orix
# Import core external
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

# Colorisation and visualisation
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.lines import Line2D
from skimage.color import label2rgb

# Import orix classes
from orix import data, plot
from orix.quaternion import Orientation, OrientationRegion, Rotation
from orix.quaternion.symmetry import D6
from orix.vector import AxAngle, Vector3d


plt.rcParams.update(
    {"font.size": 20, "figure.figsize": (10, 10), "figure.facecolor": "w"}
)

euler = np.loadtxt("A01_XY_100_nm.ang", skiprows=1, usecols=(0,1,2))
ori = orix.quaternion.Orientation.from_euler(np.deg2rad(euler),
                                   orix.quaternion.symmetry.D6)
ori = ~ori # Might need to remove this.
ori = ori.reshape(450,450)

ckey = plot.IPFColorKeyTSL(D6)

directions = [(1, 0, 0), (0, 1, 0)]
titles = ["X", "Y"]

fig, axes = plt.subplots(ncols=2, figsize=(15, 10))
for i, ax in enumerate(axes):
    ckey.direction = Vector3d(directions[i])
    # Invert because orix assumes lab2crystal when coloring orientations
    ax.imshow(ckey.orientation2color(~ori))
    ax.set_title(f"IPF-{titles[i]}")
    ax.axis("off")

# Add color key
ax_ipfkey = fig.add_axes(
    [0.932, 0.37, 0.1, 0.1],  # (Left, bottom, width, height)
    projection="ipf",
    symmetry=ori.symmetry.laue,
)
ax_ipfkey.plot_ipf_color_key()
ax_ipfkey.set_title("")
fig.subplots_adjust(wspace=0.01)
plt.show()
plt.clf()

ori = ori.map_into_symmetry_reduced_zone()

D = ori.get_distance_matrix(lazy=True, chunk_size=20)
D = D.reshape(ori.size, ori.size)
# This call will use about 6 GB of memory, but the data precision of
# the D matrix can be reduced from float64 to float32 save memory:
D = D.astype(np.float32)
import pdb;pdb.set_trace()

dbscan = DBSCAN(
    eps=0.05,  # Max. distance between two samples in radians
    min_samples=40,
    metric="precomputed",
).fit(D)

unique_labels, all_cluster_sizes = np.unique(
    dbscan.labels_, return_counts=True
)
print("Labels:", unique_labels)

all_labels = dbscan.labels_.reshape(ori.shape)
n_clusters = unique_labels.size - 1
print("Number of clusters:", n_clusters)
unique_cluster_labels = unique_labels[
    1:
]  # Without the "no-cluster" label -1
cluster_sizes = all_cluster_sizes[1:]

q_mean = [ori[all_labels == l].mean() for l in unique_cluster_labels]
cluster_means = Orientation.stack(q_mean).flatten()

# Map into the fundamental zone
cluster_means.symmetry = D6
cluster_means = cluster_means.map_into_symmetry_reduced_zone()
cluster_means
cluster_means.axis
ori_recentered = (~cluster_means[0]) * ori

# Map into the fundamental zone
ori_recentered.symmetry = D6
ori_recentered = ori_recentered.map_into_symmetry_reduced_zone()

cluster_means_recentered = Orientation.stack(
    [ori_recentered[all_labels == l].mean() for l in unique_cluster_labels]
).flatten()
cluster_means_recentered

cluster_means_recentered_axangle = AxAngle.from_rotation(
    cluster_means_recentered
)
cluster_means_recentered_axangle.axis

cluster_means_recentered_axangle = AxAngle.from_rotation(
    cluster_means_recentered
)
cluster_means_recentered_axangle.axis

colors = []
lines = []

for i, cm in enumerate(cluster_means_recentered_axangle):
    colors.append(to_rgb(f"C{i}"))
    lines.append([(0, 0, 0), tuple(cm.data[0])])
labels_rgb = label2rgb(all_labels, colors=colors, bg_label=-1)

cluster_sizes_scaled = 5000 * cluster_sizes / cluster_sizes.max()
fig, ax = plt.subplots(
    figsize=(5, 5), subplot_kw=dict(projection="ipf", symmetry=D6)
)
ax.scatter(
    cluster_means.axis, c=colors, s=cluster_sizes_scaled, alpha=0.5, ec="k"
)
plt.show()
plt.clf()

wireframe_kwargs = dict(
    color="black", linewidth=0.5, alpha=0.1, rcount=181, ccount=361
)
fig = ori_recentered.scatter(
    projection="axangle",
    wireframe_kwargs=wireframe_kwargs,
    c=labels_rgb.reshape(-1, 3),
    s=1,
    return_figure=True,
)
ax = fig.axes[0]
ax.view_init(elev=90, azim=-30)
ax.add_collection3d(Line3DCollection(lines, colors=colors))

handle_kwds = dict(marker="o", color="none", markersize=10)
handles = []
for i in range(n_clusters):
    line = Line2D(
        [0], [0], label=i + 1, markerfacecolor=colors[i], **handle_kwds
    )
    handles.append(line)
ax.legend(
    handles=handles,
    loc="lower right",
    ncol=2,
    numpoints=1,
    labelspacing=0.15,
    columnspacing=0.15,
    handletextpad=0.05,
);
plt.show()
plt.clf()

fig2 = ori_recentered.scatter(
    return_figure=True,
    wireframe_kwargs=wireframe_kwargs,
    c=labels_rgb.reshape(-1, 3),
    s=1,
)
ax2 = fig2.axes[0]
ax2.add_collection3d(Line3DCollection(lines, colors=colors))
ax2.view_init(elev=0, azim=-30)
plt.show()
plt.clf()


fig3, ax3 = plt.subplots(figsize=(15, 10))
ax3.imshow(labels_rgb)
ax3.axis("off");
plt.show()
plt.clf()


import pdb;pdb.set_trace()
