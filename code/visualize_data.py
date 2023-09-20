import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


mesh_info = pd.read_csv("data/mesh_info.csv")

# Read number of vertices and faces
data = []
for _, m in mesh_info.iterrows():
    data.append([m["Vertices"], m["Faces"]])

# Convert to numpy array and split data 
data = np.array(data)
vertices, faces = zip(*data)

# Boxplot showing number of vertices
plt.boxplot(vertices)
plt.savefig("figures/boxplot_vertices.png")

# Boxplot showing number of faces
plt.boxplot(faces)
plt.savefig("figures/boxplot_faces.png")

# Min and max number of faces and vertices
print(min(vertices), min(faces))
print(max(vertices), max(faces))

# Create a 3D histogram using the hist function
hist, xedges, yedges = np.histogram2d(vertices, faces, bins=(15, 15), range=[[0, 100000], [0, 140000]])

# Construct arrays for the X and Y positions (vertices and faces respectively)
xpos, ypos = np.meshgrid(xedges[:-1] + 0.5 * np.diff(xedges), yedges[:-1] + 0.5 * np.diff(yedges))

# Flatten the histogram values and convert to numpy arrays
xpos = xpos.flatten()
ypos = ypos.flatten()
zpos = np.zeros_like(xpos)
dx = dy = np.diff(xedges)[0] * np.ones_like(zpos)
dy = dy * np.ones_like(zpos)
dz = hist.flatten()

# Create a figure showing the histogram
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.bar3d(xpos, ypos, zpos, dx, dy, dz, shade=True)
ax.set_xlabel("Vertices")
ax.set_ylabel("Faces")
ax.set_zlabel("Frequency")
ax.set_title("3D mesh histogram")

plt.savefig("figures/3D_histogram.png")
