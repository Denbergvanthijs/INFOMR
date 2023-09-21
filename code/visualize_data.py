import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Function to calculate outliers (shapes outside of 4th quartile of boxplot)
def calc_outliers(data):
    # Calculate the quartiles and IQR
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    # Define the upper bound for outliers
    upper_bound = q3 + 1.5 * iqr
    print(upper_bound)

    # Count the number of outliers
    outliers = [x for x in data if x > upper_bound]
    num_outliers = len(outliers)
    print(num_outliers)


# Function to create boxplots of number of vertices and faces
def boxplot(mesh_info):
    # Read number of vertices and faces from mesh info
    data = []
    for _, m in mesh_info.iterrows():
        data.append([m["Vertices"], m["Faces"]])

    # Convert to numpy array and split data 
    data = np.array(data)
    vertices, faces = zip(*data)

    # Min and max number of faces and vertices
    # print(min(vertices), min(faces))      16, 16
    # print(max(vertices), max(faces))      98256, 129881

    # Number of outliers (shapes outside of 4th quartile)
    # calc_outliers(vertices)      upper whisker: 15032, num. outliers: 247
    # calc_outliers(faces)         upper whisker: 31986.5, num. outliers: 256

    # Boxplot showing number of vertices
    plt.boxplot(vertices)
    plt.savefig("figures/boxplot_vertices.png")

    # Boxplot showing number of faces
    plt.boxplot(faces)
    plt.savefig("figures/boxplot_faces.png")


# Function to create a 3D histogram of shapes in dataset
def histogram(mesh_info):
    # Read number of vertices and faces from mesh info
    data = []
    for _, m in mesh_info.iterrows():
        data.append([m["Vertices"], m["Faces"]])

    # Convert to numpy array and split data 
    data = np.array(data)
    vertices, faces = zip(*data)

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


# Load mesh info from existing CSV file
mesh_info = pd.read_csv("data/mesh_info.csv")
boxplot(mesh_info)
histogram(mesh_info)