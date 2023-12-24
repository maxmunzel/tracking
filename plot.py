import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from relative import main
from collections import defaultdict


# Function to update the plot
def update(frame):
    global ax, trace

    marker_id, transform = frame
    marker_id = int(marker_id)
    x, y, z = transform[:3, 3].flatten()

    # Update trace
    trace[marker_id].append((x, y, z))

    # Clear the axes for the updated plot
    ax.cla()
    set_axes_equal(ax)

    # Plot each trace
    for marker_trace in trace.values():
        trace_points = np.array(marker_trace)
        ax.plot(trace_points[:, 0], trace_points[:, 1], trace_points[:, 2], marker="o")

    # Setting labels
    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    ax.set_title("3D Trace of Marker Positions")


# Function to set equal scaling for all axes
def set_axes_equal(ax):
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    ax.set_xlim(limits.min(), limits.max())
    ax.set_ylim(limits.min(), limits.max())
    ax.set_zlim(limits.min(), limits.max())


# Initialize plot
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

# Initialize a dictionary to hold traces of each marker
trace = defaultdict(list)

# Create animation
ani = FuncAnimation(fig, update, frames=main, blit=False, interval=100)

plt.show()
