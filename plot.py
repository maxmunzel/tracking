import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from relative import main
from collections import defaultdict


# Function to update the plot
def update(frame):
    global ax3d, ax2d, trace

    marker_id, transform = frame
    marker_id = int(marker_id)
    x, y, z = transform[:3, 3].flatten()

    # Update trace
    trace[marker_id].append((x, y, z))
    if len(trace[marker_id]) > 10:
        trace[marker_id] = trace[marker_id][1:]

    # Clear the axes for the updated plot
    ax3d.cla()
    ax2d.cla()

    # Plot each trace in 3D
    for marker_trace in reversed(trace.values()):
        trace_points = np.array(marker_trace)
        ax3d.plot(
            trace_points[:, 0], trace_points[:, 1], trace_points[:, 2], marker="o"
        )

    # Plot each trace in 2D
    for marker_trace in reversed(trace.values()):
        trace_points = np.array(marker_trace)
        ax2d.plot(trace_points[:, 0], trace_points[:, 1], marker="o")

    # Setting labels for 3D plot
    ax3d.set_xlabel("X axis")
    ax3d.set_ylabel("Y axis")
    ax3d.set_zlabel("Z axis")
    ax3d.set_title("3D Trace of Marker Positions")

    # Setting labels for 2D plot
    ax2d.set_xlabel("X axis")
    ax2d.set_ylabel("Y axis")
    ax2d.set_ylim(-0.2, 0.2)
    ax2d.set_xlim(-0.2, 0.2)
    ax2d.set_title("2D Trace of Marker Positions")


# Initialize plot with two subplots
fig = plt.figure()
ax3d = fig.add_subplot(121, projection="3d")
ax2d = fig.add_subplot(122)

# Initialize a dictionary to hold traces of each marker
trace = defaultdict(list)

# Create animation
ani = FuncAnimation(fig, update, frames=main, blit=False, interval=100)

plt.show()
