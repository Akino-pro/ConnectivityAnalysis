import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def spherical_to_cartesian(theta, phi):
    """
    Convert spherical angles (theta, phi) to 3D Cartesian coordinates (x, y, z).
    Here:
      - theta in [0, 2π) is the azimuth angle,
      - phi   in [0,   π] is the polar angle from the +z axis.
    """
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return x, y, z

# --- 1) Choose a single 'base' point on the sphere ---
theta0 = 1.2       # some azimuth
phi0   = 0.8       # some colatitude

# --- 2) Define the "shift": theta ∈ [theta0 + 1, theta0 + 2] ---
N = 100  # number of sample points to plot the arc
thetas_arc = np.linspace(theta0 + 1, theta0 + 2, N)

# Compute Cartesian coords of that arc at constant phi = phi0
arc_x, arc_y, arc_z = [], [], []
for t in thetas_arc:
    x, y, z = spherical_to_cartesian(t, phi0)
    arc_x.append(x)
    arc_y.append(y)
    arc_z.append(z)

# --- 3) (Optional) Create a sphere mesh to visualize reference ---
phi_s    = np.linspace(0, np.pi, 50)       # for the mesh
theta_s  = np.linspace(0, 2*np.pi, 50)
phi_s, theta_s = np.meshgrid(phi_s, theta_s)
x_s = np.sin(phi_s) * np.cos(theta_s)
y_s = np.sin(phi_s) * np.sin(theta_s)
z_s = np.cos(phi_s)

# --- 4) Plot the sphere and the arc ---
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Light gray sphere surface
ax.plot_surface(x_s, y_s, z_s, color='gray', alpha=0.2, linewidth=0)

# The shifted arc
ax.plot(arc_x, arc_y, arc_z, 'r-', lw=3, label='theta ∈ [theta0+1, theta0+2]')

# Original point (theta0, phi0)
x0, y0, z0 = spherical_to_cartesian(theta0, phi0)
ax.scatter([x0], [y0], [z0], color='blue', s=50, label='Original point')

# Some cosmetic touches
ax.set_xlim([-1,1]); ax.set_ylim([-1,1]); ax.set_zlim([-1,1])
ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.legend()
plt.show()
