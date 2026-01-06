import numpy as np
from matplotlib import pyplot as plt
from roboticstoolbox import DHRobot, RevoluteDH

from spatial_7r_FTW_estimation_ssm import zyz_to_R, convert_to_C_dot_A, compute_beta_range

alpha = [-62 * np.pi / 180, -79 * np.pi / 180, 90 * np.pi / 180, 29 * np.pi / 180, 81 * np.pi / 180, -80 * np.pi / 180,0]#was -90*np.pi/180
l = [0.4, 0.8, 0.2, 1, 0.6, 0.4, 0]#last one was 0.2
d = [-0.4, -0.6, 0.2, 0.6, -0.8, 0.2, 0]# was 0.8
CA = [(-107 * np.pi / 180, 107 * np.pi / 180), (-164 * np.pi / 180, 141 * np.pi / 180),
      (-132 * np.pi / 180, 132 * np.pi / 180), (-151 * np.pi / 180, 102 * np.pi / 180),
      (-115 * np.pi / 180, 149 * np.pi / 180), (-75 * np.pi / 180, 129 * np.pi / 180),
      (16 * np.pi / 180, 193 * np.pi / 180)]

alpha = [-np.pi/2, np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, np.pi/2,0]
l = [0, 0, 0, 0, 0, 0, 0]
d = [0.2848, -0.0118, 0.4208, -0.0128, 0.3143, 0, 0]
CA = [(-180 * np.pi / 180, 180 * np.pi / 180), (-180 * np.pi / 180, 180 * np.pi / 180),
       (-180 * np.pi / 180, 180 * np.pi / 180), (-180 * np.pi / 180, 180 * np.pi / 180),
       (-180 * np.pi / 180, 180 * np.pi / 180), (-180 * np.pi / 180, 180 * np.pi / 180),
       (-180 * np.pi / 180, 180 * np.pi / 180)]



C_dot_A = CA.copy()
C_dot_A[0] = (-np.pi, np.pi)
C_dot_A = convert_to_C_dot_A(C_dot_A)
robot = DHRobot(
        [
            RevoluteDH(d=d[0], alpha=alpha[0], a=l[0], qlim=CA[0]),
            RevoluteDH(d=d[1], alpha=alpha[1], a=l[1], qlim=CA[1]),
            RevoluteDH(d=d[2], alpha=alpha[2], a=l[2], qlim=CA[2]),
            RevoluteDH(d=d[3], alpha=alpha[3], a=l[3], qlim=CA[3]),
            RevoluteDH(d=d[4], alpha=alpha[4], a=l[4], qlim=CA[4]),
            RevoluteDH(d=d[5], alpha=alpha[5], a=l[5], qlim=CA[5]),
            RevoluteDH(d=d[6], alpha=alpha[6], a=l[6], qlim=CA[6]),
        ], name="spatial 7R")


#position1=[ 0.55774151, 0, -2.78870757]
#position2=[0.48292833, 0.27887076, -2.78870757]
#orientation1=[5.582885474598648, -0.032263661668246936]
#orientation2=[5.582885474598648+30*np.pi/180, -0.032263661668246936]
#target_x = np.array([position1[0],position1[1], position1[2]]).T.reshape((3, 1))
#sampled_orientation = zyz_to_R(orientation1[0], orientation1[1], 0)
#beta_ranges, alpha_ranges,reliability,F_list = compute_beta_range(sampled_orientation, target_x, robot, C_dot_A, CA)

#target_x = np.array([position2[0],position2[1], position2[2]]).T.reshape((3, 1))
#sampled_orientation = zyz_to_R(orientation2[0], orientation2[1], 0)
#beta_ranges, alpha_ranges,reliability,F_list = compute_beta_range(sampled_orientation, target_x, robot, C_dot_A, CA)

#position1=[0.261125, 0, 0.261125]
#position2=[0.22618496, 0.13056250, 0.261125]
#orientation1=[3.5330721611604616, 0.4327345342669689,0]
#orientation2=[3.5330721611604616+30*np.pi/180, 0.4327345342669689,0]

#target_x = np.array([position1[0],position1[1], position1[2]]).T.reshape((3, 1))
#sampled_orientation = zyz_to_R(orientation1[0], orientation1[1], 0)
#beta_ranges, alpha_ranges,reliability,F_list = compute_beta_range(sampled_orientation, target_x, robot, C_dot_A, CA)
#target_x = np.array([position2[0],position2[1], position2[2]]).T.reshape((3, 1))
#sampled_orientation = zyz_to_R(orientation2[0], orientation2[1], 0)
#beta_ranges, alpha_ranges,reliability,F_list = compute_beta_range(sampled_orientation, target_x, robot, C_dot_A, CA)
#print(orientation1[0]*180/np.pi)
#print(orientation1[1]*180/np.pi)
#print(orientation2[0]*180/np.pi)
#print(orientation2[1]*180/np.pi)




points1 = np.load("ssm_theta_points1.npy")
points2 = np.load("ssm_theta_points2.npy")

plt.figure()
plt.scatter(points1[:, 0], points1[:, 1], c='r', marker='o')
plt.scatter(points2[:, 0], points2[:, 1], c='b', marker='o')

# Set labels
plt.xlabel('theta1')
plt.ylabel('theta2')

# Set plot limits for better visualization
plt.xlim([-np.pi, np.pi])
plt.ylim([-np.pi, np.pi])

plt.show()

plt.figure()
plt.scatter(points1[:, 2], points1[:, 3], c='r', marker='o')
plt.scatter(points2[:, 2], points2[:, 3], c='b', marker='o')

# Set labels
plt.xlabel('theta3')
plt.ylabel('theta4')

# Set plot limits for better visualization
plt.xlim([-np.pi, np.pi])
plt.ylim([-np.pi, np.pi])

plt.show()

plt.figure()
plt.scatter(points1[:, 4], points1[:, 5], c='r', marker='o')
plt.scatter(points2[:, 4], points2[:, 5], c='b', marker='o')

# Set labels
plt.xlabel('theta5')
plt.ylabel('theta6')

# Set plot limits for better visualization
plt.xlim([-np.pi, np.pi])
plt.ylim([-np.pi, np.pi])

plt.show()

plt.figure()
plt.scatter(points1[:, 6], points1[:, 0], c='r', marker='o')
plt.scatter(points2[:, 6], points2[:, 0], c='b', marker='o')

# Set labels
plt.xlabel('theta7')
plt.ylabel('theta1')

# Set plot limits for better visualization
plt.xlim([-np.pi, np.pi])
plt.ylim([-np.pi, np.pi])

plt.show()
