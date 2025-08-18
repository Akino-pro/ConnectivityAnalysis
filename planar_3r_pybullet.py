# planar_3r_velocity_traj.py
import os, time, math, tempfile
import pybullet as p
import pybullet_data

# ---------- 1) Define your velocity trajectory ----------
# Each item: ( [q1_dot, q2_dot, q3_dot] (rad/s), duration_seconds )
JOINT_VEL_TRAJ = [
    ([ math.radians(10),  math.radians(0),   math.radians(0)],  2.0),  # +30°/s on joint1 for 2s
    ([ math.radians(0),   math.radians(-20), math.radians(0)],  2.0),  # -20°/s on joint2 for 2s
    ([ math.radians(0),   math.radians(0),   math.radians(40)], 2.0),  # +40°/s on joint3 for 2s
    ([ math.radians(-10), math.radians(20),  math.radians(-40)], 2.0), # reverse for 2s
    ([0.0, 0.0, 0.0], 1.0),  # brief stop
]
REPEAT_TRAJ = False  # play forever

# Optional safety/comfort limits
MAX_VEL = math.radians(90)       # per-joint |velocity| cap (rad/s)
MAX_ACCEL = math.radians(360)    # per-joint |accel| cap (rad/s^2), used for smoothing velocity changes

URDF = f"""
<?xml version="1.0" ?>
<robot name="planar3r">
  <!-- base -->
  <link name="base"/>

  <!-- link1 -->
  <link name="link1">
    <inertial><origin xyz="0.5 0 0"/><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01"/></inertial>
    <visual>
      <origin xyz="0.5 0 0"/>
      <geometry><box size="1 0.05 0.05"/></geometry>
      <material name="red"><color rgba="0.85 0.2 0.2 1"/></material>
    </visual>
    <collision><origin xyz="0.5 0 0"/><geometry><box size="1 0.05 0.05"/></geometry></collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base"/>
    <child  link="link1"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.31778" upper="0.31778" effort="100" velocity="3"/>
  </joint>

  <!-- link2 -->
  <link name="link2">
    <inertial><origin xyz="0.5 0 0"/><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01"/></inertial>
    <visual>
      <origin xyz="0.5 0 0"/>
      <geometry><box size="1 0.05 0.05"/></geometry>
      <material name="green"><color rgba="0.2 0.8 0.2 1"/></material>
    </visual>
    <collision><origin xyz="0.5 0 0"/><geometry><box size="1 0.05 0.05"/></geometry></collision>
  </link>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child  link="link2"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.94328" upper="1.943278" effort="100" velocity="3"/>
  </joint>

  <!-- link3 -->
  <link name="link3">
    <inertial><origin xyz="0.5 0 0"/><mass value="1"/><inertia ixx="0.01" iyy="0.01" izz="0.01"/></inertial>
    <visual>
      <origin xyz="0.5 0 0"/>
      <geometry><box size="1 0.05 0.05"/></geometry>
      <material name="blue"><color rgba="0.2 0.2 0.85 1"/></material>
    </visual>
    <collision><origin xyz="0.5 0 0"/><geometry><box size="1 0.05 0.05"/></geometry></collision>
  </link>

  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child  link="link3"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.94328" upper="1.94328" effort="100" velocity="3"/>
  </joint>

  <!-- end-effector marker -->
  <link name="ee">
    <visual>
      <origin xyz="0 0 0"/>
      <geometry><sphere radius="0.06"/></geometry>
      <material name="ee"><color rgba="1 0.8 0 1"/></material>
    </visual>
  </link>
  <joint name="ee_fixed" type="fixed">
    <parent link="link3"/>
    <child  link="ee"/>
    <origin xyz="1 0 0" rpy="0 0 0"/>
  </joint>
</robot>
"""


def clamp(v, lo, hi): return max(lo, min(hi, v))


def slew_towards(current, target, max_dv):
  """Limit per-step delta (accel)."""
  dv = clamp(target - current, -max_dv, max_dv)
  return current + dv


def play_velocity_trajectory(robot, joint_ids, vel_traj,
                             dt=1.0 / 240.0, max_force=50.0,
                             v_limit=MAX_VEL, a_limit=MAX_ACCEL):
  """
  vel_traj: list of (velocities, duration_sec)
            velocities is [q1_dot, q2_dot, q3_dot] in rad/s
  """
  n = len(joint_ids)
  # current commanded velocities (start at zero)
  v_cmd = [0.0] * n
  for (v_target, dur) in vel_traj:
    # sanitize inputs
    if len(v_target) != n:
      raise ValueError("Each velocity waypoint must have 3 values")
    # clamp target velocities
    v_target = [clamp(v, -v_limit, v_limit) for v in v_target]
    steps = max(1, int(dur / dt))
    for _ in range(steps):
      # accelerate towards target with per-step limit
      if a_limit is not None and a_limit > 0:
        max_dv = a_limit * dt
        v_cmd = [slew_towards(vc, vt, max_dv) for vc, vt in zip(v_cmd, v_target)]
      else:
        v_cmd = v_target[:]

      # send velocity control
      p.setJointMotorControlArray(
        robot, joint_ids, p.VELOCITY_CONTROL,
        targetVelocities=v_cmd,
        forces=[max_force] * n
      )
      p.stepSimulation()
      time.sleep(dt)


def main():
  # Write URDF to a temp file
  with tempfile.TemporaryDirectory() as td:
    urdf_path = os.path.join(td, "planar3r.urdf")
    with open(urdf_path, "w") as f:
      f.write(URDF)

    # Start PyBullet
    p.connect(p.GUI)  # close window or Ctrl+C to stop
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    p.resetDebugVisualizerCamera(4.0, 90, -30, [1.5, 0, 0])
    p.setGravity(0, 0, 0)  # planar demo; no gravity
    p.loadURDF("plane.urdf", [0, 0, -0.1])

    robot = p.loadURDF(urdf_path, [0, 0, 0], useFixedBase=True, flags=p.URDF_MERGE_FIXED_LINKS)
    joint_ids = [0, 1, 2]

    # mild damping for stability
    for j in joint_ids:
      p.changeDynamics(robot, j, linearDamping=0.04, angularDamping=0.04)

    dt = 1.0 / 240.0

    try:
      while True:
        play_velocity_trajectory(robot, joint_ids, JOINT_VEL_TRAJ, dt=dt)
        if not REPEAT_TRAJ:
          # stop commanding; hold still
          p.setJointMotorControlArray(robot, joint_ids, p.VELOCITY_CONTROL,
                                      targetVelocities=[0.0] * 3, forces=[50] * 3)
          # keep sim alive
          while True:
            p.stepSimulation()
            time.sleep(dt)
        # else: loop again
    except KeyboardInterrupt:
      pass
    finally:
      try:
        p.disconnect()
      except Exception:
        pass


if __name__ == "__main__":
  main()