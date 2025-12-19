import mujoco
from mujoco import viewer
import numpy as np
import pandas as pd
from ur_ikfast import ur_kinematics
from utils.transform_utils import position_quaternion_to_transform, transform_relative
from ur_ikfast import ur_kinematics
import pickle
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from builtin_interfaces.msg import Time

import os
from ament_index_python.packages import get_package_share_directory

model_path = os.path.join("~/hri_ws/src/hand_clap/third_party/mujoco_menagerie/universal_robots_ur5e/bimanual_scene.xml")

class BimanualSim(Node):
    def __init__(self, model_path=model_path, loop_callback=None):
        super().__init__('bimanual_sim_node')

        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)
        self.loop_callback = loop_callback if loop_callback is not None else self.default_loop_callback
        self.paused = False
        self.ee_trajectory = None
        self.robot1_controls = np.zeros(6)
        self.robot2_controls = np.zeros(6)
        self.timestep = 0

        self.left_joint_pub = self.create_publisher(JointState, '/left_joint_states', 10)
        self.right_joint_pub = self.create_publisher(JointState, '/right_joint_states', 10)

        self.joint_state_timer = self.create_timer(0.01, self.publish_joint_states)

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

    def loop(self):
        def key_callback(keycode):
            if chr(keycode) == ' ':
                self.paused = not self.paused

        with viewer.launch_passive(self.m, self.d, key_callback=key_callback) as v:
            while v.is_running():
                if not self.paused:
                    self.loop_callback()
                    mujoco.mj_step(self.m, self.d)
                    v.sync()

    def default_loop_callback(self):
        self.d.ctrl[:] = np.concatenate([self.robot1_controls, self.robot2_controls])

    def set_robot_control(self, robot, controls):
        if robot == 1:
            self.robot1_controls = controls
        elif robot == 2:
            self.robot2_controls = controls

    def update_robot_joints(self, robot, joint_angles):
        if robot == 1:
            self.robot1_controls = joint_angles
        elif robot == 2:
            self.robot2_controls = joint_angles
        self.d.ctrl[:] = np.concatenate([self.robot1_controls, self.robot2_controls])

    def set_robot_control_by_ee(self, robot, ee_pos, ee_quat, make_relative=False):
        ur5e_arm = ur_kinematics.URKinematics('ur5e')
        ee_trans = position_quaternion_to_transform(ee_pos, ee_quat)

        if make_relative:
            base_trans = self.get_base_transform(robot)
        ee_trans = transform_relative(base_trans, ee_trans)

        joint_configs = ur5e_arm.inverse(ee_trans[:-1,:], False)

        if robot == 1:
            self.robot1_controls[:-1] = joint_configs
        elif robot == 2:
            self.robot2_controls[:-1] = joint_configs

    def set_robot_control_by_ee(self, robot, ee_transform, make_relative=False):
        ur5e_arm = ur_kinematics.URKinematics('ur5e')

        ee_trans = ee_transform

        if make_relative:
            base_trans = self.get_base_transform(robot)
        ee_trans = transform_relative(base_trans, ee_trans)

        joint_configs = ur5e_arm.inverse(ee_trans[:-1,:], False)

        if robot == 1:
            self.robot1_controls = joint_configs
        elif robot == 2:
            self.robot2_controls = joint_configs

    def set_gripper(self, robot, ee_angle):
        if robot == 1:
            self.robot1_controls[-1] = ee_angle
        elif robot == 2:
            self.robot2_controls[-1] = ee_angle

    def get_base_transform(self, robot):
        if robot == 1:
            body = self.m.body('arm1_base')
        elif robot == 2:
            body = self.m.body('arm2_base')
        return position_quaternion_to_transform(body.pos, body.quat)

    def publish_joint_states(self):
        current_time = self.get_clock().now().to_msg()

        left_msg = JointState()
        left_msg.header.stamp = current_time
        left_msg.header.frame_id = 'left_base_link'
        left_msg.name = self.joint_names
        left_msg.position = self.robot1_controls.tolist()
        left_msg.velocity = []
        left_msg.effort = []
        self.left_joint_pub.publish(left_msg)

        right_msg = JointState()
        right_msg.header.stamp = current_time
        right_msg.header.frame_id = 'right_base_link'
        right_msg.name = self.joint_names
        right_msg.position = self.robot2_controls.tolist()
        right_msg.velocity = []
        right_msg.effort = []
        self.right_joint_pub.publish(right_msg)

def get_joint_angles(data, idx):
  shoulder = data.iloc[idx]['shoulder_joint_position']
  upper_arm = data.iloc[idx]['upper_arm_joint_position']
  forearm = data.iloc[idx]['forearm_joint_position'] 
  wrist_1 = data.iloc[idx]['wrist_1_joint_position']
  wrist_2 = data.iloc[idx]['wrist_2_joint_position']
  wrist_3 = data.iloc[idx]['wrist_3_joint_position']
  return -1.0 * np.deg2rad(np.array([shoulder, upper_arm, forearm, wrist_1, wrist_2, wrist_3]))

import threading

def main():
    global sim, left_traj, right_traj, cnt, last_time, tf2, forward_kinematics

    rclpy.init()

    n = 7
    folder_path =f"data/new_data/double/demo_{n}/"

    left_traj = pd.read_csv(folder_path + f"left_joint_trajectory_demo_{n}.csv")
    right_traj = pd.read_csv(folder_path + f"right_joint_trajectory_demo_{n}.csv")
    columns = left_traj.columns.str.contains('position')
    
    left_traj = left_traj.loc[:, columns]
    right_traj = right_traj.loc[:, columns]

    cnt = 0

    last_time = None

    def loop_callback():
        global sim, left_traj, right_traj, cnt, last_time

        current_time = time.time()
        if last_time is not None and (current_time - last_time) < 0.1:
            return

        last_time = current_time

        sim.robot1_controls = get_joint_angles(left_traj, cnt)
        sim.robot2_controls = get_joint_angles(right_traj, cnt)
        sim.d.ctrl[:] = np.concatenate([sim.robot2_controls, sim.robot1_controls])

        cnt += 1
        if cnt >= left_traj.shape[0]:
            cnt = 0
        pass

    # tf2 = np.array([
    #     [1, 0, 0, -0.5],
    #     [0, 1, 0, 0.0],
    #     [0, 0, 1, 0.0],
    #     [0, 0, 0, 1]
    # ])

    # def forward_kinematics(joint_angles, full_transform=False, base_transform=None):
    #     ur5e_arm = ur_kinematics.URKinematics('ur5e')
    #     fk = ur5e_arm.forward(joint_angles[:6], 'matrix')
    #     fk = np.asarray(fk).reshape(3, 4)  # 3x4 rigid transformation matrix

    #     # Convert to 4x4 homogeneous transformation
    #     T = np.eye(4)
    #     T[:3, :] = fk

    #     # Apply base transform if provided
    #     if base_transform is not None:
    #         T = base_transform @ T

    #     if full_transform:
    #         return T

    #     # Extract position and orientation (roll, pitch, yaw) from the final transform
    #     x, y, z = T[:3, 3]
    #     roll = np.arctan2(T[2, 1], T[2, 2])
    #     pitch = np.arctan2(-T[2, 0], np.sqrt(T[2, 1]**2 + T[2, 2]**2))
    #     yaw = np.arctan2(T[1, 0], T[0, 0])
    #     return (x, y, z, roll, pitch, yaw)

    # def loop_callback():
    #     global sim, left_traj, right_traj, cnt, last_time, tf2, forward_kinematics
    #     # ik testing

    #     # Rate limiting based on dt
    #     current_time = time.time()
    #     if last_time is not None and (current_time - last_time) < 0.1:
    #         return

    #     last_time = current_time

    #     sim.robot1_controls = get_joint_angles(left_traj, cnt)

    #     ee_transform = forward_kinematics(sim.robot1_controls, full_transform=True, base_transform=tf2)

    #     sim.set_robot_control_by_ee(2, ee_transform, make_relative=True)
    #     # sim.robot2_controls = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    #     sim.d.ctrl[:] = np.concatenate([sim.robot1_controls, sim.robot2_controls])
    #     cnt += 1
    #     if cnt >= left_traj.shape[0]:
    #         cnt = 0
    #     pass

    sim = BimanualSim(loop_callback=loop_callback)

    ros_thread = threading.Thread(target=lambda: rclpy.spin(sim), daemon=True)
    ros_thread.start()

    try:
        sim.loop()
    except KeyboardInterrupt:
        pass
    finally:
        sim.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()