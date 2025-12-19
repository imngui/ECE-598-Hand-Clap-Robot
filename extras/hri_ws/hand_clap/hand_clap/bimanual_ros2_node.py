import mujoco
from mujoco import viewer
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

def quaternion_to_rotation_matrix(quat):
    rotation_matrix = R.from_quat(quat).as_matrix()
    return rotation_matrix

def position_quaternion_to_transform(pos, quat):
    transform = np.eye(4)
    transform[:3, 3] = pos
    transform[:3, :3] = quaternion_to_rotation_matrix(quat)
    return transform

def inverse_transform(transform):
    return np.linalg.inv(transform)

def transform_relative(transform1, transform2):
    return np.dot(inverse_transform(transform1), transform2)


class BimanualSimNode(Node):
    def __init__(self, model_path="./mujoco_menagerie/universal_robots_ur5e/bimanual_scene.xml",
                 loop_callback=None):
        super().__init__('bimanual_sim_node')

        self.m = mujoco.MjModel.from_xml_path(model_path)
        self.d = mujoco.MjData(self.m)

        self.loop_callback = loop_callback if loop_callback is not None else self.default_loop_callback
        self.paused = False
        self.timestep = 0

        self.robot1_controls = np.zeros(6)
        self.robot2_controls = np.zeros(6)

        self.joint_names = [
            'shoulder_pan_joint',
            'shoulder_lift_joint',
            'elbow_joint',
            'wrist_1_joint',
            'wrist_2_joint',
            'wrist_3_joint'
        ]

        self.left_joint_pub = self.create_publisher(JointState, '/left_joint_states', 10)
        self.right_joint_pub = self.create_publisher(JointState, '/right_joint_states', 10)

        self.left_control_sub = self.create_subscription(
            JointState,
            '/left_joint_commands',
            self.left_control_callback,
            10
        )
        self.right_control_sub = self.create_subscription(
            JointState,
            '/right_joint_commands',
            self.right_control_callback,
            10
        )

        self.joint_state_timer = self.create_timer(0.01, self.publish_joint_states)

    def loop(self):
        def key_callback(keycode):
            if chr(keycode) == ' ':
                self.paused = not self.paused
                state = "Paused" if self.paused else "Running"

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
            self.robot1_controls = np.array(controls)
        elif robot == 2:
            self.robot2_controls = np.array(controls)

    def update_robot_joints(self, robot, joint_angles):
        if robot == 1:
            self.robot1_controls = np.array(joint_angles)
        elif robot == 2:
            self.robot2_controls = np.array(joint_angles)
        self.d.ctrl[:] = np.concatenate([self.robot1_controls, self.robot2_controls])

    def set_robot_control_by_ee(self, robot, ee_transform, make_relative=False):
        from ur_ikfast import ur_kinematics

        ur5e_arm = ur_kinematics.URKinematics('ur5e')
        ee_trans = ee_transform

        if make_relative:
            base_trans = self.get_base_transform(robot)
            ee_trans = transform_relative(base_trans, ee_trans)

        joint_configs = ur5e_arm.inverse(ee_trans[:-1, :], False)

        if robot == 1:
            self.robot1_controls = joint_configs
        elif robot == 2:
            self.robot2_controls = joint_configs

    def get_base_transform(self, robot):
        if robot == 1:
            body = self.m.body('arm1_base')
        elif robot == 2:
            body = self.m.body('arm2_base')
        return position_quaternion_to_transform(body.pos, body.quat)

    def left_control_callback(self, msg):
        if len(msg.position) == 6:
            self.robot1_controls = np.array(msg.position)
            self.d.ctrl[:6] = self.robot1_controls

    def right_control_callback(self, msg):
        if len(msg.position) == 6:
            self.robot2_controls = np.array(msg.position)
            self.d.ctrl[6:12] = self.robot2_controls

    def publish_joint_states(self):
        current_time = self.get_clock().now().to_msg()

        left_positions = self.d.qpos[:6].tolist()
        left_velocities = self.d.qvel[:6].tolist()

        right_positions = self.d.qpos[6:12].tolist()
        right_velocities = self.d.qvel[6:12].tolist()

        left_msg = JointState()
        left_msg.header.stamp = current_time
        left_msg.header.frame_id = 'left_base_link'
        left_msg.name = self.joint_names
        left_msg.position = left_positions
        left_msg.velocity = left_velocities
        left_msg.effort = []
        self.left_joint_pub.publish(left_msg)

        right_msg = JointState()
        right_msg.header.stamp = current_time
        right_msg.header.frame_id = 'right_base_link'
        right_msg.name = self.joint_names
        right_msg.position = right_positions
        right_msg.velocity = right_velocities
        right_msg.effort = []
        self.right_joint_pub.publish(right_msg)


def main(args=None):
    rclpy.init(args=args)

    sim_node = BimanualSimNode()

    ros_thread = threading.Thread(target=lambda: rclpy.spin(sim_node), daemon=True)
    ros_thread.start()

    try:
        sim_node.loop()
    except KeyboardInterrupt:
        sim_node.get_logger().info('Keyboard interrupt received')
    finally:
        sim_node.get_logger().info('Shutting down')
        sim_node.destroy_node()
        rclpy.shutdown()
        ros_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()
