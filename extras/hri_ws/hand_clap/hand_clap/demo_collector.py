import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory
import csv
from datetime import datetime
import os
import ast

class DemoCollector(Node):

    def __init__(self):
        super().__init__('demo_collector')

        self.declare_parameter('traj_dir', '/home/ingui/hri_ws/src/trajectory_utils/trajectories')
        self.declare_parameter('topics', "['/left/joint_trajectory', '/right/joint_trajectory']")

        traj_dir = self.get_parameter('traj_dir').get_parameter_value().string_value
        topics_param = self.get_parameter('topics').get_parameter_value().string_value

        hand_topics = ['/left_hand_joints', '/right_hand_joints']

        topics = ast.literal_eval(topics_param) if isinstance(topics_param, str) else topics_param

        self.subs = []
        self.csv_initialized = {}

        for topic in topics:
            filename = os.path.join(traj_dir, topic.replace('/', '_')[1:] + '_')
            self.csv_initialized[filename] = False

            self.subs.append(
                self.create_subscription(
                    JointTrajectory,
                    topic,
                    lambda msg, fname=filename: self.listener_callback(fname, msg),
                    10
                )
            )

        for topic in hand_topics:
            filename = os.path.join(traj_dir, topic.replace('/', '_')[1:] + '_')
            self.csv_initialized[filename] = False
            
            self.subs.append(
                self.create_subscription(
                    JointTrajectory,
                    topic,
                    lambda msg, fname=filename: self.hand_listener_callback(fname, msg),
                    10
                )
            )


    def listener_callback(self, filename, msg):
        header = ['stamp_sec', 'stamp_nanosec']
        for joint in msg.joint_names:
            header.append(f'{joint}_position')
            header.append(f'{joint}_velocity')
            header.append(f'{joint}_effort')

        now = datetime.now()
        datetime_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        fname = os.path.join(filename + datetime_string +'.csv')

        file_exists = os.path.exists(fname)
        with open(fname, 'w') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or not self.csv_initialized[fname]:
                writer.writerow(header)
                self.csv_initialized[fname] = True

            stamp_sec = msg.header.stamp.sec
            stamp_nanosec = msg.header.stamp.nanosec

            for point in msg.points:
                row = [stamp_sec, stamp_nanosec]
                for i, joint in enumerate(msg.joint_names):
                    pos = point.positions[i] if i < len(point.positions) else ''
                    vel = point.velocities[i] if point.velocities and i < len(point.velocities) else ''
                    eff = point.effort[i] if point.effort and i < len(point.effort) else ''
                    row.extend([pos, vel, eff])
                writer.writerow(row)

    def hand_listener_callback(self, filename, msg):
        header = ['stamp_sec', 'stamp_nanosec']
        for joint in msg.joint_names:
            header.append(joint + "_x")
            header.append(joint + "_y")
            header.append(joint + "_z")

        now = datetime.now()
        datetime_string = now.strftime("%Y_%m_%d_%H_%M_%S")
        fname = os.path.join(filename + datetime_string +'.csv')

        file_exists = os.path.exists(fname)
        with open(fname, 'w') as csvfile:
            writer = csv.writer(csvfile)
            if not file_exists or not self.csv_initialized[fname]:
                writer.writerow(header)
                self.csv_initialized[fname] = True

            stamp_sec = msg.header.stamp.sec
            stamp_nanosec = msg.header.stamp.nanosec

            for point in msg.points:
                row = [stamp_sec, stamp_nanosec]
                for i, joint in enumerate(msg.joint_names):
                    base_idx = i * 3
                    pos_x = point.positions[base_idx] if base_idx < len(point.positions) else ''
                    pos_y = point.positions[base_idx + 1] if base_idx + 1 < len(point.positions) else ''
                    pos_z = point.positions[base_idx + 2] if base_idx + 2 < len(point.positions) else ''
                    row.extend([pos_x, pos_y, pos_z])
                writer.writerow(row)

def main(args=None):
    rclpy.init(args=args)
    node = DemoCollector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
