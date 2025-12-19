import os
import numpy as np
import pandas as pd

import torch
from mp_pytorch.mp import MPFactory
from mp_pytorch.util import tensor_linspace


class ProDMPController:
    def __init__(self, num_dof=6, num_basis=20, dt=0.01, tau=1.0):
        """Initialize ProDMP controller.

        Args:
            num_dof: Number of degrees of freedom (6 for UR5e)
            num_basis: Number of basis functions
            dt: Time step for trajectory generation
            tau: Default trajectory duration (normalized to [0, 1] for demos)
        """
        self.num_dof = num_dof
        self.num_basis = num_basis
        self.dt = dt
        self.tau = tau

        # ProDMP configuration
        self.config = {
            "mp_type": "prodmp",
            "num_dof": num_dof,
            "tau": tau,
            "learn_tau": False,  # Disabled due to mp_pytorch bug
            "mp_args": {
                "num_basis": num_basis,
                "basis_bandwidth_factor": 2,
                "num_basis_outside": 0,
                "alpha": 25,
                "alpha_phase": 2,
                "dt": dt,
                "relative_goal": True,
                "auto_scale_basis": True
            }
        }

        self.mp = MPFactory.init_mp(**self.config)

        # Learned parameters from demonstrations
        self.learned_params = None
        self.mean_params = None
        self.params_cov = None

        # Current trajectory state
        self.current_time = 0.0
        self.current_pos = None
        self.current_vel = None
        self.trajectory_times = None
        self.trajectory_pos = None
        self.trajectory_vel = None
        self.current_step = 0

    def load_demos_from_folder(self, demo_folder_path, is_left=True, drop=0):
        robot = "left" if is_left else "right"

        demo_subdirs = sorted([d for d in os.listdir(demo_folder_path)
                              if os.path.isdir(os.path.join(demo_folder_path, d)) and d.startswith("demo_")])

        all_demos = []

        if demo_subdirs:
            # Load from subdirectories
            for subdir in demo_subdirs:
                subdir_path = os.path.join(demo_folder_path, subdir)
                demo_files = [f for f in os.listdir(subdir_path)
                             if f.startswith(f"{robot}_joint_trajectory_demo") and f.endswith(".csv")]

                if demo_files:
                    demo_file = os.path.join(subdir_path, demo_files[0])
                    demo_data = pd.read_csv(demo_file)
                    columns = demo_data.columns.str.contains('position')
                    all_demos.append(demo_data.loc[:, columns].values)
        else:
            demo_files = sorted([f for f in os.listdir(demo_folder_path)
                                if f.startswith(f"{robot}_joint_trajectory_demo") and f.endswith(".csv")])

            for file in demo_files:
                demo_data = pd.read_csv(os.path.join(demo_folder_path, file))
                columns = demo_data.columns.str.contains('position')
                all_demos.append(demo_data.loc[:, columns].values)

        if not all_demos:
            raise ValueError(f"No demos found for {robot} robot in {demo_folder_path}")

        min_len = min(len(d) for d in all_demos)
        all_demos = np.array([d[:min_len] for d in all_demos])

        # Convert from degrees to radians
        all_demos = np.deg2rad(all_demos)

        # Split into train and test sets
        num_demos = len(all_demos)

        if drop > 0:
            np.random.seed(11)
            test_indices = np.random.choice(num_demos, size=drop, replace=False)
            train_indices = np.array([i for i in range(num_demos) if i not in test_indices])

            train_demos = all_demos[train_indices]
            test_demos = all_demos[test_indices]
        else:
            train_demos = all_demos
            test_demos = np.array([])

        return train_demos, test_demos

    def learn_from_demos(self, demos):
        demo_times = tensor_linspace(0.0, 3.0, demos.shape[1]).unsqueeze(0).repeat(demos.shape[0], 1)
        demo_trajs = torch.tensor(demos, dtype=torch.float32)

        # Learn parameters from all demos together
        params_dict = self.mp.learn_mp_params_from_trajs(demo_times, demo_trajs)

        # Store learned parameters
        self.learned_params = params_dict["params"]
        self.learned_init_time = params_dict["init_time"]
        self.learned_init_pos = params_dict["init_pos"]
        self.learned_init_vel = params_dict["init_vel"]

        return params_dict

    def _set_goals_in_params(self, params, goal_relative):
        params = params.clone()
        batch_size = params.shape[0]

        start_idx = 1 if hasattr(self.mp, 'learn_tau') and self.mp.learn_tau else 0

        for dof in range(self.num_dof):
            goal_idx = start_idx + dof * (self.num_basis + 1) + self.num_basis
            params[:, goal_idx] = goal_relative[dof]

        return params

    def create_prodmp_params(self, weights, goals, tau=None):
        batch_size = goals.shape[0]

        if weights.ndim == 2:
            weights = weights.reshape(batch_size, self.num_dof, self.num_basis)

        params_list = []
        for b in range(batch_size):
            param_b = []

            if tau is not None:
                param_b.append(tau[b, 0].item())

            for dof in range(self.num_dof):
                param_b.extend(weights[b, dof, :].tolist())
                param_b.append(goals[b, dof].item())

            params_list.append(param_b)

        return torch.tensor(params_list, dtype=torch.float32)

    def condition_trajectory(self, start_pos, goal_pos, start_vel=None, speed_factor=1.0):
        if start_vel is None:
            start_vel = np.zeros(self.num_dof)

        # Convert to tensors
        init_pos = torch.tensor(start_pos, dtype=torch.float32).unsqueeze(0)
        init_vel = torch.tensor(start_vel, dtype=torch.float32).unsqueeze(0)
        init_time = torch.zeros(1)

        # Calculate relative goal (goal - start)
        goal_relative = torch.tensor(goal_pos - start_pos, dtype=torch.float32).unsqueeze(0)

        # Adjust tau based on speed factor
        adjusted_tau = self.tau / speed_factor

        # Calculate number of timesteps based on adjusted tau
        num_timesteps = int(adjusted_tau / self.dt)
        times = torch.linspace(0, adjusted_tau, num_timesteps).unsqueeze(0)

        # Use learned weights if available, otherwise use zeros (straight-line)
        if self.mean_params is not None:
            # Extract weights from learned mean parameters
            weights = self._extract_weights_from_params(self.mean_params)
        else:
            weights = torch.zeros(1, self.num_dof, self.num_basis)

        # Create parameters with new goal
        params = self.create_prodmp_params(weights, goal_relative)

        # Generate trajectory
        self.mp.update_inputs(
            times=times,
            params=params,
            init_time=init_time,
            init_pos=init_pos,
            init_vel=init_vel
        )

        traj_dict = self.mp.get_trajs(get_pos=True, get_vel=True)

        # Store trajectory state
        self.trajectory_times = times.squeeze(0).numpy()
        self.trajectory_pos = traj_dict["pos"].squeeze(0).numpy()
        self.trajectory_vel = traj_dict["vel"].squeeze(0).numpy()
        self.current_step = 0
        self.current_time = 0.0
        self.current_pos = init_pos.squeeze(0).numpy().copy()
        self.current_vel = init_vel.squeeze(0).numpy().copy()

        return self.trajectory_pos, self.trajectory_vel

    def _extract_weights_from_params(self, params):
        batch_size = params.shape[0]
        weights = torch.zeros(batch_size, self.num_dof, self.num_basis)

        start_idx = 1 if hasattr(self.mp, 'learn_tau') and self.mp.learn_tau else 0

        for dof in range(self.num_dof):
            dof_start = start_idx + dof * (self.num_basis + 1)
            weights[:, dof, :] = params[:, dof_start:dof_start + self.num_basis]

        return weights

    def get_next_step(self, current_robot_pos=None, current_robot_vel=None):
        if self.trajectory_pos is None:
            raise RuntimeError("No trajectory generated. Call condition_trajectory first.")

        # Check if trajectory is complete
        if self.current_step >= len(self.trajectory_pos):
            return self.trajectory_pos[-1], self.trajectory_vel[-1], True

        # Get target for this step
        target_pos = self.trajectory_pos[self.current_step]
        target_vel = self.trajectory_vel[self.current_step]

        # Update current state (could be used for replanning in advanced scenarios)
        if current_robot_pos is not None:
            self.current_pos = current_robot_pos
        else:
            self.current_pos = target_pos.copy()

        if current_robot_vel is not None:
            self.current_vel = current_robot_vel
        else:
            self.current_vel = target_vel.copy()

        # Advance step counter
        self.current_step += 1
        self.current_time = self.current_step * self.dt

        return target_pos, target_vel, False

    def reset(self):
        self.current_step = 0
        self.current_time = 0.0
        self.trajectory_pos = None
        self.trajectory_vel = None


def main():
    demo_folder_path = "data/new_data/"

    # ProDMP parameters
    num_basis = 20
    dt = 0.01  # Time step for trajectory generation
    tau = 3.0  # Trajectory duration

    # Number of trajectories to hold out for testing
    num_holdout = 5


    # Create ProDMP controllers for each gesture
    prodmp_library = {
        'front_five': (ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau),
                      ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau)),
        'left_five': (ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau),
                     ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau)),
        'right_five': (ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau),
                      ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau)),
        'double': (ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau),
                  ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau)),
        'clap': (ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau),
                ProDMPController(num_dof=6, num_basis=num_basis, dt=dt, tau=tau))
    }

    all_results = {}

    for gesture_name, (prodmp_left, prodmp_right) in prodmp_library.items():

        gesture_demo_folder = os.path.join(demo_folder_path, gesture_name)

        if not os.path.exists(gesture_demo_folder):
            continue

        try:
            left_train_demos, left_test_demos = prodmp_left.load_demos_from_folder(
                gesture_demo_folder, is_left=True, drop=num_holdout
            )

            # Learn from training demos
            prodmp_left.learn_from_demos(left_train_demos)

            # Evaluate on held-out test demos
            left_errors = []

            for i, test_demo in enumerate(left_test_demos):
                demo_length = len(test_demo)
                adjusted_tau = demo_length * dt
                adjusted_speed_factor = tau / adjusted_tau

                trajectory_pos, _ = prodmp_left.condition_trajectory(
                    start_pos=test_demo[0],
                    goal_pos=test_demo[-1],
                    start_vel=np.zeros(6),
                    speed_factor=adjusted_speed_factor
                )

                if len(trajectory_pos) > demo_length:
                    trajectory_pos = trajectory_pos[:demo_length]
                elif len(trajectory_pos) < demo_length:
                    padding = np.tile(trajectory_pos[-1], (demo_length - len(trajectory_pos), 1))
                    trajectory_pos = np.vstack([trajectory_pos, padding])

                error = np.abs(trajectory_pos - test_demo).mean()
                left_errors.append(error)

            left_avg_error = np.mean(left_errors)
            left_std_error = np.std(left_errors)

        except Exception as e:
            left_avg_error = None
            left_std_error = None

        try:
            right_train_demos, right_test_demos = prodmp_right.load_demos_from_folder(
                gesture_demo_folder, is_left=False, drop=num_holdout
            )

            print(f"  Training demos: {len(right_train_demos)}")
            print(f"  Test demos: {len(right_test_demos)}")

            # Learn from training demos
            prodmp_right.learn_from_demos(right_train_demos)

            # Evaluate on held-out test demos
            right_errors = []

            for i, test_demo in enumerate(right_test_demos):
                trajectory_pos, _ = prodmp_right.condition_trajectory(
                    start_pos=test_demo[0],
                    goal_pos=test_demo[-1],
                    start_vel=np.zeros(6),
                    speed_factor=1.0
                )

                error = np.abs(trajectory_pos - test_demo).mean()
                right_errors.append(error)

            right_avg_error = np.mean(right_errors)
            right_std_error = np.std(right_errors)

        except Exception as e:
            right_avg_error = None
            right_std_error = None

        all_results[gesture_name] = {
            'left': (left_avg_error, left_std_error),
            'right': (right_avg_error, right_std_error)
        }

    for gesture_name, results in all_results.items():
        left_avg, left_std = results['left']
        right_avg, right_std = results['right']

        left_str = f"{left_avg:.6f} ± {left_std:.6f}" if left_avg is not None else "N/A"
        right_str = f"{right_avg:.6f} ± {right_std:.6f}" if right_avg is not None else "N/A"

        print(f"{gesture_name:<15} {left_str:<25} {right_str:<25}")


if __name__ == "__main__":
    main()
