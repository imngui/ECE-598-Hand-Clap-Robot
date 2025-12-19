import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D


class HandAnimationViewer:
    # Define hand skeleton connections (bone structure)
    HAND_CONNECTIONS = [
        # Thumb
        ("Wrist", "ThumbMetacarpal"),
        ("ThumbMetacarpal", "ThumbProximal"),
        ("ThumbProximal", "ThumbDistal"),
        ("ThumbDistal", "ThumbTip"),
        # Index
        ("Palm", "IndexMetacarpal"),
        ("IndexMetacarpal", "IndexProximal"),
        ("IndexProximal", "IndexIntermediate"),
        ("IndexIntermediate", "IndexDistal"),
        ("IndexDistal", "IndexTip"),
        # Middle
        ("Palm", "MiddleMetacarpal"),
        ("MiddleMetacarpal", "MiddleProximal"),
        ("MiddleProximal", "MiddleIntermediate"),
        ("MiddleIntermediate", "MiddleDistal"),
        ("MiddleDistal", "MiddleTip"),
        # Ring
        ("Palm", "RingMetacarpal"),
        ("RingMetacarpal", "RingProximal"),
        ("RingProximal", "RingIntermediate"),
        ("RingIntermediate", "RingDistal"),
        ("RingDistal", "RingTip"),
        # Little
        ("Palm", "LittleMetacarpal"),
        ("LittleMetacarpal", "LittleProximal"),
        ("LittleProximal", "LittleIntermediate"),
        ("LittleIntermediate", "LittleDistal"),
        ("LittleDistal", "LittleTip"),
        # Palm connections
        ("Wrist", "Palm"),
    ]

    def __init__(self, folder_path):
        """Initialize viewer with data from folder containing hand tracking files."""
        self.folder_path = folder_path
        self.export_counter = 1

        # Find and load hand data files
        self.left_hand_file = None
        self.right_hand_file = None
        self.all_files = []  # Store all files in directory for segmentation

        # Search for hand files
        all_csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

        for file_path in all_csv_files:
            filename = os.path.basename(file_path)
            if "left_hand" in filename.lower():
                self.left_hand_file = file_path
            elif "right_hand" in filename.lower():
                self.right_hand_file = file_path
            # Store all files for later segmentation
            self.all_files.append(file_path)

        if not self.left_hand_file and not self.right_hand_file:
            raise ValueError(
                f"No hand data files found in {folder_path}. Looking for files with 'left_hand' or 'right_hand' in the name."
            )

        # Load and process data
        self.positions = {}  # Store processed positions as numpy arrays
        self.joint_names = {}
        self.joint_to_idx = {}  # Mapping from joint name to index

        if self.left_hand_file:
            df = pd.read_csv(self.left_hand_file)
            self.joint_names["left"] = self._extract_joint_names(df)
            self.positions["left"] = self._process_hand_data(
                df, self.joint_names["left"]
            )
            self.joint_to_idx["left"] = {
                name: i for i, name in enumerate(self.joint_names["left"])
            }
            print(
                f"Loaded left hand: {os.path.basename(self.left_hand_file)} ({self.positions['left'].shape[0]} frames, {len(self.joint_names['left'])} joints)"
            )

        if self.right_hand_file:
            df = pd.read_csv(self.right_hand_file)
            self.joint_names["right"] = self._extract_joint_names(df)
            self.positions["right"] = self._process_hand_data(
                df, self.joint_names["right"]
            )
            self.joint_to_idx["right"] = {
                name: i for i, name in enumerate(self.joint_names["right"])
            }
            print(
                f"Loaded right hand: {os.path.basename(self.right_hand_file)} ({self.positions['right'].shape[0]} frames, {len(self.joint_names['right'])} joints)"
            )

        # Use the number of frames from the first loaded dataset
        self.num_frames = list(self.positions.values())[0].shape[0]

        # Verify all datasets have the same number of frames
        for hand, pos in self.positions.items():
            if pos.shape[0] != self.num_frames:
                print(
                    f"Warning: {hand} hand has {pos.shape[0]} frames, expected {self.num_frames}"
                )

        # Animation state
        self.current_frame = 0
        self.is_playing = True
        self.loop_animation = False  # Toggle with space key
        self.start_frame = 0
        self.end_frame = self.num_frames - 1

        # Setup figure and axes
        self.setup_figure()

        # Connect keyboard event
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        # Create animation
        self.anim = FuncAnimation(
            self.fig,
            self.update_animation,
            frames=self.num_frames,
            interval=50,  # 20 FPS
            repeat=True,
        )

    def _extract_joint_names(self, df):
        """Extract joint names from dataframe columns."""
        joint_names = []
        for col in df.columns:
            if col.endswith("_x"):
                joint_name = col[:-2]  # Remove '_x' suffix
                joint_names.append(joint_name)
        return joint_names

    def _process_hand_data(self, df, joint_names):
        """Process hand data following the exact approach from data_cleaner.ipynb.

        Returns numpy array of shape (num_frames, num_joints, 3) where:
        - positions[:, :, 0] = x coordinates
        - positions[:, :, 1] = z coordinates (swapped from y)
        - positions[:, :, 2] = y coordinates (swapped from z)
        """
        num_frames = len(df)
        num_joints = len(joint_names)

        # Store positions as (num_frames, num_joints, 3)
        positions = np.zeros((num_frames, num_joints, 3))

        for i, joint_name in enumerate(joint_names):
            # Load data with Y/Z swap (Unity to standard coordinates)
            positions[:, i, 0] = df[f"{joint_name}_x"].values
            positions[:, i, 1] = df[f"{joint_name}_z"].values  # Unity Z -> standard Y
            positions[:, i, 2] = df[f"{joint_name}_y"].values  # Unity Y -> standard Z

        return positions

    def setup_figure(self):
        """Setup matplotlib figure with 3D plot and GUI controls."""
        self.fig = plt.figure(figsize=(16, 10))

        # Create 3D subplots for each hand
        self.axes = {}

        # Calculate axis limits from all data (matching data_cleaner.ipynb)
        # Note: positions array has shape (num_frames, num_joints, 3)
        # where [:, :, 0] = x, [:, :, 1] = z (from Unity), [:, :, 2] = y (from Unity)
        all_positions = np.concatenate(
            [pos.reshape(-1, 3) for pos in self.positions.values()], axis=0
        )
        x_min, y_min, z_min = all_positions.min(axis=0) - 0.05
        x_max, y_max, z_max = all_positions.max(axis=0) + 0.05

        # Store axis limits for use in update_animation
        self.axis_limits = (x_min, x_max, y_min, y_max, z_min, z_max)

        # Setup left hand if available
        if "left" in self.positions:
            self.axes["left"] = self.fig.add_subplot(121, projection="3d")
            self.axes["left"].set_xlim(x_min, x_max)
            self.axes["left"].set_ylim(z_min, z_max)  # y-axis shows z values
            self.axes["left"].set_zlim(y_min, y_max)  # z-axis shows y values
            self.axes["left"].set_xlabel("X")
            self.axes["left"].set_ylabel("Z")
            self.axes["left"].set_zlabel("Y")
            self.axes["left"].set_title("Left Hand")

        # Setup right hand if available
        if "right" in self.positions:
            subplot_idx = 122 if "left" in self.positions else 111
            self.axes["right"] = self.fig.add_subplot(subplot_idx, projection="3d")
            self.axes["right"].set_xlim(x_min, x_max)
            self.axes["right"].set_ylim(z_min, z_max)  # y-axis shows z values
            self.axes["right"].set_zlim(y_min, y_max)  # z-axis shows y values
            self.axes["right"].set_xlabel("X")
            self.axes["right"].set_ylabel("Z")
            self.axes["right"].set_zlabel("Y")
            self.axes["right"].set_title("Right Hand")

        # Adjust subplot to make room for sliders and buttons
        self.fig.subplots_adjust(left=0.1, bottom=0.25, right=0.95)

        # Create sliders with increment/decrement buttons
        # Start frame slider and buttons
        ax_start_dec = plt.axes([0.10, 0.12, 0.03, 0.03])
        ax_start = plt.axes([0.15, 0.12, 0.7, 0.03])
        ax_start_inc = plt.axes([0.87, 0.12, 0.03, 0.03])

        self.slider_start = Slider(
            ax_start,
            "Start Frame",
            0,
            self.num_frames - 1,
            valinit=self.start_frame,
            valstep=1,
            color="lightblue",
        )
        self.btn_start_dec = Button(ax_start_dec, "-", color="lightblue")
        self.btn_start_inc = Button(ax_start_inc, "+", color="lightblue")

        # End frame slider and buttons
        ax_end_dec = plt.axes([0.10, 0.07, 0.03, 0.03])
        ax_end = plt.axes([0.15, 0.07, 0.7, 0.03])
        ax_end_inc = plt.axes([0.87, 0.07, 0.03, 0.03])

        self.slider_end = Slider(
            ax_end,
            "End Frame",
            0,
            self.num_frames - 1,
            valinit=self.end_frame,
            valstep=1,
            color="lightgreen",
        )
        self.btn_end_dec = Button(ax_end_dec, "-", color="lightgreen")
        self.btn_end_inc = Button(ax_end_inc, "+", color="lightgreen")

        # Create export button
        ax_export = plt.axes([0.4, 0.01, 0.2, 0.04])
        self.btn_export = Button(ax_export, "Export Selected Range", color="lightcoral")

        # Connect callbacks
        self.slider_start.on_changed(self.update_start_frame)
        self.slider_end.on_changed(self.update_end_frame)
        self.btn_start_dec.on_clicked(self.decrement_start_frame)
        self.btn_start_inc.on_clicked(self.increment_start_frame)
        self.btn_end_dec.on_clicked(self.decrement_end_frame)
        self.btn_end_inc.on_clicked(self.increment_end_frame)
        self.btn_export.on_clicked(self.export_data)

    def update_animation(self, frame):
        """Update animation for each frame."""
        if not self.is_playing:
            return

        # Handle looping vs play-once
        self.loop_animation = True
        if self.loop_animation:
            # Loop between start and end frames
            frame_range = self.end_frame - self.start_frame + 1
            self.current_frame = self.start_frame + (frame % frame_range)
        else:
            # Play once from start to end, then stop at end
            if self.current_frame >= self.end_frame:
                self.current_frame = self.end_frame
                self.is_playing = False
            else:
                self.current_frame = min(self.start_frame + frame, self.end_frame)

        # Update each hand
        for hand in self.positions.keys():
            ax = self.axes[hand]
            ax.clear()

            # Restore axis limits from stored values
            x_min, x_max, y_min, y_max, z_min, z_max = self.axis_limits

            ax.set_xlim(x_min, x_max)
            ax.set_ylim(z_min, z_max)  # y-axis shows z values
            ax.set_zlim(y_min, y_max)  # z-axis shows y values
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("Y")
            ax.set_title(
                f"{hand.capitalize()} Hand - Frame {self.current_frame}/{self.num_frames - 1}"
            )

            # Get positions for current frame
            frame_positions = self.positions[hand][self.current_frame]

            # Plot joints as scatter points
            # frame_positions has shape (num_joints, 3) where [:, 0]=x, [:, 1]=z, [:, 2]=y
            # We plot as (x, z, y) to match the notebook's visualization
            ax.scatter(
                frame_positions[:, 0],
                frame_positions[:, 2],
                frame_positions[:, 1],
                c="red",
                s=50,
                alpha=0.8,
            )

            # Draw connections between joints
            # Use the same axis order (x, z, y) as scatter points
            for joint1, joint2 in self.HAND_CONNECTIONS:
                if (
                    joint1 in self.joint_to_idx[hand]
                    and joint2 in self.joint_to_idx[hand]
                ):
                    idx1 = self.joint_to_idx[hand][joint1]
                    idx2 = self.joint_to_idx[hand][joint2]

                    pos1 = frame_positions[idx1]
                    pos2 = frame_positions[idx2]

                    # Use [0], [2], [1] to plot as (x, z, y)
                    ax.plot(
                        [pos1[0], pos2[0]],
                        [pos1[2], pos2[2]],
                        [pos1[1], pos2[1]],
                        "b-",
                        linewidth=2,
                        alpha=0.6,
                    )

        return []

    def on_key_press(self, event):
        """Handle keyboard events."""
        if event.key == " ":  # Space key
            self.loop_animation = not self.loop_animation
            if self.loop_animation:
                print("Loop mode: ON (animation will repeat)")
            else:
                print("Loop mode: OFF (animation will play once)")
                # Reset animation to start if we just disabled looping
                self.current_frame = self.start_frame
                self.is_playing = True

    def update_start_frame(self, val):
        """Callback for start frame slider."""
        self.start_frame = int(val)
        # Ensure start <= end
        if self.start_frame > self.end_frame:
            self.start_frame = self.end_frame
            self.slider_start.set_val(self.start_frame)

    def update_end_frame(self, val):
        """Callback for end frame slider."""
        self.end_frame = int(val)
        # Ensure end >= start
        if self.end_frame < self.start_frame:
            self.end_frame = self.start_frame
            self.slider_end.set_val(self.end_frame)

    def increment_start_frame(self, event):
        """Increment start frame by 1."""
        new_val = min(self.start_frame + 1, self.num_frames - 1)
        self.slider_start.set_val(new_val)

    def decrement_start_frame(self, event):
        """Decrement start frame by 1."""
        new_val = max(self.start_frame - 1, 0)
        self.slider_start.set_val(new_val)

    def increment_end_frame(self, event):
        """Increment end frame by 1."""
        new_val = min(self.end_frame + 1, self.num_frames - 1)
        self.slider_end.set_val(new_val)

    def decrement_end_frame(self, event):
        """Decrement end frame by 1."""
        new_val = max(self.end_frame - 1, 0)
        self.slider_end.set_val(new_val)

    def export_data(self, event):
        """Export selected frame range to CSV files for ALL files in the directory."""
        # Find next available export number
        while True:
            export_suffix = f"_demo_{self.export_counter}"
            # Check if any file with this export number exists
            test_file = os.path.join(self.folder_path, f"test{export_suffix}.csv")
            if not any(
                os.path.exists(f.replace(".csv", f"{export_suffix}.csv"))
                for f in self.all_files
            ):
                break
            self.export_counter += 1

        print(f"\nExporting frames {self.start_frame}-{self.end_frame}:")
        print(f"  Total frames: {self.end_frame - self.start_frame + 1}")

        exported_files = []

        # Export all CSV files in the directory with the same frame range
        for file_path in self.all_files:
            try:
                # Load the file
                df = pd.read_csv(file_path)

                # Check if file has enough frames
                if len(df) <= self.end_frame:
                    print(
                        f"  Skipping {os.path.basename(file_path)}: only has {len(df)} frames"
                    )
                    continue

                # Extract data for selected range
                selected_data = df.iloc[self.start_frame : self.end_frame + 1].copy()

                # Generate export filename
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                export_filename = f"{base_name}{export_suffix}.csv"
                export_path = os.path.join(self.folder_path, export_filename)

                # Save to CSV
                selected_data.to_csv(export_path, index=False)
                exported_files.append(export_filename)
                print(f"  ✓ {export_filename}")

            except Exception as e:
                print(f"Failed to export {os.path.basename(file_path)}: {e}")

        print(f"\nSuccessfully exported {len(exported_files)} file(s)")
        print(f"Location: {self.folder_path}")

        # Increment counter for next export
        self.export_counter += 1

    def show(self):
        """Display the viewer."""
        plt.show()


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python hand_animation_viewer.py <path_to_folder>")
        print("\nExample:")
        print("  python hand_animation_viewer.py data/wave_3/")
        print(
            "\nThe folder should contain CSV files with 'left_hand' and/or 'right_hand' in their names."
        )
        print(
            "When exporting, all CSV files in the folder will be segmented to the selected range."
        )
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    if not os.path.isdir(folder_path):
        print(f"Error: Not a directory: {folder_path}")
        sys.exit(1)


    try:
        viewer = HandAnimationViewer(folder_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    viewer.show()


if __name__ == "__main__":
    main()
