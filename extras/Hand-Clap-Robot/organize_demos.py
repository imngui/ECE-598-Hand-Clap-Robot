import os
import re
import shutil
from pathlib import Path
from collections import defaultdict

def organize_demo_files(base_path="data/new_data"):
    base_dir = Path(base_path)

    if not base_dir.exists():
        print(f"Error: {base_path} does not exist")
        return

    # Process each subfolder in processed_data
    for gesture_folder in base_dir.iterdir():
        if not gesture_folder.is_dir():
            continue

        print(f"\nProcessing folder: {gesture_folder.name}")

        # Find all files with demo numbers
        demo_files = defaultdict(list)
        pattern = re.compile(r'_demo_(\d+)\.csv$')

        for file_path in gesture_folder.glob("*.csv"):
            match = pattern.search(file_path.name)
            if match:
                demo_num = match.group(1)
                demo_files[demo_num].append(file_path)

        # Create demo folders and move files
        for demo_num, files in sorted(demo_files.items()):
            demo_folder = gesture_folder / f"demo_{demo_num}"
            demo_folder.mkdir(exist_ok=True)

            print(f"  Moving {len(files)} files to demo_{demo_num}/")
            for file_path in files:
                dest_path = demo_folder / file_path.name
                if dest_path.exists():
                    print(f"    Warning: {dest_path} already exists, skipping")
                else:
                    shutil.move(str(file_path), str(dest_path))
                    print(f"    Moved: {file_path.name}")

        print(f"  Created {len(demo_files)} demo folders in {gesture_folder.name}")

if __name__ == "__main__":
    organize_demo_files()
    print("\n✓ Organization complete!")
