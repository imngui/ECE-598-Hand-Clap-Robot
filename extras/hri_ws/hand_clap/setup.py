import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hand_clap'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ingui',
    maintainer_email='ingui2@illinois.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gesture_classifier = hand_clap.gesture_classifier:main',
            'new_gesture_classifier = hand_clap.new_gesture_classifier:main',
            'bimanual_sim_node = hand_clap.bimanual_sim_node:main',
            # 'hand_motion_predictor = hand_clap.hand_motion_predictor:main',
            # 'hand_velocity_estimator = hand_clap.hand_velocity_estimator:main',
            # 'prodmp_reaction_planner = hand_clap.prodmp_reaction_planner:main',
            'demo_collector = hand_clap.demo_collector:main',
        ],
    },
)
