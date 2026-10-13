from glob import glob

from setuptools import find_packages, setup

PACKAGE = "amr_learned_planner"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml", "README.md"]),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="OptiCore AMR team",
    maintainer_email="opticore@example.com",
    description="AMR-Lite learned local planner integration for opticore-amr",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            f"learned_planner = {PACKAGE}.learned_planner_node:main",
            f"cmd_arbiter = {PACKAGE}.cmd_arbiter_node:main",
            f"nav_episode_recorder = {PACKAGE}.nav_episode_recorder_node:main",
            f"goal_sequence_runner = {PACKAGE}.goal_sequence_runner_node:main",
        ],
    },
)
