from setuptools import find_packages
from setuptools import setup

setup(
    name='energy-transition-models',
    packages=find_packages(),
    # python_requires='=3.9.7',
    install_requires=[
        "torch<=2.0.1",
        "gym==0.23.1",
        "mujoco_py",
        "matplotlib",
        "numpy",
        "scipy",
        "pandas",
        "tensorboard",
        "tqdm",
        "argparse",
        "h5py",
        "pickle",
        "tensorflow",
    ],
    package_data={
        # include default config files
        "": ["*.yaml", "*.xml"],
    }
)