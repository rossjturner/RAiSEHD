from setuptools import setup, Extension

from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name='RAiSEHD',
    version='0.0.6',
    author = "Ross Turner",
    author_email = "turner.rj@icloud.com",
    description = ("RAiSE HD: Lagrangian particle-based radio AGN model."),
    long_description = long_description,
    long_description_content_type = "text/markdown",
    packages=['RAiSEHD'],
    package_dir={'RAiSEHD': 'RAiSEHD'},
    package_data={'RAiSEHD': ['RAiSE_particles.hdf5']},
    install_requires=[
        'numpy', 'pandas', 'matplotlib', 'h5py', 'astropy', 'scipy', 'numba'
    ],
)
