import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="stained_glass",
    version="0.1",
    author="Zach Hafen",
    author_email="zachary.h.hafen@gmail.com",
    description="For mocking up and interpreting multiple observations through a halo.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zhafen/stained_glass",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        'scipy>=1.7.1',
        'Shapely>=1.7.1',
        'h5py>=3.4.0',
        'numpy>=1.21.2',
        'numba>=0.53.1',
        'palettable>=3.3.0',
        'matplotlib>=3.4.3',
        'tqdm>=4.62.2',
        'descartes>=1.1.0',
        'mock>=4.0.3',
        'augment>=0.4',
        'verdict>=1.1.4',
    ],
)
