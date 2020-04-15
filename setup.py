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
        'mock>=3.0.5',
        'numpy>=1.18.2',
        'h5py>=2.9.0',
        'palettable>=3.1.1',
        'scipy>=1.2.1',
        'descartes>=1.1.0',
        'Shapely>=1.7.0',
        'verdict>=1.1.3',
    ],
)
