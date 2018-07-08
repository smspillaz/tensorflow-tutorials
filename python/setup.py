# /setup.py
#
# Installation and setup script for tensorflow-tutorials
#
# See /LICENCE.md for Copyright information
"""Installation and setup script for tensorflow-tutorials."""

from setuptools import find_packages, setup

setup(name="tensorflow-tutorials",
      version="0.0.1",
      description="""Examples for TensorFlow.""",
      long_description="""TensorFlow exampleprograms.""",
      author="Sam Spilsbury",
      author_email="smspillaz@gmail.com",
      classifiers=["Development Status :: 3 - Alpha",
                   "Programming Language :: Python :: 2",
                   "Programming Language :: Python :: 2.7",
                   "Programming Language :: Python :: 3",
                   "Programming Language :: Python :: 3.1",
                   "Programming Language :: Python :: 3.2",
                   "Programming Language :: Python :: 3.3",
                   "Programming Language :: Python :: 3.4",
                   "Intended Audience :: Developers",
                   "Topic :: Software Development :: Build Tools",
                   "License :: OSI Approved :: MIT License"],
      url="http://github.com/smspillaz/tensorflow-tutorials",
      license="ISC",
      keywords="nlp",
      packages=find_packages(
          exclude=["cc"]
      ),
      install_requires=[
          "setuptools"
      ],
      entry_points={
          "console_scripts": [
              "tensorflow-tutorial-mnist=tensorflow_tutorials.mnist:main"
          ]
      },
      zip_safe=True,
      include_package_data=True)
