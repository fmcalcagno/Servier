from setuptools import setup
from setuptools import find_packages
import setuptools
from setuptools.command.develop import develop
from main import train
import setuptools.command.build_py



setup(name="servier",
      version="0.0.1",
      description="Exam test for Servier",
      author="Facundo Calcagno",
      author_email="fmcalcagno@gmail.com",
      packages=find_packages() ,
      install_requires=["requests"],

      license="Apache 2.0",
      entry_points={
            'console_scripts': [
                  'train = main:train',
                  'evaluate = main:evaluate',
                  'predict = main:predict',
            ],
      }
      )