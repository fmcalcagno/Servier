from setuptools import setup, find_packages


setup(
    name='servier',
    author='facundo',
    author_email='fmcalcagno@gmail.com',
    packages= find_packages(include=['src','src.*']),
    install_requires=[
            'torch',
            'pandas',
            'numpy>=1.14.5',
            'xnetwork',
            'flask',
            'scikit-learn'
        ],
    entry_points={
        'console_scripts': [
            'train = src.main:train',
            'evaluate = src.main:evaluate',
            'predict = src.main:predict']
    },
)
