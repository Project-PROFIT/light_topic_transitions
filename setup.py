from distutils.core import setup

setup(
    name='tt_light',
    version='0.1dev',
    packages=['tt_light'],
    license='MIT',
    requires=open('requirements.txt', 'r').read().split()
)