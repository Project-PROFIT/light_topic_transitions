from distutils.core import setup

setup(
    name='tt_light',
    version='0.1dev',
    packages=['light_topic_transition'],
    license='MIT',
    requires=open('requirements.txt', 'r').read().split()
)