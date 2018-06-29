from distutils.core import setup

setup(
    name='tt_light',
    version='profit-v15',
    packages=['light_topic_transitions'],
    license='MIT',
    requires=open('requirements.txt', 'r').read().split()
)