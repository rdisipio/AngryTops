"""
AngryTops
"""

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open( 'AngryTops/__init__.py' ) as f:
    version = f.readlines()[-1].split()[-1].strip("\"'")

setup(
	name='AngryTops',
	version=version,
	packages=['AngryTops'],
	url='https://github.com/rdisipio/AngryTops',
	keywords=[
		'top quark',
		'kinematics',
		'reconstruction'
		],
	license='LICENSE',
	description='Probabilistic kinematics reconstruction of top quark decay chain',
	classifiers=[
        	'License :: OSI Approved :: MIT License',
        	'Operating System :: OS Independent',
        	'Development Status :: 3 - Alpha',
        	'Topic :: Scientific/Engineering :: Mathematics',
        	'Topic :: Scientific/Engineering :: Physics',
        	'Intended Audience :: Science/Research',
        	'Programming Language :: Python'
    	],
	install_requires=requirements,
	scripts=[
		'scripts/check_nan.py',
		'scripts/fit.py',
		'scripts/histograms.py',
		'scripts/launch_gpu_job.sh',
		'scripts/root2csv.py',
		'scripts/setup_env.sh',
		'scripts/plot_observables.py',
		]
)

