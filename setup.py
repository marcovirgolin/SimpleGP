from distutils.core import setup

setup(
    name='SimpleGP',
    version='0.2',
    packages=['simplegp',
    	'simplegp.Evolution',
    	'simplegp.Fitness',
    	'simplegp.Nodes',
    	'simplegp.Selection',
    	'simplegp.Variation',
    	],
    license='The MIT License',
    long_description=open('README.md').read(),
    install_requires=[
    	"numpy >= 1.16.1",
    ],
)
