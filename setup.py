import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='SimpleGP',
    version='1.0.1',
    author='Marco Virgolin',
    author_email='marco.virgolin@gmail.com',
    url='https://github.com/marcovirgolin/SimpleGP',
    packages=setuptools.find_packages(),
    long_description=long_description,
    long_description_content_type='text/markdown',
    setup_requires=['wheel'],
    install_requires=[
    	'numpy >= 1.16.1','scikit-learn >= 0.20.0'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
)
