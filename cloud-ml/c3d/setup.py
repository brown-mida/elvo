import setuptools

setuptools.setup(
    name='cloudml-c3d',
    version='0.0.2',
    install_requires=[
        'google-cloud-storage>=1.10.0',
        'keras==2.1.3',
        'matplotlib>=2.2.2',
        'numpy>=1.14.5',
        'pandas>=0.23.1',
        'Pillow>=5.1.0',
        'requests>=2.19.1',
        'scikit-image>=0.14.0',
        'scikit-learn>=0.19.1',
        'scipy>=1.1.0',
    ],
    packages=setuptools.find_packages(),
    include_package_data=True,
)
