from setuptools import setup

setup(
    name='blueno',
    version='0.1.0',
    description='A tracking and preprocessing toolkit for practical ML',
    long_description_content_type='text/markdown',
    author='luke-zhu',
    author_email='luke_zhu@brown.edu',
    python_requires='>=3.6',
    url='https://github.com/elvoai/elvo-analysis',
    packages=['blueno'],
    install_requires=[
        'dataclasses==0.6',
        'elasticsearch-dsl==6.2.1',
        'elasticsearch==6.3.0',
        'google-cloud-storage==1.10.0',
        'h5py==2.8.0',
        'matplotlib==2.2.2',
        'numpy==1.14.5',
        'pandas==0.23.1',
        'Pillow==5.1.0',
        'requests==2.19.1',
        'scikit-image==0.14.0',
        'scikit-learn==0.19.1',
        'scipy==1.1.0',
    ],
    extras_require={
        'cpu': [
            'tensorflow==1.4',
            'Keras==2.1.3',
        ],
        'gpu': [
            'tensorflow-gpu==1.4.1',
            'keras==2.1.3',
        ],
        'test': [
            'pytest==3.6.1',
            'pytest-cov==2.5.1',
            'codecov==2.0.15',
            'flake8==3.5.0',
        ],
        'docs': [
            'recommonmark==0.4.0',
            'sphinx==1.7.6',
        ],
    },
    include_package_data=True,
    license='MIT',
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
