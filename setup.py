import io

from setuptools import find_packages, setup

setup(
    name='pycbclstm',
    version='1.0.0',
    url='https://github.com/marcin119a/lstm-pycbc',
    maintainer='marcin119a@gmail.com',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'pycbc',
        'lalsuite',
        'ligo-common',
        'hickle',
        'keras==2.4.3',
        'tensorflow==2.3.0'
    ],
)