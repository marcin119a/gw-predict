import io

from setuptools import find_packages, setup

setup(
    name='lstm-pycbc',
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
        'hickle'
    ],
)