from setuptools import find_packages, setup

setup(
    name='EasyDatas',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'torch'
    ],
    # package_data={
    #     "EasyDatas":[]
    # },
    python_requires=">=3.6" 
)