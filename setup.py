from setuptools import setup, find_packages

setup(
    name='outfit_segmentation_app',
    version='0.1.0',
    author='Nana Casmana Ade Wikarta',
    description='API for Outfit Segmentation using FastAPI',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires= [requirement.strip() for requirement in open("requirements.txt", encoding="utf-8").readlines()],
)