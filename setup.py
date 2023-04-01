from setuptools import find_packages, setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(
    name='ashafm package',
    version="0.0.1",
    description="asha fm package",
    author="Sophia Bouchama & Jack Sibley",
    author_email="info@asha.fm",
    #url="asha.fm",
    install_requires=requirements,
    packages=find_packages(),
    test_suite="tests",
    # include_package_data: to install data from MANIFEST.in
    include_package_data=True,
    zip_safe=False)
