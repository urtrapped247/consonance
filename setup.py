from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='consonance',
      version="0.0.2",
      description="Image to Audio",
      author="urtrapped247, TravisPM, emzys",
      author_email="ninjarroyo07@gmail.com",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
