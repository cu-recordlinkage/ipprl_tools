from setuptools import setup

setup(name="ipprl_tools",
      version="1.0",
	  description="A collection of tools to assist with performing IPPRL Linkage",
	  author="Andrew Hill",
	  author_email="andrew.2.hill@ucdenver.edu",
	  install_requires=[
		"numpy",
		"scipy",
		"pandas",
		"fuzzy"],
	  packages=["ipprl_tools"])