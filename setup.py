from setuptools import setup

setup(
    name="ipprl_tools",
    version="1.0",
    description="A collection of tools to assist with performing IPPRL Linkage",
    author="Andrew Hill",
    author_email="andrew.2.hill@cuanschutz.edu",
    install_requires=[
        "numpy",
        "scipy",
        "pandas"],
    packages=["ipprl_tools", "ipprl_tools.utils"],
    package_data={"ipprl_tools": ["docs/*"]},
    include_package_data=True
)
