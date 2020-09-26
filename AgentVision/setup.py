import setuptools

with open('README.md', 'r') as fh:
    long_description = fh.read()

with open('requirements.txt', 'r') as fh:
    required = fh.read().splitlines()

setuptools.setup(
    name="agentvision",
    version="0.0.1",
    author="Arun Kumar",
    author_email="arunkumar.rn.eee@gmail.com",
    description="vision package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/arunsechergy/EVA/AgentVision",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=required,
)
