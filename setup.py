from setuptools import setup, find_packages

setup(
    name='teeteeass',  # Replace with your package name
    version='0.4',  # Package version
    packages=find_packages(),  # Automatically find all packages
    description='Simple TTS for Bahasa Melayu',  # Short description
    long_description=open('README.md').read(),  # Long description from README.md
    long_description_content_type='text/markdown',  # Content type of the long description
    url='http://github.com/khursani8/teeteeass',  # Replace with your project's URL
    author='Sani',  # Replace with your name
    author_email='khursani8@gmail.com',  # Replace with your email
    license='MIT',  # License type
    install_requires=[
        # Dependencies from requirements.txt can be listed here
        # Example: 'numpy', 'pandas'
        open('requirements.txt').read().split("\n")
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.10',  # Minimum Python version requirement
    package_data={
        'teeteeass.nlp.english': ['cmudict.rep'],
    },
)
