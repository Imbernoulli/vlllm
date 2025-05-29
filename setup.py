import os
from setuptools import setup, find_packages

# Read README.md for long description
readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name='vlllm', # Changed name slightly to be more unique
    version='0.2.1', # Incremented version
    author='Bohan Lyu', # Or your name
    author_email='lyubh22@gmail.com', # Or your email
    description='A utility package for text generation using vLLM with multiprocessing support.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/Imbernoulli/vlllm', # Replace with your actual URL
    packages=find_packages(where="."), # Correctly find the package
    install_requires=[
        'vllm',    # Specify a version range for vllm
        'transformers',
        'torch',
        # 'ray' # vLLM installs Ray if needed for pipeline parallelism.
                  # Users might need to install it separately if they use pp > 1.
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
)
