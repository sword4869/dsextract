from importlib.metadata import entry_points
from setuptools import setup
import os
import pkg_resources

setup(
    name='dsextract',
    version='0.1',
    description='A Python Package for extracting deepspeech features from audio files',
    author='sword4869',
    install_requires=[
        str(r) for r in pkg_resources.parse_requirements(open(os.path.join(os.path.dirname(__file__), "requirements.txt")))
    ],
    entry_points={
        'console_scripts': [
            'dsextract = dsextract.audio_handler:main',
            'extract_wav = dsextract.extract_wav:main',
        ]
    }
)