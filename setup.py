from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import subprocess
import os
import sys

class CMakeBuild(build_ext):
    def run(self):
        # Build C library first
        subprocess.check_call(['make', 'clean'])
        subprocess.check_call(['make', '-j4'])
        super().run()

setup(
    name="venus-inference",
    version="0.1.0",
    author="Venus Contributors",
    description="Universal cross-platform inference engine with OpenAI-compatible API",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/venus",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    package_data={
        "": ["*.so", "*.dylib", "*.dll"],
    },
    install_requires=[
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "pydantic>=2.5.0",
        "torch>=2.1.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.19.0",
        "numpy>=1.24.0",
        "tqdm>=4.66.0",
        "click>=8.1.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "httpx>=0.25.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
            "mypy>=1.7.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "venus-server=python.api_server:main",
            "venus-convert=scripts.convert_hf_model:main",
            "venus-download=scripts.download_model:main",
        ],
    },
    python_requires=">=3.8",
    cmdclass={
        "build_ext": CMakeBuild,
    },
    ext_modules=[Extension("venus._c_lib", [])],  # Dummy extension to trigger build
)