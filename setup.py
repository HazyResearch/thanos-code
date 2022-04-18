"""For pip."""

from setuptools import setup

exec(open("unagi/_version.py").read())
setup(
    name="unagi",
    version=__version__,
    description="Official repo for the paper 'Perfectly Balanced: Improving Transfer and Robustness of Supervised Contrastive Learning'",
    long_description=open("README.md").read(),
    packages=['unagi'],
    scripts=["bin/unagi"],
    install_requires=[
        "cmake>=3.21.2, <4.0.0",
        "datasets>=1.11.0, <2.0.0",
        "einops>=0.3.2, <1.0.0",
        "meerkat-ml",
        "opt-einsum>=3.3.0, <4.0.0",
        "pykeops>=1.5, <2.0",
        "pytorch-lightning>=1.4.5, <1.4.9",
        "torch",
        "torchvision>=0.10.0, <2.0.0",
        "transformers",
    ],
    include_package_data=True,
    url="https://github.com/HazyResearch/thanos-code",
    classifiers=[  # https://pypi.python.org/pypi?:action=list_classifiers
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires=">=3.8",
    author="HazyResearch Team",
)
