from setuptools import setup


setup(
    name="tripsynth",
    version="0.1.0",
    description="Survey tuple transformation, counting, and synthesis utilities.",
    long_description=(
        "Utilities for transforming survey CSVs, generating count tables, and "
        "training/sampling a contrastive VAE synthesizer."
    ),
    py_modules=["tuple", "get_counts", "get_tupled_counts", "synthesize", "howdy"],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24",
        "tqdm>=4.66",
        "torch>=2.2",
    ],
    entry_points={
        "console_scripts": [
            "tuple-survey=tuple:main",
            "get-tupled-counts=get_tupled_counts:main",
            "synthesize-household-trips=synthesize:main",
        ]
    },
)
