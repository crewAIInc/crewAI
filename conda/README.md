# Conda Package for CrewAI

This directory contains the necessary files to build a conda package for CrewAI.

## Files

- `meta.yaml`: The main conda recipe file that defines package metadata, dependencies, and build requirements
- `build.sh`: Build script for Unix-like systems (Linux, macOS)
- `bld.bat`: Build script for Windows

## Building the Package

To build the package, you need to have conda-build installed:

```bash
conda install conda-build
```

Then, from the repository root directory:

```bash
conda build conda
```

## Testing the Package

After building, you can install and test the package:

```bash
conda install --use-local crewai
```

## Uploading to Anaconda

To upload the package to Anaconda, you need to have anaconda-client installed:

```bash
conda install anaconda-client
anaconda login
anaconda upload /path/to/conda-bld/noarch/crewai-*.tar.bz2
```

## Compatibility Notes

This package addresses compatibility issues with:
- Python 3.10: Adds typing_extensions as a dependency to provide the Self type
- Python 3.12: Ensures compatibility with the tokenizers package
