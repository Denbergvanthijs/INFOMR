# INFOMR

Multimedia Retrieval

## Requirements

Please install [Python 3.8.10](https://www.python.org/downloads/release/python-3810/).

- Update pip: `python -m pip install --upgrade pip`
- Install packages for easier/faster/better installation of other packages: `pip install Cython wheel`
- Install required packages: `pip install -r requirements.txt`
- Check if installation was succesfull: `python -c "import open3d as o3d; print(o3d.__version__)"`

## Testing data

Multiple files for testing the 3D mesh environment are available:

- `./data/D00921.obj`
  - `.obj` file provided by the TAs, coming from the ShapeDatabase and belonging to the class `Apartment`
- `./data/Airplane_61.off`
  - `.off` file originally belonging to the Labeled PSB dataset class `Airplane`
- `./data/bearded_guy.ply`
  - `.ply` file from [artec3d.com](https://www.artec3d.com/3d-models/bearded-guy-hd) to test `.ply` files while the website for the Princeton Shape Benchmark dataset is down

## Resources

- [The assignment](https://webspace.science.uu.nl/~telea001/MR/Assignment)
- [Open3D - Getting started](open3d.org/docs/release/getting_started.html)
- [Princeton Shape Benchmark dataset](https://shape.cs.princeton.edu/benchmark/)
- [Labeled PSB dataset](https://people.cs.umass.edu/~kalo/papers/LabelMeshes/)
  - [Download link](https://people.cs.umass.edu/~kalo/papers/LabelMeshes/labeledDb.7z)
- [ShapeDatabase provided by TA's for testing](https://github.com/MaxRee94/ShapeDatabase_INFOMR)
