# Global-continuous-tool-path-with-controllable-local-direction
##  Our code consists of three main modules. At first, you need to input the mesh model into the slicer to generate 2D connected region. A 2D connected region is saved as seperate inner and outer contours. Our slicer refernces the paper "An Optimal Algorithm for 3D Triangle Mesh Slicing" . 

### 1.Two-dimensional continuous toolpath generation:  
In this moudle you can generate the continuous toolpath in any fill angle for the inputed 2D connected region. 

Please run [`main.py `](#main.py ) of [`sclicer_tool`](#sclicer_tool ) to slice stl models in [`stl_model`](#stl_model ) at first.

Domains belonging to the same layer are stored under a common layer folder, while the inner and outer contours of each domain are stored together within a dedicated domain folder.

After that, you can run the [`main.py `](#main.py ) in the [`Source`](#Source ) folder to obtain the 2D continuous filling path.
