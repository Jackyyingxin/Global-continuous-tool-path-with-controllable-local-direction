### Global-continuous-tool-path-with-controllable-local-direction
##  Our code consists of four main modules. At first, you need to input the mesh model into the slicer to generate 2D connected region. A 2D connected region is saved as seperate inner and outer contours. Our slicer refernces the paper "An Optimal Algorithm for 3D Triangle Mesh Slicing" . 
# 1.Two-dimensional continuous toolpath generation:  
In this moudle you can generate the continuous toolpath in any fill angle for the inputed 2D connected region. The input to this module is a 2D connected region $\mathbf{R}$consisting of inner and outer contours, fill angle and fill interval. 
 .   
 
