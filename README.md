### Global-continuous-tool-path-with-controllable-local-direction
##  Our code consists of four main modules. At first, you need to input the mesh model into the slicer to generate 2D connected region. A 2D connected region is saved as seperate inner and outer contours. Our slicer refernces the paper "An Optimal Algorithm for 3D Triangle Mesh Slicing" . 
# 1.Two-dimensional continuous toolpath generation:  
In this moudle you can generate the continuous toolpath in any fill angle for the inputed 2D connected region. The input to this module is a 2D connected region  $\mathbf{R}$ consisting of inner and outer contours, fill angle $\alpha$ and fill interval w.  
This moudle have four main algorithms.  
1. Pre-pcossing: this algorithm is used to smooth the $\mathbf{R}$, convert the any fill direction into horizontal fill and offset the $\mathbf{R}$ to generate conformal toolpath $b$ and fill regions $mathbf{o}$ .  
2. Geomtry Decomposition: this algorithm is used to decompose the inputed fill region into sub-regions and keep the number of sub-regions as small as possible.  
3. fill the subregions and connect the sub-paths: this algorithm is used to generate the sub-paths of the sub-regions and connect the sub-paths into a continuous path $\zeta$.  
4. toolpath optimization: this algorithm is used to rotate the $\zeta_i$ by $\alpha$ and connect every $\zeta_i$ to $b$ to generate $\gamma$. Using gradient descent to optimize $\zeta$ for uniform fill spacing.  
# 2. Cotinuous IICP:
 
