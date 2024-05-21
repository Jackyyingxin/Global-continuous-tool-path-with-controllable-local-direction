# Global-continuous-tool-path-with-controllable-local-direction
##  Our code consists of three main modules. At first, you need to input the mesh model into the slicer to generate 2D connected region. A 2D connected region is saved as seperate inner and outer contours. Our slicer refernces the paper "An Optimal Algorithm for 3D Triangle Mesh Slicing" . 
### 1.Two-dimensional continuous toolpath generation:  
In this moudle you can generate the continuous toolpath in any fill angle for the inputed 2D connected region. The input to this module are a 2D connected region  $\mathbf{R}$ consisting of inner and outer contours, fill angle $\alpha$ and fill interval w.  
This moudle have four main algorithms.  
1. Pre-pcossing: this algorithm is used to smooth the $\mathbf{R}$, convert the any fill direction into horizontal fill and offset the $\mathbf{R}$ to generate conformal toolpath $b$ and fill regions $mathbf{o}$ .  
2. Geomtry Decomposition: this algorithm is used to decompose the inputed fill region into sub-regions and keep the number of sub-regions as small as possible.  
3. fill the subregions and connect the sub-paths: this algorithm is used to generate the sub-paths of the sub-regions and connect the sub-paths into a continuous path $\zeta$.  
4. toolpath optimization: this algorithm is used to rotate the $\zeta_i$ by $\alpha$ and connect every $\zeta_i$ to $b$ to generate $\gamma$. Using gradient descent to optimize $\zeta$ for uniform fill spacing.\
   The path optimization algorithm still works for the rest of the filled paths, but it fails to optimize the part of the path with self-intersections
   ![before](https://github.com/Jackyyingxin/Global-continuous-tool-path-with-controllable-local-direction/assets/96177758/fb9fe40b-2caa-42a8-97c0-7c6ed453e06f)
   ![after](https://github.com/Jackyyingxin/Global-continuous-tool-path-with-controllable-local-direction/assets/96177758/7fd94eaa-1738-447f-baf2-2a8522f19892)
### 2. Cotinuous IICP:  
In this moudle you can generate the continous toolpath with controllable local direction. The input are the partitions of a 2D connected region with their fill angles. The input of this moudle cannot generate from the slicer in this project. Users need to prepare their own input. This moudle fills the parttions with the method of moudle 1. Build an undirected graph from the adjacency of partitions and use the minimum spanning tree of the undirected graph to connect the continuous paths of partitions to 
the continous toolpath with controllable local direction.  
### 3. Idle stroke length optimization:  
This moudle is used to reduce the idle stroke and has three main algorithms.  
1. Printing sequence optimizatin: this algorithm is used to optimal the printing order of the sequnce of the 2D connected regions sliced to avoid the collision and reduce idle stroke. The input are horizontal threshold $T_x$, height threshold $T_y$ and the sequnce of the 2D connected regions. The output is the printing sequence of 2D connected region $\mathbf{r}$.  
2. Using the GA to select connecting points: this algorithm is used to selcet connecting points between adjecent 2D paths in the order of $\mathbf{r}$. The input is $\mathbf{r}$ and the set of 2D continuous paths, population quantity $M$ , the number of iterations $N$ , crossover rate $Q_c$ , mutation $Q_m$ , $eps$, the number of sampling points $m$ . The output is the global continuous path. 
3. Using the SAT to avoid interference: this algorithm is used to adjust the idle stroke of the global continuous path to avoid interference. The input is $\mathbf{r}$, the radius of nozzele and the global continuous path. The output is the optimized global continuous path. 
## the code is comming soon......
