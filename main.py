import Generate_2D_continuous
import Utility

inner_path = "D:/Python/code-python/inner_contour.txt"
out_path = "D:/Python/code-python/out_contour.txt"
# inner_contour_set, out_contour_set
first_layer = 0.2004
H = 0.4
# layer_thickness
Height = 2.0
# angles is the filling angles from start layer to end layer
angles = [Utility.degToRad(45), 20]
coefficient = 10000.0
interval = 0.4
flag = 0
# flag == 1 global continuous path, flag == 0 2D continuous path
start = 1
end = 2
# the parameter in the paper "An Optimal Algorithm for 3D Triangle Mesh Slicing"
eps = 0.0004

pathes = Generate_2D_continuous.generate_2D_continuous_path(inner_path, out_path, start, coefficient, end, interval, angles, flag)
