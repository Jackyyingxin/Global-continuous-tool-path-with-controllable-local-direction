import Generate_2D_continuous
import Utility
model_name = "Gear1/"
load_path = "../Contour_Data/" + model_name
# inner_contour_set, out_contour_set
first_layer = 0.2004
H = 0.4
# layer_thickness
Height = 2.0
# angles is the filling angles from start layer to end layer
angles = [Utility.degToRad(45), Utility.degToRad(0)]
coefficient = 10000.0
interval = 0.4
flag = 0
# flag == 1 global continuous path, flag == 0 2D continuous path
start = 1
end = 2
iter_num = 60
learning_rate = 0.003
# The weight of Int item
W1 = 2
# The weight of Smooth item
W2 = 1
# Resample_accuracy
R = 0.2

pathes = Generate_2D_continuous.generate_2D_continuous_path(
    load_path,  start, coefficient, end, interval, angles, iter_num, learning_rate, W1, W2, R, flag)
