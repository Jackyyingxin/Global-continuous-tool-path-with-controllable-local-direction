import sys
import os
import argparse
from ctypes import c_void_p
import numpy as np
import math
import OpenGL.GL as gl
import OpenGL.GLUT as glut
import glutils as ut
import stlmesh
import slicer
import matplotlib as mpl
import matplotlib.pyplot as plt
import Vtkadaptor
import geotypes as gt
model_name = "Gear1.stl"
stl_folder_name = '../stl_model/'
mesh = stlmesh.stlmesh(stl_folder_name + model_name) #论文中的zigzag模型为ma2.0STL
mesh_min = mesh.min_coordinates()[2]
mesh_max = mesh.max_coordinates()[2]

save_folder_path = '../Contour_Data/' + model_name[:-4]
save_path = save_folder_path
P = None
srt = None
delta = 0.3

mesh_slicer = slicer.slicer(mesh.triangles, P, delta, srt)
mesh_slicer.incremental_slicing()


def show_plane_in_black(plane):
    va=Vtkadaptor.VtkAdaptor() #直接import要加模块.方法
    for polygon in plane:
        va.drawPolygon3f(polygon).GetProperty().SetColor(0, 0, 0)
    va.display()

def show_plane_in_red(plane):
    va=Vtkadaptor.VtkAdaptor() #直接import要加模块.方法
    for polygon in plane:
        va.drawPolygon3f(polygon).GetProperty().SetColor(1, 0, 0)
    va.display()

def show_all_plane(save = 0):
    va = Vtkadaptor.VtkAdaptor()
    for plane in mesh_slicer.planes:
        if len(plane) > 1 and (direnction(plane[0]) == direnction(plane[1]) == 0):
            for polygon in plane:
                if type(polygon.vertices)!=list:
                    polygon.vertices = polygon.vertices.tolist()
                    polygon.vertices.append(polygon.vertices[0])
                else:
                    polygon.vertices.append(polygon.vertices[0])
                temp = va.drawPolygon3f(polygon)

                temp.GetProperty().SetColor(0, 0, 0)
                temp.GetProperty().SetLineWidth(4)

        else:
            for polygon in plane:
                if type(polygon.vertices)!=list:
                    polygon.vertices = polygon.vertices.tolist()
                    polygon.vertices.append(polygon.vertices[0])
                else:
                    polygon.vertices.append(polygon.vertices[0])
                #center = polygon.centroid()
                temp = va.drawPolygon3f(polygon)
                temp.GetProperty().SetColor(0, 0, 0)
                temp.GetProperty().SetLineWidth(4)
    if save!=1:
        va.display()
    else:
        va.write_image(
            "Slicing.jpg"
                , va.window, 0)




def output(index):
    plane = mesh_slicer.planes[6]
    if len(plane) > 1 and (direnction(plane[0]) == direnction(plane[1]) == 1):
        for polygon in plane:
            polygon.vertices.append(polygon.vertices[0])
        show_plane_in_red(plane)
    else:
        for polygon in plane:
            polygon.vertices.append(polygon.vertices[0])
        show_plane_in_black(plane)


def real_output():

    for layer_num, plane in enumerate(mesh_slicer.planes):
        # 每一个Z层建立一个文件夹
        save_layer_path = save_path + "\\" + "{}".format(layer_num)
        os.makedirs(save_layer_path, exist_ok=True)
        num = 0
        for p1 in range(len(plane)):
            if plane[p1].orientation()!=gt.COUNTERCLOCKWISE:
                continue
            # 找到外轮廓
            # 一个外轮廓表示一个形状，新建一个表示polygon的数据集
            save_contour_path = save_layer_path + "\\" + "{}".format(num)
            os.makedirs(save_contour_path , exist_ok=True)
            # 写入外轮廓
            f = open(save_contour_path + "/" + "out-contour.txt", "w")
            for point in plane[p1].vertices:
                f.write(str(point.coord[0]) + "    " + str(point.coord[1]) + "    " + str(
                    point.coord[2]) + '\n')
            num = num + 1
            f.close()
            inner_num = 0
            for p2 in range(len(plane)):
                if p1 != p2:
                    if plane[p1].is_inside(plane[p2].vertices[0]):
                        # p2是内轮廓
                        f1 = open(save_contour_path + "/" + "inner-contour{}.txt".format(inner_num), "w")
                        for point_i in plane[p2].vertices:
                            f1.write(str(point_i.coord[0]) + "    " + str(point_i.coord[1]) + "    " + str(
                                point_i.coord[2]) + '\n')
                        f1.close()
                        inner_num = inner_num + 1



def output_all_layer():
    #输出所有层
    with open(save_path, 'w') as f:
        for plane in mesh_slicer.planes:
            if len(plane) > 1 and (direnction(plane[0]) == direnction(plane[1]) == 0):
                for polygon in plane:
                    f.write("逆时针同层外轮廓开始")
                    if type(polygon.vertices) != list:
                        polygon.vertices = polygon.vertices.tolist()
                        polygon.vertices.append(polygon.vertices[0])
                    else:
                        polygon.vertices.append(polygon.vertices[0])

                    for point in polygon.vertices:
                        f.write(str(point.coord[0]) + "    " + str(point.coord[1]) + "    " + str(
                            point.coord[2]) + '\n')
                    f.write("逆时针同层外轮廓结束")

            elif len(plane) > 1 :
                for polygon in plane:
                    if direnction(polygon)==1:
                        with open(save_path1, 'w') as f1:
                            if type(polygon.vertices) != list:
                                polygon.vertices = polygon.vertices.tolist()
                                polygon.vertices.append(polygon.vertices[0])
                            else:
                                polygon.vertices.append(polygon.vertices[0])
                            for point in polygon.vertices:
                                f.write(str(point.coord[0]) + "    " + str(point.coord[1]) + "    " + str(
                                    point.coord[2]) + '\n')


    f.close()


def direnction(contour):##judge the clockwise or not for polygon
    points = contour.vertices
    d=0
    for i in range(len(points)-1):
        d += -0.5*(points[i+1].coord[1]+points[i].coord[1])*(points[i+1].coord[0]-points[i].coord[0])#离散求曲面积分判断轮廓顺逆，>0逆时针，<0,顺时针
    if d>0:
        return 0
    else:
        return 1

show_all_plane()
real_output()








