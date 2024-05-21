import vtkmodules.all as vtk
from pathlib import Path


from vtkmodules.vtkIOImage import (
        vtkBMPWriter,
        vtkJPEGWriter,
        vtkPNGWriter,
        vtkPNMWriter,
        vtkPostScriptWriter,
        vtkTIFFWriter
    )
from vtkmodules.vtkRenderingCore import vtkWindowToImageFilter
from vtkmodules.util.numpy_support import numpy_to_vtk

class VtkAdaptor:
    #def __init__(self, bgClr = (0.95, 0.95, 0.95)):
    def __init__(self, bgClr=(1, 1, 1)):
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(bgClr)
        self.window = vtk.vtkRenderWindow()
        self.window.AddRenderer(self.renderer)
        self.window.SetSize(1000, 1000)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.window)
        self.interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        #self.interactor.Initialize()

    def ShowPolygon(self, data):
        points = vtk.vtkPoints()
        points.SetData(numpy_to_vtk(data, deep=True))
        # Create a polyline to define the edges of the polygon
        polyline = vtk.vtkPolyLine()
        # Set the number of points in the polyline
        polyline.GetPointIds().SetNumberOfIds(data.shape[0])
        # Add each point in the ndarray to the polyline
        for i in range(data.shape[0]):
            polyline.GetPointIds().SetId(i, i)
        # Create a polydata object to hold the points and polyline
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        # Add the polyline to the cell array of the polydata
        polydata.SetLines(vtk.vtkCellArray())
        polydata.GetLines().InsertNextCell(polyline)
        return self.drawPlsrc(polydata)

    def write_image(self, file_name, ren_win, rgba=True):
        """
        Write the render window view to an image file.

        Image types supported are:
         BMP, JPEG, PNM, PNG, PostScript, TIFF.
        The default parameters are used for all writers, change as needed.

        :param file_name: The file name, if no extension then PNG is assumed.
        :param ren_win: The render window.
        :param rgba: Used to set the buffer type.
        :return:
        """


        #camera.Zoom(0.17039)
        #camera = self.renderer.GetActiveCamera()
        #camera.ParallelProjectionOn()
        #camera.SetParallelScale(23.5)
        ren_win.Render()

        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(ren_win)
        windowto_image_filter.SetScale(4)  # image quality
        #if rgba:
            #windowto_image_filter.SetInputBufferTypeToRGBA()
        #else:
            #windowto_image_filter.SetInputBufferTypeToRGB()
            # Read from the front buffer.
            #windowto_image_filter.ReadFrontBufferOff()
        windowto_image_filter.Update()

        if file_name:
            valid_suffixes = ['.bmp', '.jpg', '.png', '.pnm', '.ps', '.tiff']
            # Select the writer to use.
            parent = Path(file_name).resolve()
            path = Path(parent)
            if path.suffix:
                ext = path.suffix.lower()
            else:
                ext = '.png'
                path = Path(str(path)).with_suffix(ext)
            if path.suffix not in valid_suffixes:
                print(f'No writer for this file suffix: {ext}')
                return
            if ext == '.bmp':
                writer = vtkBMPWriter()
            elif ext == '.jpg':
                writer = vtkJPEGWriter()
            elif ext == '.pnm':
                writer = vtkPNMWriter()
            elif ext == '.ps':
                if rgba:
                    rgba = False
                writer = vtkPostScriptWriter()
            elif ext == '.tiff':
                writer = vtkTIFFWriter()
            else:
                writer = vtkPNGWriter()

            writer.SetFileName(path)
            writer.SetInputConnection(windowto_image_filter.GetOutputPort())
            writer.Write()
            self.interactor.GetRenderWindow().Finalize()
        else:
            raise RuntimeError('Need a filename.')

    def display(self):
        self.interactor.Start()
    def setBackgroundColor(self,r,g,b):
        return self.renderer.SegBackground(r,g,b)

    def drawActor(self, actor):
        self.renderer.AddActor(actor)
        return actor

    def drawPlsrc(self, pdsrc):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(pdsrc)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return self.drawActor(actor)

    def drawArSrc(self, arSrc):
        arrow_mapper = vtk.vtkPolyDataMapper()
        arrow_mapper.SetInputConnection(arSrc.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(arrow_mapper)
        return self.drawActor(actor)

    def drawPdSrc(self, pdSrc):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(pdSrc.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return self.drawActor(actor)

    def drawPoint(self, point, radius =0.2):
        src = vtk.vtkSphereSource()
        src.SetCenter(point.x, point.y, point.z)
        src.SetRadius(radius)
        return self.drawPdSrc(src)

    def draw_vector(self, positions, directions):
        """
        :param angle: 单位向量（2， ）
        :return:
        """
        points = vtk.vtkPoints()

        for i in range(len(positions)):
            points.InsertNextPoint(positions[i, 0], positions[i, 1], 0)

        # 创建PolyData对象
        polyData = vtk.vtkPolyData()
        polyData.SetPoints(points)

        # 创建vtkFloatArray，并设置向量的组件数为2
        vectors = vtk.vtkFloatArray()
        vectors.SetNumberOfComponents(3)

        # 向vtkFloatArray添加向量数据
        for i in range(len(positions)):
            vectors.InsertNextTuple3(directions[i, 0], directions[i, 1], 0)

        # 将vtkFloatArray设置为PolyData的点数据
        polyData.GetPointData().SetVectors(vectors)  # 确保这一行没有问题

        # 创建箭头源
        arrowSource = vtk.vtkArrowSource()

        # 创建Glyph映射器
        glyph = vtk.vtkGlyph3D()
        glyph.SetSourceConnection(arrowSource.GetOutputPort())
        glyph.SetInputData(polyData)
        glyph.SetScaleFactor(1)  # 调整箭头大小

        return self.drawPdSrc(glyph)



    def showPoint(self,point, radius =0.2):
        src = vtk.vtkSphereSource()
        src.SetCenter(point[0], point[1], point[2])
        src.SetRadius(radius)
        return self.drawPdSrc(src)

    def drawSegment(self, seg):
        src = vtk.vtkLineSource()
        src.SetPoint1(seg.A.x, seg.A.y,seg.A.z)
        src.SetPoint2(seg.B.x, seg.B.y, seg.B.z)
        return self.drawPdSrc(src)

    def drawPolyline(self,polyline):
        src = vtk.vtkLineSource()
        points = vtk.vtkPoints()
        for i in range(polyline.count()):
            pt = polyline.point(i)
            points.InsertNextPoint((pt.x, pt.y, pt.z))
        src.SetPoints(points)
        return self.drawPdSrc(src)

    def showPolyline(self, polyline):
        src = vtk.vtkLineSource()
        points = vtk.vtkPoints()
        for i in range(polyline.shape[0]):
            points.InsertNextPoint((polyline[i][0], polyline[i][1], polyline[i][2]))
        src.SetPoints(points)
        return self.drawPdSrc(src)

    def showPolyline2D(self, polyline):
        src = vtk.vtkLineSource()
        points = vtk.vtkPoints()
        for i in range(polyline.shape[0]):
            points.InsertNextPoint((polyline[i][0], polyline[i][1]))
        src.SetPoints(points)
        return self.drawPdSrc(src)

    def showgenPolyline(self, polyline):
        src = vtk.vtkLineSource()
        points = vtk.vtkPoints()
        for i in range(len(polyline)):
            points.InsertNextPoint((polyline[i][0], polyline[i][1], polyline[i][2]))
        src.SetPoints(points)
        return self.drawPdSrc(src)






