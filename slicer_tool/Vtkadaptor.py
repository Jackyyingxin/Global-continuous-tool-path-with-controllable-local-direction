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



    def write_image(self,file_name, ren_win, rgba=True):
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
        camera = self.renderer.GetActiveCamera()

        "上面的代码都是用来调摄像头大小和距离的"
        ren_win.Render()

        windowto_image_filter = vtkWindowToImageFilter()
        windowto_image_filter.SetInput(ren_win)
        windowto_image_filter.SetScale(1)  # image quality
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
            parent = Path(file_name).resolve().parent
            path = Path(parent) / file_name
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
    def drawActor(self,actor):
        self.renderer.AddActor(actor)
        return actor
    def drawPdSrc(self,pdSrc):
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(pdSrc.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        return self.drawActor(actor)
    def drawPoint(self,point, radius =0.1):
        src = vtk.vtkSphereSource()
        src.SetCenter(point.x, point.y, point.z)
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
    def drawPolygon3f(self,Polygon):
        src = vtk.vtkLineSource()
        points = vtk.vtkPoints()
        for vertex in Polygon.vertices:
            pt = vertex
            points.InsertNextPoint((pt.coord[0],pt.coord[1],pt.coord[2]))
        src.SetPoints(points)
        return self.drawPdSrc(src)










