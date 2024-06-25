from rit_window import *
from cgI_engine import *
from vertex import *
from clipper import *
from shapes_new import *
from PIL import Image


def default_action ():
   #create your scene here
   myEngine.win.clearFB(0.25,0,0.35)
   im = Image.open("christmas.jpg")
   im2 = Image.open("christmasLeaves.jpg")
   im3 = Image.open("wood.jpg")
   myEngine.setCamera([0.0, 0.0, 2.0], [0, 0, 20], [0, 1, 0])
   myEngine.setOrtho(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0)
   myEngine.pushTransform()
   myEngine.translate(-0.55, -15, 1.0)
   myEngine.rotatey(30)
   myEngine.rotatex(30)
   myEngine.scale(0.9, 0.2, 1.2)
   myEngine.setLight([-1.0, 1.0, 2.0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([1.0, 1.0, 1.0])
   myEngine.pushTransform()
   myEngine.drawTrianglesPhong(cylinder_new, cylinder_new_idx, cylinder_new_normals, [1.9, 1.5, 1.0], [1.0, 1.0, 1.0], [0.3, 0.9, 0.8], 10,
                               False)
   myEngine.popTransform()
   myEngine.translate(0, -2.2, 1.0)
   myEngine.pushTransform()
   myEngine.rotatey(30)
   myEngine.rotatex(30)
   myEngine.rotatez(180)
   myEngine.scale(0.6, 0.8, 0.4)
   myEngine.pushTransform()
   myEngine.drawTrianglesTextures(cylinder_new, cylinder_new_idx, cylinder_new_uv, im3)
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.rotatey(30)
   myEngine.rotatex(30)
   myEngine.scale(2.5, 3.5, 1.5)
   myEngine.pushTransform()
   myEngine.drawTrianglesTextures(cone_new, cone_new_idx, cone_new_uv, im2)
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.setCamera([0.0, 0.0, 2.0], [0, 0, 20], [0, 1, 0])
   myEngine.setOrtho(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0)
   myEngine.setLight([-1.0, 1.0, 2.0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([1.0, 1.0, 1.0])
   myEngine.translate(0, 2.0, 1.0)
   myEngine.pushTransform()
   myEngine.drawTrianglesPhong(sphere_new, sphere_new_idx, sphere_new_normals, [1.5, 1.5, 0.0], [1.0, 1.0, 1.0], [0.3, 0.9, 0.8], 10,
                               False)
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.translate(-1.75, -2.5, -4.5)
   myEngine.pushTransform()
   myEngine.rotatey(30)
   myEngine.rotatex(30)
   myEngine.rotatez(180)
   myEngine.scale(1.0, 1.0, 1.0)
   myEngine.pushTransform()
   myEngine.drawTrianglesTextures(cube_new, cube_new_idx, cube_new_uv, im)
   myEngine.popTransform()
   myEngine.popTransform()
   myEngine.setCamera([0.0, 0.0, 2.0], [0, 0, 20], [0, 1, 0])
   myEngine.setOrtho(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0)
   myEngine.setLight([-1.0, 1.0, 2.0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([1.0, 1.0, 1.0])
   myEngine.translate(2.0, 0, 0)
   myEngine.scale(2.0, 2.0, 2.0)
   myEngine.pushTransform()
   myEngine.drawTrianglesPhong(sphere_new, sphere_new_idx, sphere_new_normals, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.9, 0.4, 0.4], 10,
                               False)
   myEngine.popTransform()
   myEngine.setCamera([0.0, 0.0, 2.0], [0, 0, 20], [0, 1, 0])
   myEngine.setOrtho(-3.0, 3.0, -3.0, 3.0, -3.0, 3.0)
   myEngine.setLight([-1.0, 1.0, 2.0], [1.0, 1.0, 1.0])
   myEngine.setAmbient([1.0, 1.0, 1.0])
   myEngine.translate(4, 2.0, 0)
   myEngine.scale(0.5, 0.5, 0.5)
   myEngine.pushTransform()
   myEngine.drawTrianglesPhong(sphere_new, sphere_new_idx, sphere_new_normals, [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [0.9, 0.4, 0.4], 10,
                               False)
   myEngine.clearModelTransform()
   myEngine.translate(0, 1.0, 0.0)
   myEngine.scale(0.75, 0.75, 0.75)
   myEngine.pushTransform()
   myEngine.drawTrianglesMyTextures(cone_new, cone_new_idx, cone_new_uv, 50)
   
window = RitWindow(800, 800)
myEngine = CGIengine (window, default_action)

def main():
    window.run (myEngine)
    



if __name__ == "__main__":
    main()
