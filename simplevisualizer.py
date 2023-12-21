"""
A simple visualization tool for the machining feature dataset.
This module requires PythonOCC to run.
"""

import os
import random
import glob

from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Display.SimpleGui import init_display

from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_BSplineSurface
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Display.SimpleGui import init_display
from OCC.Extend.DataExchange import read_step_file
from OCC.Extend.TopologyUtils import TopologyExplorer

shape = None

def read_step(filename):
    """Reads STEP file with labels on each B-Rep face."""
    if not os.path.exists(filename):
        print(filename, ' not exists')
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    return shape

def display():
    global shape
    global shape_index
    global shape_paths

    shape = read_step(shape_paths[shape_index])
    if shape == None:
        return

    occ_display.EraseAll()
    AIS = AIS_ColoredShape(shape)

    occ_display.Context.Display(AIS, True)
    occ_display.View_Iso()
    occ_display.FitAll()


def show_first():
    global shape_index
    shape_index = 0
    display()


def show_last():
    global shape_index
    global shape_paths

    shape_index = len(shape_paths) - 1
    display()


def show_next():
    global shape_index
    global shape_paths

    shape_index = (shape_index + 1) % len(shape_paths)
    display()


def show_previous():
    global shape_index
    global shape_paths

    shape_index = (shape_index - 1 + len(shape_paths)) % len(shape_paths)
    display()

def show_random():
    global shape_index
    global shape_paths

    shape_index = random.randrange(0, len(shape_paths))
    display()


def recognize_face(a_face):
    """Takes a TopoDS shape and tries to identify its nature
    whether it is a plane a cylinder a torus etc.
    if a plane, returns the normal
    if a cylinder, returns the radius
    """
    if not type(a_face) is TopoDS_Face:
        print("Please hit the 'G' key to switch to face selection mode")
        return False
    surf = BRepAdaptor_Surface(a_face, True)
    surf_type = surf.GetType()
    if surf_type == GeomAbs_Plane:
        print("Identified Plane Geometry")
        # look for the properties of the plane
        # first get the related gp_Pln
        gp_pln = surf.Plane()
        location = gp_pln.Location()  # a point of the plane
        normal = gp_pln.Axis().Direction()  # the plane normal
        # then export location and normal to the console output
        print(
            "--> Location (global coordinates)",
            location.X(),
            location.Y(),
            location.Z(),
        )
        print("--> Normal (global coordinates)", normal.X(), normal.Y(), normal.Z())
    elif surf_type == GeomAbs_Cylinder:
        print("Identified Cylinder Geometry")
        # look for the properties of the cylinder
        # first get the related gp_Cyl
        gp_cyl = surf.Cylinder()
        location = gp_cyl.Location()  # a point of the axis
        axis = gp_cyl.Axis().Direction()  # the cylinder axis
        # then export location and normal to the console output
        print(
            "--> Location (global coordinates)",
            location.X(),
            location.Y(),
            location.Z(),
        )
        print("--> Axis (global coordinates)", axis.X(), axis.Y(), axis.Z())
    elif surf_type == GeomAbs_BSplineSurface:
        print("Identified BSplineSurface Geometry")
        # gp_bsrf = surf.Surface()
        # degree = gp_bsrf.NbUKnots()
        # TODO use a model that provided BSplineSurfaces, as1_pe_203.stp only contains
        # planes and cylinders
    else:
        # TODO there are plenty other type that can be checked
        # see documentation for the BRepAdaptor class
        # https://www.opencascade.com/doc/occt-6.9.1/refman/html/class_b_rep_adaptor___surface.html
        print(surf_type, "recognition not implemented")

def recognize_clicked(shp, *kwargs):
    """ This is the function called every time
    a face is clicked in the 3d view
    """

    for shape in shp:
        print("Face selected: ", shape)
        print("Hash: ", shape.__hash__())
        recognize_face(shape)

def recognize_batch(event=None):
    """Menu item : process all the faces of a single shape"""
    global shape

    # loop over faces only
    for face in TopologyExplorer(shape).faces():
        # call the recognition function
        recognize_face(face)

if __name__ == '__main__':

    # User Defined
    #dataset_dir = "data"
    #dataset_dir = "more_data"
    #dataset_dir = "data_suter"
    #dataset_dir = "data_misc"
    dataset_dir = "data_autogen"

    occ_display, start_occ_display, add_menu, add_function_to_menu = init_display("pyqt6")
    
    #occ_display.SetSelectionModeVertex() # This is the required function
    occ_display.SetSelectionModeFace()
    occ_display.register_select_callback(recognize_clicked)

    add_menu('explore')
    add_function_to_menu('explore', show_random)
    add_function_to_menu('explore', show_next)
    add_function_to_menu('explore', show_previous)
    add_function_to_menu('explore', show_first)
    add_function_to_menu('explore', show_last)

    add_menu("recognition")
    add_function_to_menu("recognition", recognize_batch)

    shape_paths = glob.glob(os.path.join(dataset_dir, '*.step'))
    if len(shape_paths) > 0:
        show_random()

    start_occ_display()
