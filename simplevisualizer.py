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

def recognize_clicked(shp, *kwargs):
    """ This is the function called every time
    a face is clicked in the 3d view
    """

    for shape in shp:
        print("Face selected: ", shape)
        print("Face selected: ", shape.Location().HashCode(10000))

if __name__ == '__main__':

    # User Defined
    dataset_dir = "data"
    #dataset_dir = "more_data"
    #dataset_dir = "data_suter"
    #dataset_dir = "data_misc"

    occ_display, start_occ_display, add_menu, add_function_to_menu = init_display("pyqt6")
    
    occ_display.SetSelectionModeVertex() # This is the required function
    occ_display.register_select_callback(recognize_clicked)

    add_menu('explore')
    add_function_to_menu('explore', show_random)
    add_function_to_menu('explore', show_next)
    add_function_to_menu('explore', show_previous)
    add_function_to_menu('explore', show_first)
    add_function_to_menu('explore', show_last)

    shape_paths = glob.glob(os.path.join(dataset_dir, '*.step'))
    if len(shape_paths) > 0:
        show_random()

    start_occ_display()
