"""
A visualization tool for the machining feature dataset.

This module requires PythonOCC to run.
"""
import os
import random
import glob

from OCC.Core.Quantity import Quantity_NOC_WHITE, Quantity_Color
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.StepRepr import StepRepr_RepresentationItem
from OCC.Core.AIS import AIS_ColoredShape
from OCC.Display.SimpleGui import init_display


LABELS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

FEAT_NAMES = ["Chamfer", "Through hole", "Triangular passage", "Rectangular passage", "6-sides passage",
                "Triangular through slot", "Rectangular through slot", "Circular through slot",
                "Rectangular through step", "2-sides through step", "Slanted through step", "O-ring", "Blind hole",
                "Triangular pocket", "Rectangular pocket", "6-sides pocket", "Circular end pocket",
                "Rectangular blind slot", "Vertical circular end blind slot", "Horizontal circular end blind slot",
                "Triangular blind step", "Circular blind step", "Rectangular blind step", "Round", "Stock"]

COLORS = {"Chamfer": 0, "Through hole": 490, "Triangular passage": 500, "Rectangular passage": 470, "6-sides passage": 100,
                "Triangular through slot": 120, "Rectangular through slot": 140, "Circular through slot": 160,
                "Rectangular through step": 180, "2-sides through step": 200, "Slanted through step": 220, "O-ring": 240, "Blind hole": 260,
                "Triangular pocket": 280, "Rectangular pocket": 300, "6-sides pocket": 320, "Circular end pocket": 340,
                "Rectangular blind slot": 360, "Vertical circular end blind slot": 380, "Horizontal circular end blind slot": 400,
                "Triangular blind step": 420, "Circular blind step": 440, "Rectangular blind step": 460, "Round": 480, "Stock": 60}


def read_step_with_labels(filename):
    """Reads STEP file with labels on each B-Rep face."""
    if not os.path.exists(filename):
        print(filename, ' not exists')
        return

    reader = STEPControl_Reader()
    reader.ReadFile(filename)
    reader.TransferRoots()
    shape = reader.OneShape()

    treader = reader.WS().TransferReader()

    id_map = {}
    topo = TopologyExplorer(shape)
    faces = list(topo.faces())

    for face in faces:
        item = treader.EntityFromShapeResult(face, 1)
        if item is None:
            print(face)
            continue
        item = StepRepr_RepresentationItem.DownCast(item)
        name = item.Name().ToCString()
        if name:
            nameid = name
            id_map[face] = int(nameid)

    return shape, id_map



def display():
    global shape_index
    global shape_paths

    shape, face_ids = read_step_with_labels(shape_paths[shape_index])
    if shape == None:
        return

    occ_display.EraseAll()
    AIS = AIS_ColoredShape(shape)

    for face, label in face_ids.items():
        feat_name = FEAT_NAMES[label]
        AIS.SetCustomColor(face, colors[feat_name])

    occ_display.Context.Display(AIS, True)
    occ_display.View_Iso()
    occ_display.FitAll()

    print(f"STEP: {shape_paths[shape_index]}")


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


if __name__ == '__main__':
    # User Defined
    dataset_dir = "step"

    colors = {name: Quantity_Color(COLORS[name]) for name in COLORS}

    occ_display, start_occ_display, add_menu, add_function_to_menu = init_display()

    add_menu('explore')
    add_function_to_menu('explore', show_random)
    add_function_to_menu('explore', show_next)
    add_function_to_menu('explore', show_previous)
    add_function_to_menu('explore', show_first)
    add_function_to_menu('explore', show_last)

    shape_paths = glob.glob(os.path.join(dataset_dir, '*.step'))

    print(len(shape_paths), 'shapes')

    if len(shape_paths) > 0:
        show_random()

    start_occ_display()
