#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 23:46:17 2020

@author: vriplabpc3
"""

import os, sys
import bpy
from mathutils import Matrix, Vector
import numpy as np
import random
import json
import ipdb
import matplotlib.pyplot as plt
import math
from math import radians
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(cur_dir)
from render_utils import *

'''
# Define pose grid for 20 vertex on the regular dodecahedron
phi = (1. + np.sqrt(5) / 2)
dodecahedron_vertex_coord = np.array(
    [[-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
     [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
     [0, -phi, -1/phi], [0, -phi, 1/phi], [0, phi, -1/phi], [0, phi, 1/phi],
     [-1/phi, 0, -phi], [-1/phi, 0, phi], [1/phi, 0, -phi], [1/phi, 0, phi],
     [-phi, -1/phi, 0], [-phi, 1/phi, 0], [phi, -1/phi, 0], [phi, 1/phi, 0]]
)
'''
# Define pose grid for 8 vertex on the regular cube

cube__vertex_coord = np.array(
    [[0,-1,0],[-0.1,-0.95,0],[-0.2,-0.9,0],[-0.3,-0.85,0],[-0.4,-0.8,0],[-0.5,-0.75,0],[-0.6,-0.7,0],[-0.7,-0.6,0],[-0.8,-0.5,0],\
     [-0.85,-0.3,0],[-0.9,-0.1,0],[-0.91,0,0],[-0.9,0.1,0],[-0.85,0.3,0],[-0.8,0.5,0],[-0.7,0.6,0],[-0.6,0.7,0],[-0.5,0.75,0],\
     [-0.4,0.8,0],[-0.3,0.85,0],[-0.2,0.9,0],[-0.1,0.95,0],[0,1,0],\
     [0,-0.95,-0.1],[0,-0.9,-0.2],[0,-0.85,-0.3],[0,-0.8,-0.4],[0,-0.7,-0.5],[0,-0.6,-0.6],[0,-0.5,-0.7],[0,-0.4,-0.8],[0,-0.3,-0.9],[0,-0.2,-1],\
     [0,0.95,-0.1],[0,0.9,-0.2],[0,0.85,-0.3],[0,0.8,-0.4],[0,0.7,-0.5],[0,0.6,-0.6],[0,0.5,-0.7],[0,0.4,-0.8],[0,0.3,-0.9],[0,0.2,-1],\
     [-0.1,-0.95,-0.1],[-0.15,-0.9,-0.2],[-0.2,-0.85,-0.3],[-0.25,-0.8,-0.4],[-0.3,-0.7,-0.5],[-0.35,-0.6,-0.6],[-0.4,-0.5,-0.7],[-0.45,-0.4,-0.8],[-0.5,-0.3,-0.9],[-0.55,-0.2,-1],\
     [-0.1,0.95,-0.1],[-0.15,0.9,-0.2],[-0.2,0.85,-0.3],[-0.25,0.8,-0.4],[-0.3,0.7,-0.5],[-0.35,0.6,-0.6],[-0.4,0.5,-0.7],[-0.45,0.4,-0.8],[-0.5,0.3,-0.9],[-0.55,0.2,-1],\
    ]
)
'''
#REVERSE ROTATION
cube__vertex_coord = np.array(
    [[0.11,-1.16,0],[0.22,-1.09,0],[0.33,-1.07,0],[0.45,-1.03,0],[0.57,-0.99,0],[0.68,-0.94,0],[0.8,-0.85,0],[0.9,-0.8,0],\
     [0.97,-0.7,0],[1.07,-0.6,0],[1.15,0.4,0],[1.07,0.6,0],[0.97,0.7,0],[0.9,0.8,0],[0.8,0.85,0],[0.68,0.94,0],[0.57,0.99,0],[0.45,1.03,0]\
     ,[0.33,1.07,0],[0.22,1.09,0],[0.11,1.16,0]]
)
'''
'''
#SMALL SCALE
cube__vertex_coord = np.array(
        [[0,-1.2,0],[-0.11,-1.16,0],[-0.22,-1.09,0],[-0.33,-1.07,0],[-0.45,-1.03,0],[-0.57,-0.99,0],[-0.68,-0.94,0],[-0.8,-0.85,0],[-0.9,-0.8,0],\
     [-0.97,-0.7,0],[-1.07,-0.6,0],[-1.15,0.4,0],[-1.07,0.6,0],[-0.97,0.7,0],[-0.9,0.8,0],[-0.8,0.85,0],[-0.68,0.94,0],[-0.57,0.99,0],[-0.45,1.03,0]\
     ,[-0.33,1.07,0],[-0.22,1.09,0],[-0.11,1.16,0],[0,1.2,0]])
'''
'''
#LARGE SCALE
cube__vertex_coord = np.array(
        [[0,-0.8,0],[-0.1,-0.78,0],[-0.2,-0.75,0],[-0.28,-0.73,0],[-0.35,-0.7,0],[-0.4,-0.68,0],[-0.45,-0.65,0],[-0.48,-0.63,0],[-0.5,-0.61,0],\
     [-0.53,-0.59,0],[-0.56,-0.56,0],[-0.79,0,0],[-0.56,0.56,0],[-0.53,0.59,0],[-0.5,0.61,0],[-0.48,0.63,0],[-0.45,0.65,0],[-0.4,0.68,0],\
     [-0.35,0.7,0],[-0.28,0.73,0],[-0.2,0.75,0],[-0.1,0.78,0],[0,0.8,0]])
'''
# Add constraint to the camera
def parent_obj_to_camera(b_camera):
    # set the parenting to the origin
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


# Setup the camera
def setup_camera(scene):
    cam = scene.objects['Camera']
    cam.location = (0,1,0)#default
    #cam.rotation_euler = (0.0,0.0,radians(180))
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    #cam_constraint.up_axis = 'UP_X'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty
    return cam


# Import 3D model from .obj files
def import_model(model_file, axis_forward=None, axis_up=None):
    if axis_forward is not None and axis_up is not None:
        bpy.ops.import_scene.obj(filepath=model_file, axis_forward=axis_forward, axis_up=axis_up)
    else:
        bpy.ops.import_scene.obj(filepath=model_file)
    model_name = model_file.split('/')[-1].split('.')[0]
    return model_name


# Normalize the 3D model
def normalize_model(obj):
    bpy.context.scene.objects.active = obj
    bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
    obj.location = (0, 0, 0)
    max_dim = max(obj.dimensions)
    obj.dimensions = obj.dimensions / (max_dim*2)
    # obj.dimensions /=np.linalg.norm(obj.dimensions)
       
# Create normalized coordinate map as a color map
def create_coord_map(obj):
    mesh = obj.data
    vert_list = mesh.vertices
    vcos = [obj.matrix_world * v.co for v in vert_list]
    x, y, z = [[v[i] for v in vcos] for i in range(3)]
    
    #print(len(z))
    #sys.exit()
    min_x, min_y, min_z = min(x), min(y), min(z)
    size_x, size_y, size_z = max(x) - min(x), max(y) - min(y), max(z) - min(z)
    D = math.sqrt((size_x**2)+(size_y**2)+(size_z**2))
    print('D',D)
    # get the color map to create as coordinate map
    if mesh.vertex_colors:
        color_map = mesh.vertex_colors.active
    else:
        color_map = mesh.vertex_colors.new()
    # apply the corresponding color to each vertex
    i = 0
    for poly in mesh.polygons:
        for idx in poly.loop_indices:
            #print('poly',idx)
            loop = mesh.loops[idx]
            v = vert_list[loop.vertex_index]
            '''
            r = (v.co.x - min_x) / size_x + 0.5
            g = (v.co.y - min_y) / size_y + 0.5
            b = (v.co.z - min_z) / size_z + 0.5
            
            '''
            r = (-v.co.y) / D + 0.5
            g = (v.co.z) / D + 0.5
            b = (v.co.x) / D + 0.5
            
            #print('x',v.co.x)
            #sys.exit()
            color_map.data[i].color = (r, g, b)
            '''
            if i<=10:
                print(color_map.data[i].color)
            else:
                break
            '''
            i += 1
            #print('i',i)
           

    mat = bpy.data.materials.new('vertex_material')
    mat.use_shadeless = True
    mat.use_vertex_color_paint = True
    if mesh.materials:
        mesh.materials[0] = mat
    else:
        mesh.materials.append(mat)


class RenderMachine:
    """Creates a python blender render machine.
    model_files: a list containing all the obj files
    out_dir: where to save the render results
    rad: lamp radiance to adjust the lightness
    clip_end: rendering range in mm
    """
    def __init__(self,
                 model_file, out_dir, rad=3000, clip_end=100, height=128, width=128):
        # Setting up the environment
        remove_obj_lamp_and_mesh(bpy.context) 
        self.scene = bpy.context.scene
        self.depthFileOutput, self.normalFileOutput = setup_env(self.scene, True, True, height, width, clip_end)

        self.camera = setup_camera(self.scene)
        self.lamp = make_lamp(rad)
        self.scene.world.horizon_color=(255,255,255)
        self.height, self.width = height, width

        # Import 3D models and create the normalized object coordinate space as material
        
        #self.model = import_model(model_file, axis_forward='Y', axis_up='Z')
        
        #############
        mod=bpy.ops.import_scene.obj(filepath="/home/anju/Anju/FINAL_yr_project/skin_colour_hand_right_100.obj",axis_forward='Y', axis_up='Z')
        #mod=bpy.ops.import_scene.obj(filepath="/home/anju/Anju/FINAL_yr_project/NOCS_CVPR2019-master/data/obj_models/real_test/laptop_air_xin_norm.obj",axis_forward='Y', axis_up='Z')
        #mod=bpy.ops.import_scene.obj(filepath="/home/anju/Anju/FINAL_yr_project/NOCS_CVPR2019-master/data/obj_models/real_test/can_lotte_milk_norm.obj",axis_forward='Y', axis_up='Z')
        obj_object = bpy.context.selected_objects[0]
        self.model = obj_object.name
        ###########3
         
        #print('mean',sum(a)/len(a))
        normalize_model(bpy.data.objects[self.model])
        K,s_u,s_v=get_calibration_matrix_K_from_blender(bpy.data.objects[0].data)
        print(K,'s_u:',s_u,'s_v:',s_v)
        print(bpy.data.objects[self.model].dimensions)
        
        #bpy.ops.export_scene.obj(filepath="trying_to_save.obj",  use_selection=False, use_animation=False, use_mesh_modifiers=True, use_edges=True, use_smooth_groups=False, use_smooth_groups_bitflags=False, use_normals=True, use_uvs=True, use_materials=True, use_triangles=False, use_nurbs=False, use_vertex_groups=False, use_blen_objects=True, group_by_object=False, group_by_material=False, keep_vertex_order=False, global_scale=1.0, path_mode='AUTO', axis_forward='Y', axis_up='Z')
        create_coord_map(bpy.data.objects[self.model]) 
        #self.normalFileOutput.base_path = os.path.join(out_dir, self.model, 'original')
        # Output setting
        self.out_dir = os.path.join(out_dir, self.model, 'nocs')
        self.depthFileOutput.base_path = os.path.join(out_dir, self.model, 'depth')
        #self.normalFileOutput.base_path = os.path.join(out_dir, self.model, 'normal')
        self.scene.render.image_settings.file_format = 'PNG'
        #self.depthFileOutput.format.file_format = 'OPEN_EXR'

    def render_grid_pose(self, pose_grid):
        x, y, z = bpy.context.active_object.dimensions
        print('x',x,'y',y,'z',z)
        for i in range(pose_grid.shape[0]):
            self.camera.location = pose_grid[i]
            self.lamp.location = pose_grid[i]
            i=i+67
            self.scene.render.filepath = os.path.join(self.out_dir, '{:04d}_coord'.format(i))
            self.depthFileOutput.format.color_depth = '16'
            self.depthFileOutput.file_slots[0].path = '{:04d}_depth'.format(i)
            #self.normalFileOutput.file_slots[0].path = '{:04d}_normal'.format(i)
            render_without_output(use_antialiasing=True)
            


if __name__ == '__main__':
    # input and output directory
    model_dir = '/home/anju/Anju/FINAL_yr_project'
    out_dir = '/home/anju/Anju/FINAL_yr_project/NOCS_CVPR2019-master1/output_NOCSmap'

    model_files = [name for name in os.listdir(model_dir) if
                   os.path.getsize(os.path.join(model_dir, name)) / (2 ** 20) < 50]
    # model_file = random.choice(model_files)
    model_file = 'skin_colour_hand_right_100.obj'
    model_file = os.path.join(model_dir, model_file)

    render_machine = RenderMachine(model_file, out_dir, rad=30, height=480, width=640)

    render_machine.render_grid_pose(cube__vertex_coord)
    
    os.system('rm blender_render.log')