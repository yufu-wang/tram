from optparse import Option
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
#os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
import torch
import numpy as np
import pyrender
import trimesh
import cv2
from typing import List, Optional

# RGB-A
blue = (0.3, 0.5, 0.9, 1.0)
green = (0.45, 0.75, 0.533, 1.0)
yellow = (0.88, 0.85, 0.528, 1.0)


class Renderer:

    def __init__(self, faces, color=(0.3, 0.5, 0.9, 1.0), size=None):
        """
        Wrapper around the pyrender renderer to render SMPL meshes.
        Args:
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """

        self.size = size
        self.faces = faces

        self.light_nodes = create_raymond_lights()

        self.material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.1,
            alphaMode='OPAQUE',
            baseColorFactor=color)
        self.renderer = None


    def init_renderer(self, height=None, width=None, image=None):
        if height is None or width is None:
            height, width = image.shape[:2]
        self.renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                                  viewport_height=height,
                                                  point_size=1.0)


    def __call__(self, vertices, camera_translation, image=None, focal=None, center=None, return_depth=False) :
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (np.array): Array of shape (H, W, 3) containing the image crop with normalized pixel values.
        """

        height, width = image.shape[:2]

        if self.renderer is None:
            renderer = pyrender.OffscreenRenderer(viewport_width=width,
                                                  viewport_height=height,
                                                  point_size=1.0)
        else:
            renderer = self.renderer


        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.5, 0.5, 0.5))

        if focal is None:
            focal = np.sqrt(width**2 + height**2)

        if center is None:
            center = [width / 2., height / 2.]

        camera_translation = np.array(camera_translation) # also make a copy
        camera_translation[0] *= -1.

        # Create mesh
        if len(vertices.shape) == 2:
            vertices = vertices[None]

        for vert in vertices:
            mesh = trimesh.Trimesh(vert, self.faces, process=False)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])

            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=self.material, smooth=True)
            scene.add(mesh, 'mesh')

        # Create camera
        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera = pyrender.IntrinsicsCamera(fx=focal, fy=focal,
                                           cx=center[0], cy=center[1], zfar=1000)
        scene.add(camera, pose=camera_pose)

        # Create light
        for node in self.light_nodes: scene.add_node(node)

        # Render
        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.NONE)

        
        # Composite
        if image is None:
            output_img = color[:, :, :3]
        else:
            valid_mask = (rend_depth > 0)[:, :, np.newaxis].astype(np.uint8)
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image) 
        
        del scene

        if return_depth:
            return output_img, rend_depth
        else:
            return output_img


def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes
