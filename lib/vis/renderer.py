# Useful rendering functions from WHAM (some modification)

import cv2
import torch
import numpy as np

from pytorch3d.renderer import (
    PerspectiveCameras,
    TexturesVertex,
    PointLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftPhongShader,
)
from pytorch3d.structures import Meshes
from pytorch3d.structures.meshes import join_meshes_as_scene
from pytorch3d.renderer.cameras import look_at_rotation
from pytorch3d.renderer.camera_conversions import _cameras_from_opencv_projection

from .tools import get_colors, checkerboard_geometry


def overlay_image_onto_background(image, mask, bbox, background):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()

    out_image = background.copy()
    bbox = bbox[0].int().cpu().numpy().copy()
    roi_image = out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]

    roi_image[mask] = image[mask]
    out_image[bbox[1]:bbox[3], bbox[0]:bbox[2]] = roi_image

    return out_image


def update_intrinsics_from_bbox(K_org, bbox):
    device, dtype = K_org.device, K_org.dtype
    
    K = torch.zeros((K_org.shape[0], 4, 4)
    ).to(device=device, dtype=dtype)
    K[:, :3, :3] = K_org.clone()
    K[:, 2, 2] = 0
    K[:, 2, -1] = 1
    K[:, -1, 2] = 1
    
    image_sizes = []
    for idx, bbox in enumerate(bbox):
        left, upper, right, lower = bbox
        cx, cy = K[idx, 0, 2], K[idx, 1, 2]

        new_cx = cx - left
        new_cy = cy - upper
        new_height = max(lower - upper, 1)
        new_width = max(right - left, 1)
        new_cx = new_width - new_cx
        new_cy = new_height - new_cy

        K[idx, 0, 2] = new_cx
        K[idx, 1, 2] = new_cy
        image_sizes.append((int(new_height), int(new_width)))

    return K, image_sizes


def perspective_projection(x3d, K, R=None, T=None):
    if R != None:
        x3d = torch.matmul(R, x3d.transpose(1, 2)).transpose(1, 2)
    if T != None:
        x3d = x3d + T.transpose(1, 2)

    x2d = torch.div(x3d, x3d[..., 2:])
    x2d = torch.matmul(K, x2d.transpose(-1, -2)).transpose(-1, -2)[..., :2]
    return x2d


def compute_bbox_from_points(X, img_w, img_h, scaleFactor=1.2):
    left = torch.clamp(X.min(1)[0][:, 0], min=0, max=img_w)
    right = torch.clamp(X.max(1)[0][:, 0], min=0, max=img_w)
    top = torch.clamp(X.min(1)[0][:, 1], min=0, max=img_h)
    bottom = torch.clamp(X.max(1)[0][:, 1], min=0, max=img_h)

    cx = (left + right) / 2
    cy = (top + bottom) / 2
    width = (right - left)
    height = (bottom - top)

    new_left = torch.clamp(cx - width/2 * scaleFactor, min=0, max=img_w-1)
    new_right = torch.clamp(cx + width/2 * scaleFactor, min=1, max=img_w)
    new_top = torch.clamp(cy - height / 2 * scaleFactor, min=0, max=img_h-1)
    new_bottom = torch.clamp(cy + height / 2 * scaleFactor, min=1, max=img_h)

    bbox = torch.stack((new_left.detach(), new_top.detach(),
                        new_right.detach(), new_bottom.detach())).int().float().T
    
    return bbox


class Renderer():
    def __init__(self, width, height, focal_length, device, faces=None, 
                 bin_size=None, max_faces_per_bin=None):

        self.width = width
        self.height = height
        self.focal_length = focal_length

        self.device = device
        if faces is not None:
            self.faces = torch.from_numpy(
                (faces).astype('int')
            ).unsqueeze(0).to(self.device)

        self.initialize_camera_params()
        self.lights = PointLights(device=device, location=[[0.0, 0.0, -10.0]])
        self.create_renderer(bin_size, max_faces_per_bin)

    def create_renderer(self, bin_size, max_faces_per_bin):
        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                raster_settings=RasterizationSettings(
                    image_size=self.image_sizes[0],
                    blur_radius=1e-5, bin_size=bin_size, 
                    max_faces_per_bin=max_faces_per_bin),
            ),
            shader=SoftPhongShader(
                device=self.device,
                lights=self.lights,
            )
        )

    def initialize_camera_params(self):
        """Hard coding for camera parameters
        TODO: Do some soft coding"""

        # Extrinsics
        self.R = torch.diag(
            torch.tensor([1, 1, 1])
        ).float().to(self.device).unsqueeze(0)

        self.T = torch.tensor(
            [0, 0, 0]
        ).unsqueeze(0).float().to(self.device)

        # Intrinsics
        self.K = torch.tensor(
            [[self.focal_length, 0, self.width/2],
            [0, self.focal_length, self.height/2],
            [0, 0, 1]]
        ).unsqueeze(0).float().to(self.device)
        self.bboxes = torch.tensor([[0, 0, self.width, self.height]]).float()
        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, self.bboxes)

        # self.K_full = self.K  # test
        self.cameras = self.create_camera()

    def create_camera(self, R=None, T=None):
        if R is not None:
            self.R = R.clone().view(1, 3, 3).to(self.device)
        if T is not None:
            self.T = T.clone().view(1, 3).to(self.device)

        return PerspectiveCameras(
            device=self.device,
            R=self.R, #.mT,
            T=self.T,
            K=self.K_full,
            image_size=self.image_sizes,
            in_ndc=False)
    
    def create_camera_from_cv(self, R, T, K=None, image_size=None):
        # R: [1, 3, 3] Tensor
        # T: [1, 3] Tensor
        # K: [1, 3, 3] Tensor
        # image_size: [1, 2] Tensor in HW
        if K is None:
            K = self.K

        if image_size is None:
            image_size = torch.tensor(self.image_sizes)

        cameras = _cameras_from_opencv_projection(R, T, K, image_size)
        lights = PointLights(device=K.device, location=T)

        return cameras, lights
               
    def set_ground(self, length, center_x, center_z):
        device = self.device
        v, f, vc, fc = map(torch.from_numpy, checkerboard_geometry(length=length, c1=center_x, c2=center_z, up="y"))
        v, f, vc = v.to(device), f.to(device), vc.to(device)
        self.ground_geometry = [v, f, vc]


    def update_bbox(self, x3d, scale=2.0, mask=None):
        """ Update bbox of cameras from the given 3d points

        x3d: input 3D keypoints (or vertices), (num_frames, num_points, 3)
        """

        if x3d.size(-1) != 3:
            x2d = x3d.unsqueeze(0)
        else:
            x2d = perspective_projection(x3d.unsqueeze(0), self.K, self.R, self.T.reshape(1, 3, 1))

        if mask is not None:
            x2d = x2d[:, ~mask]

        bbox = compute_bbox_from_points(x2d, self.width, self.height, scale)
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def reset_bbox(self,):
        bbox = torch.zeros((1, 4)).float().to(self.device)
        bbox[0, 2] = self.width
        bbox[0, 3] = self.height
        self.bboxes = bbox

        self.K_full, self.image_sizes = update_intrinsics_from_bbox(self.K, bbox)
        self.cameras = self.create_camera()
        self.create_renderer()

    def render_mesh(self, vertices, background, colors=[0.8, 0.8, 0.8]):
        self.update_bbox(vertices[::50], scale=1.2)
        vertices = vertices.unsqueeze(0)
        
        if colors[0] > 1: colors = [c / 255. for c in colors]
        verts_features = torch.tensor(colors).reshape(1, 1, 3).to(device=vertices.device, dtype=vertices.dtype)
        verts_features = verts_features.repeat(1, vertices.shape[1], 1)
        textures = TexturesVertex(verts_features=verts_features)
        
        mesh = Meshes(verts=vertices,
                      faces=self.faces,
                      textures=textures,)
        
        materials = Materials(
            device=self.device,
            specular_color=(colors, ),
            shininess=0
            )

        results = torch.flip(
            self.renderer(mesh, materials=materials, cameras=self.cameras, lights=self.lights),
            [1, 2]
        )
        image = results[0, ..., :3] * 255
        mask = results[0, ..., -1] > 1e-3

        image = overlay_image_onto_background(image, mask, self.bboxes, background.copy())
        self.reset_bbox()
        return image
    
    
    def render_with_ground(self, verts, faces, colors, cameras, lights):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        
        # (B, V, 3), (B, F, 3), (B, V, 3)
        verts, faces, colors = prep_shared_geometry(verts, faces, colors)
        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        verts = list(torch.unbind(verts, dim=0)) + [gv]
        faces = list(torch.unbind(faces, dim=0)) + [gf]
        colors = list(torch.unbind(colors, dim=0)) + [gc[..., :3]]
        mesh = create_meshes(verts, faces, colors)

        materials = Materials(
            device=self.device,
            shininess=0
        )
        
        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
            
        return image
    
    def render_with_ground_multiple(self, verts_list, faces, colors_list, cameras, lights):
        """
        :param verts (B, V, 3)
        :param faces (F, 3)
        :param colors (B, 3)
        """
        # (B, V, 3), (B, F, 3), (B, V, 3)
        verts_, faces_, colors_ = [], [], []
        for i, verts in enumerate(verts_list):
            colors = colors_list[[i]]
            verts_i, faces_i, colors_i = prep_shared_geometry(verts, faces, colors)
            if i == 0:
                verts_ = list(torch.unbind(verts_i, dim=0))
                faces_ = list(torch.unbind(faces_i, dim=0)) 
                colors_ = list(torch.unbind(colors_i, dim=0))
            else:
                verts_ += list(torch.unbind(verts_i, dim=0))
                faces_ += list(torch.unbind(faces_i, dim=0)) 
                colors_ += list(torch.unbind(colors_i, dim=0)) 

        # (V, 3), (F, 3), (V, 3)
        gv, gf, gc = self.ground_geometry
        verts_ += [gv]
        faces_ += [gf]
        colors_ += [gc[..., :3]]
        mesh = create_meshes(verts_, faces_, colors_)

        materials = Materials(
            device=self.device,
            shininess=0
        )
        results = self.renderer(mesh, cameras=cameras, lights=lights, materials=materials)
        image = (results[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        return image
    
    
def prep_shared_geometry(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (F, 3)
    :param colors (B, 4)
    """
    B, V, _ = verts.shape
    F, _ = faces.shape
    colors = colors.unsqueeze(1).expand(B, V, -1)[..., :3]
    faces = faces.unsqueeze(0).expand(B, F, -1)
    return verts, faces, colors


def create_meshes(verts, faces, colors):
    """
    :param verts (B, V, 3)
    :param faces (B, F, 3)
    :param colors (B, V, 3)
    """
    textures = TexturesVertex(verts_features=colors)
    meshes = Meshes(verts=verts, faces=faces, textures=textures)
    return join_meshes_as_scene(meshes)


def get_global_cameras(verts, device, distance=5, position=(-5.0, 5.0, 0.0)):
    positions = torch.tensor([position]).repeat(len(verts), 1)
    targets = verts.mean(1)
    
    directions = targets - positions
    directions = directions / torch.norm(directions, dim=-1).unsqueeze(-1) * distance
    positions = targets - directions
    
    rotation = look_at_rotation(positions, targets, ).mT
    translation = -(rotation @ positions.unsqueeze(-1)).squeeze(-1)
    
    lights = PointLights(device=device, location=[position])
    return rotation, translation, lights


