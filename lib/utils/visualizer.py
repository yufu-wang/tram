import cv2
import numpy as np
import torch
from plyfile import PlyData, PlyElement
from matplotlib.colors import Normalize
import matplotlib.cm as cm


def draw_kpts(img, kpts, r=5, thickness=5, color=(255,0,0), confidence=1e-6):
    if isinstance(img, np.ndarray):
        img = img.copy().astype(np.uint8)
    if isinstance(img, torch.Tensor):
        img = img.numpy()
        img = img.copy().astype(np.uint8)
        
    for kpt in kpts:
        if len(kpt)>2:
            x, y, c = kpt
        else:
            x, y = kpt
            c = 1

        if c >= confidence:
            cv2.circle(img, (int(x), int(y)), r, color, thickness)

    return img

def draw_boxes(img, boxes, thickness=5, color=(0,255,0)):
    img_box = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        img_box = cv2.rectangle(img_box, (int(x1),int(y1)), (int(x2),int(y2)), 
                                color, thickness)
    return img_box


def to_rgb(grey, cmap='YlGnBu', resize=[224, 224], normalize=True):
    # cmap_list = ['YlGnBu', 'coolwarm', 'RdBu']
    g = np.array(grey)
    cmap = cm.get_cmap(cmap)

    if normalize:
        norm = Normalize(vmin=g.min(), vmax=g.max())
        g = norm(g)
    rgb = cmap(g)[:,:,:3]

    if resize is not None:
        rgb = cv2.resize(rgb, resize)

    rgb = (rgb * 255).astype(int)
    return rgb


def to_rgb_norm(grey, cmap='YlGnBu', resize=[224, 224], min_v=0.0, max_v=1.0, normalize=True):
    # cmap_list = ['YlGnBu', 'coolwarm', 'RdBu']
    g = np.array(grey)
    cmap = cm.get_cmap(cmap)

    if normalize:
        norm = Normalize(vmin=min_v, vmax=max_v)
        g = norm(g)
    rgb = cmap(g)[:,:,:3]

    if resize is not None:
        rgb = cv2.resize(rgb, resize)

    rgb = (rgb * 255).astype(int)
    return rgb

    
### Save for visualization
def save_ply(vert, face=None, color=None, filename='file.ply'):

    # Colors
    if color is None:
        vtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    else:
        vtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        vert = np.concatenate([vert, color], axis=-1)

    # Vertices
    if isinstance(vert, np.ndarray):
        vert = vert.tolist()
    vert = [tuple(v) for v in vert]
    vert = np.array(vert, dtype=vtype)
    vert = PlyElement.describe(vert, 'vertex')

    # Faces
    if face is not None:
        if isinstance(face, np.ndarray):
            face = face.tolist()
        face = [(face[i], 255, 255, 255) for i in range(len(face))]
        face = np.array(face, dtype=[('vertex_indices', 'i4', (3,)),
                                     ('red', 'u1'),
                                     ('green', 'u1'),
                                     ('blue', 'u1')])
        face = PlyElement.describe(face, 'face')
    
    # Save
    if face is not None:
        with open(filename, 'wb') as f:
            PlyData([vert, face]).write(f)
    else:
        with open(filename, 'wb') as f:
            PlyData([vert]).write(f)


def read_ply(plyfile):
    plydata = PlyData.read(plyfile)
    v = plydata['vertex'].data
    v = [list(i) for i in v]
    v = np.array(v)
    f = plydata['face'].data
    f = [list(i) for i in f]
    f = np.array(f).squeeze()
    return v, f


# from transforms3d.euler import euler2mat

# def novel_view(s_out, angle=-0.25*np.pi, axis='y', trans=8):
#     vertices = s_out.vertices.clone().cpu()
#     joints = s_out.joints.clone().cpu()

#     j3d = joints[:, 25:]
#     pelvis3d = j3d[:, [14]]
#     verts = vertices - pelvis3d

#     if axis=='x':
#         rot = euler2mat(angle, 0, 0, "sxyz")
#     elif axis=='y':
#         rot = euler2mat(0, angle, 0, "sxyz")
#     else:
#         rot = euler2mat(0, 0, angle, "sxyz")

#     rot = torch.from_numpy(rot).float()
#     verts = torch.einsum('ij, bvj->bvi', rot, verts)

#     # verts[:,:,1] -= 0.1
#     verts[:,:,2] += trans

#     return verts
        

