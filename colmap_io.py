import numpy as np
from collections import namedtuple

Image = namedtuple("Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Camera = namedtuple("Camera", ["id", "model", "width", "height", "params"])

def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1]*qvec[2] - 2 * qvec[0]*qvec[3],
         2 * qvec[0]*qvec[2] + 2 * qvec[1]*qvec[3]],
        [2 * qvec[1]*qvec[2] + 2 * qvec[0]*qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2]*qvec[3] - 2 * qvec[0]*qvec[1]],
        [2 * qvec[1]*qvec[3] - 2 * qvec[0]*qvec[2],
         2 * qvec[0]*qvec[1] + 2 * qvec[2]*qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]
    ])

def read_points3D_text(path):
    xyzs, rgbs, errors = [], [], []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            elems = line.split()
            xyzs.append([float(e) for e in elems[1:4]])
            rgbs.append([int(e) for e in elems[4:7]])
            errors.append(float(elems[7]))
    return np.array(xyzs), np.array(rgbs), np.array(errors)

def read_intrinsics_text(path):
    cameras = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            elems = line.split()
            cam_id = int(elems[0])
            model = elems[1]
            width, height = int(elems[2]), int(elems[3])
            params = np.array([float(x) for x in elems[4:]])
            cameras[cam_id] = Camera(cam_id, model, width, height, params)
    return cameras

def read_extrinsics_text(path):
    images = {}
    with open(path, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith("#") or line.strip() == "":
                continue
            elems = line.strip().split()
            image_id = int(elems[0])
            qvec = np.array([float(e) for e in elems[1:5]])
            tvec = np.array([float(e) for e in elems[5:8]])
            camera_id = int(elems[8])
            name = elems[9]
            elems2 = f.readline().strip().split()
            xys = np.column_stack((map(float, elems2[0::3]), map(float, elems2[1::3])))
            point3D_ids = np.array([int(i) for i in elems2[2::3]])
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images
