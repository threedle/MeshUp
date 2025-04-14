import os
import numpy as np
from plyfile import PlyData
from nvdiffmodeling.src import obj


def load_ply(file_name):
    ply_data = PlyData.read(file_name)
    vertices = ply_data["vertex"]
    vertices = np.vstack([vertices["x"], vertices["y"], vertices["z"]]).T
    data = {"vertices": vertices}

    faces = np.vstack(ply_data["face"]["vertex_indices"])
    data["faces"] = faces

    try:
        vertex_quality = np.vstack(ply_data["vertex"]["quality"])
        vertex_selection = np.float32(vertex_quality > 0)
        data["vertex_selection"] = vertex_selection
    except ValueError:
        data["vertex_selection"] = None
        print("The ply file %s does not contain quality property for vertex selection." % file_name)

    try:
        face_quality = np.vstack(ply_data["face"]["quality"])
        face_selection = np.float32(face_quality > 0)
        data["face_selection"] = face_selection
    except ValueError:
        data["face_selection"] = None
        print("The ply file %s does not contain quality property for face selection." % file_name)

    return data


def write_ply(file, verts, faces, color=None):
    with open(file, 'w+') as f:
        # header
        f.write("ply\nformat ascii 1.0\n")
        f.write("element vertex {}\n".format(verts.shape[0]))
        f.write("property float32 x\nproperty float32 y\nproperty float32 z\n")
        if color is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("element face {}\n".format(faces.shape[0]))
        f.write("property list uint8 int32 vertex_index\n")
        f.write("end_header\n")

        # vertices
        for vi, v in enumerate(verts):
            if color is not None:
                f.write("%f %f %f %d %d %d\n" % (v[0], v[1], v[2], color[vi][0], color[vi][1], color[vi][2]))
            else:
                f.write("%f %f %f\n" % (v[0], v[1], v[2]))

        # faces
        for face in faces:
            f.write("3 %d %d %d\n" % (face[0], face[1], face[2]))


def load_mesh(base_path, load_deformed=True, epoch=2500):
    if load_deformed:
        mesh_path = os.path.join(base_path, "mesh_final", "mesh.obj")
    else:
        mesh_path = os.path.join(base_path, "tmp", "mesh.obj")

    mesh = obj.load_obj(mesh_path)
    mesh_verts = mesh.v_pos.cpu().numpy()
    mesh_faces = mesh.t_pos_idx.cpu().numpy()

    data = {"vertices": mesh_verts, "faces": mesh_faces}

    if load_deformed:
        mesh_deformed_jacobians_path = os.path.join(base_path, "jacobians", "jacobians_epoch_%d.npy" % epoch)
        mesh_deformed_jacobians = np.load(mesh_deformed_jacobians_path)
        data["jacobians"] = mesh_deformed_jacobians
    
    return data

def load_selection(sel_path, invert_sel=False):
    data = load_ply(sel_path)

    # selection region
    f_weight_local1 = data["face_selection"]
    f_indicator = np.squeeze(f_weight_local1 > 0, 1)

    if invert_sel:
        f_indicator = np.invert(f_indicator)

    return f_indicator
