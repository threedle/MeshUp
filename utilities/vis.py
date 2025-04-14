# Helper functions for visualization
import numpy as np
import imageio
from pathlib import Path


def create_uniform_color(verts):
    """
    Creates a uniform color map on a mesh.

    Args:
        vert (Nx3 ndarray): The vertices of the mesh.

    Returns:
        colors (Nx3 ndarray): The RGB colors per mesh vertex.
    """
    verts = np.double(verts)
    minx = np.min(verts[:, 0])
    miny = np.min(verts[:, 1])
    minz = np.min(verts[:, 2])
    maxx = np.max(verts[:, 0])
    maxy = np.max(verts[:, 1])
    maxz = np.max(verts[:, 2])
    colors = np.stack(
        [((verts[:, 0] - minx) / (maxx - minx)),
         ((verts[:, 1] - miny) / (maxy - miny)),
         ((verts[:, 2] - minz) / (maxz - minz))]
    ).transpose()

    return colors


def make_animation(image_seq_dir_path: Path, fps=30, add_reverse=True, extension='gif'):
    assert image_seq_dir_path.is_dir(), f"No such directory [{image_seq_dir_path}]"
    print(image_seq_dir_path)
    file_paths = sorted(image_seq_dir_path.glob("*.png"))
    images = []
    for file_path in file_paths:
        try:
            images.append(imageio.imread(str(file_path)))
        except:
            images.append(imageio.v2.imread(str(file_path)))

    if add_reverse:
        reverse = images[::-1]
        images.extend(reverse)
    
    target_file_path = image_seq_dir_path.with_suffix(f'.{extension}')
    if extension=='gif':
        imageio.mimsave(target_file_path, images, duration=(1000/fps), loop=0)
    else:
        writer = imageio.get_writer(target_file_path, fps=fps)

        for image in images:
            writer.append_data(image)

        writer.close()

    print(f"Animation saved to [{target_file_path}]")
