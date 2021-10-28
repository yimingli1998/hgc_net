import trimesh
import os


dst = '../meshes/hitdlr_hand'
for root, dirs, files in os.walk(dst):
    for filename in files:
        if filename.endswith('.stl'):
            filepath = os.path.join(root, filename)
            mesh = trimesh.load_mesh(filepath)
            convex_mesh = mesh.convex_hull
            new_filepath = filepath.replace('hitdlr_hand', 'hitdlr_hand_coarse')
            convex_mesh.export(new_filepath)