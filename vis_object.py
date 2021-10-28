import trimesh
import glob

files = glob.glob('picked_obj/*.obj')
for f in files:
    print(f)
    obj = trimesh.load(f)
    obj.show()