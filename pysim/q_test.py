import trimesh
import pymesh
import os
from trimesh.exchange import obj

# arrow_path = "arrow_man.obj"
arrow_path = 'meshes/rigidcloth/video/arrow_man.obj'
save_path = "meshes/rigidcloth/video/arrow_man_test.obj"


f_obj = open(arrow_path, "r")
print(os.path.exists(arrow_path))
# mesh = trimesh.load(arrow_path)
mesh = trimesh.exchange.obj.load_obj(f_obj)

trimesh.exchange.obj.export_obj(mesh, save_path)

# mesh2 = mesh.copy()
# mesh2.apply_scale(1.1)

# mesh2.export(file_obj=save_path)
# print(mesh2)

# mesh2 = mesh.copy()
# mesh2.export(save_path)
# mesh = pymesh.load_mesh(arrow_path);
# mesh.translate_x(1)
# mesh.save_obj(save_path);