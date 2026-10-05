import zipfile

import trimesh


def glb_to_obj_zip(glb_path, zip_path):
    scene = trimesh.load(str(glb_path), force="scene")
    with zipfile.ZipFile(zip_path, "w") as z:
        for i, m in enumerate(scene.dump(concatenate=False)):
            obj, tex = trimesh.exchange.obj.export_obj(m, include_texture=True, return_texture=True,
                                                      mtl_name=f"part{i}.mtl")
            z.writestr(f"part{i}.obj", obj)
            for name, data in (tex or {}).items():
                z.writestr(name, data)
