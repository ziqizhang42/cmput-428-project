import sys
import open3d as o3d

if len(sys.argv) != 2:
    print("Usage: python view_ply.py <path_to_ply>")
    sys.exit(1)

mesh = o3d.io.read_triangle_mesh(sys.argv[1])
if mesh.is_empty():
    pcd = o3d.io.read_point_cloud(sys.argv[1])
    o3d.visualization.draw_geometries([pcd])
else:
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])
