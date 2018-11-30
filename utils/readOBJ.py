import numpy as np

def readOBJ(filepath):
    ## read mesh from .obj file
    ##
    ## Inputs:
    ## filepath: file to obj file
    ##
    ## Outputs:
    ## V: n-by-3 numpy ndarray of vertex positions
    ## F: m-by-3 numpy ndarray of face indices
    V = []
    F = []
    with open(filepath, "rb") as f:
        lines = f.readlines()
    while True:
        for line in lines:
            if line == "":
                break
            elif line.strip().startswith("vn"):
                continue
            elif line.strip().startswith("vt"):
                continue
            elif line.strip().startswith("v"):
                vertices = line.replace("\n", "").split(" ")[1:4]
                vertices = np.delete(vertices,np.argwhere(vertices == np.array([''])).flatten())
                V.append(map(float, vertices))
            elif line.strip().startswith("f"):
                t_index_list = []
                for t in line.replace("\n", "").split(" ")[1:]:
                    t_index = t.split("/")[0]
                    try: 
                        t_index_list.append(int(t_index) - 1)
                    except ValueError:
                        continue
                F.append(t_index_list)
            else:
                continue
        break
    V = np.asarray(V)
    F = np.asarray(F)
    return V, F