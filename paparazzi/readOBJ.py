import numpy as np

def readOBJ(filepath, returnColor = False):
    V = []
    F = []
    VC = []
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
                lineLength = len(line.replace("\n", "").split(" "))
                if lineLength == 7: # has vertex color and vertices
                    vertices = line.replace("\n", "").split(" ")[1:4]
                    vertices = np.delete(vertices,np.argwhere(vertices == np.array([''])).flatten())
                    V.append(map(float, vertices))
                    vertexColor = line.replace("\n", "").split(" ")[4:]
                    vertexColor = np.delete(vertexColor,np.argwhere(vertexColor == np.array([''])).flatten())
                    VC.append(map(float, vertexColor))
                elif lineLength == 4:
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
    VC = np.asarray(VC)
    if returnColor is True:
        return V, F, VC
    else:
        return V, F