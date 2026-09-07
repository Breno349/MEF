import numpy as np

def read_mesh(path):
    inCoords = False
    inElements = False
    inGrups = False

    count = 0
    max_count = 0

    elements = []
    coords = []
    groups = {}
    gp = []

    with open(path, "r") as file:
        for line in file:
            if count > 0:
                i = max_count - count
                if inElements:
                    divid = line.split(" ")
                    nos = (int(divid[1]),int(divid[2]),int(divid[3]))
                    elements.append(nos)
                elif inCoords:
                    divid = line.split(" ")
                    xy = (float(divid[0]),float(divid[1]))
                    coords.append(xy)
                elif inGrups:
                    if max_count == 0:
                        count = int(line[14:])
                        max_count = count
                        gp.clear()
                        continue
                    divid = line.split(" ")
                    lin = (int(divid[1]),int(divid[2]))
                    groups[tag].append(lin)
                count -= 1
                if count <= 0:
                    inElements = False
                    inCoords = inElements
                    inGrups = inCoords
                continue
            if line.startswith('NELEM='):
                count = int(line[7:])
                max_count = count
                inElements = True
            if line.startswith('NPOIN='):
                count = int(line[7:])
                max_count = count
                inCoords = True
            if line.startswith('MARKER_TAG='):
                tag = line[12:-1]
                groups[tag] = []
                inGrups = True
                count = 1
                max_count = 0
    return (np.array(elements),np.array(coords),np.array(groups).item())