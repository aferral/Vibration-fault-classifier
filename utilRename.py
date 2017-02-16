import os
folder = "/home/andres/Desktop/Scalograms/Baseline - 3/"
labelToAdd = 1

for fn in os.listdir(folder):
    name=fn.split(".")[0]
    tpe = fn.split(".")[1]
    os.rename(os.path.join(folder,fn), os.path.join(folder,name+"d3_"+str(labelToAdd)+'.'+tpe))


