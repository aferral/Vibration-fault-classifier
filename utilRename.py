import os
import shutil


"""
Now this script will label image and move it to the destination folder.
destFolder = Where to put the label image (it doesnt copy the image it will move it so the folder will be empty after)
Leave empty (set to "") if you dont want to move the files

folder = Where to look for images (it will label all files in this folder)
labelToAdd = What label to add to the files in folder
dataset = A reference to see from what folder the images were originally.  (example baseline 1 have dataset 1 )

labels


"""


destFolder = "C:/Users/andres/Desktop/MFPT Fault Data Sets - Copy/LoadFaultData"
folder = "C:/Users/andres/Desktop/MFPT Fault Data Sets - Copy/" \
         "4 - Seven Inner Race Fault Conditions\InnerRaceFault_vload_7"
labelToAdd = 19
dataset = 0


validT = ['jpg']

for fn in os.listdir(folder):
    name=fn.split(".")[0]
    tpe = fn.split(".")[1]
    if tpe in validT:
        newName = name+"d"+str(dataset)+"_"+str(labelToAdd)+'.'+tpe
        pathNewFile = os.path.join(folder,newName)
        os.rename(os.path.join(folder,fn),  pathNewFile)

        if destFolder != "":
            shutil.move(pathNewFile, os.path.join(destFolder, newName) )


