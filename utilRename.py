import os
import shutil


"""
Now this script will label image and move it to the destination folder.
destFolder = Where to put the label image (it doesnt copy the image it will move it so the folder will be empty after)
Leave empty (set to "") if you dont want to move the files

folder = Where to look for images (it will label all files in this folder)
labelToAdd = What label to add to the files in folder
dataset = A reference to see from what folder the images were originally.  (example baseline 1 have dataset 1 )

labels.


Now renameFolderList has a more simple function to label the entire list of folder and to save a description

see the example below


"""

validT = ['jpg']

def simpleRename(folder,labelToAdd,dataset,destFolder,copy=True):
    for fn in os.listdir(folder):
        name=fn.split(".")[0]
        tpe = fn.split(".")[1]
        if tpe in validT:
            oldPath = os.path.join(folder, fn)

            newName = name+"d"+str(dataset)+"_"+str(labelToAdd)+'.'+tpe
            pathNewFile = os.path.join(folder,newName)

            if copy:
                shutil.copyfile(oldPath, os.path.join(destFolder, newName))
            else: #ELSE MOVE THE FILE
                os.rename(oldPath, pathNewFile)
                shutil.move(pathNewFile, os.path.join(destFolder, newName) )

def renameFolderList(nameOut,folderList,labelList,labelNames=None):
    assert(len(folderList) == len(labelList))

    outString = ""


    #Create outFolder if it doesnt exist
    if not os.path.exists(nameOut):
        os.makedirs(nameOut)

    for i in range(len(folderList)):
        f = folderList[i]
        lab = str(labelList[i])

        dataset = i

        simpleRename(f, lab, dataset, destFolder=nameOut)

        if labelNames is None:
            outString += (f+" label: "+lab+" dataset: "+str(i)+"\n")
        else:
            outString += (f + " label: " + labelNames[labelList[i]]+" -- " + lab +" dataset: "+str(i)+"\n")

    with open('description '+nameOut+".txt",'w+') as f:
        f.write(outString)

    print "The resulting folder is in ",os.getcwd()
    print "The description is in ",os.getcwd()



if __name__ == "__main__":

    folderL = ["/home/andres/Desktop/da/Fan End/12k FE Be Fault Data - 0",
               "/home/andres/Desktop/da/Fan End/12k FE Be Fault Data - 1",
               "/home/andres/Desktop/da/Fan End/12k FE Be Fault Data - 2",
               "/home/andres/Desktop/da/Fan End/12k FE Be Fault Data - 3",
               "/home/andres/Desktop/da/Fan End/FE Baseline - 0",
               "/home/andres/Desktop/da/Fan End/FE Baseline - 1",
               "/home/andres/Desktop/da/Fan End/FE Baseline - 2",
               "/home/andres/Desktop/da/Fan End/FE Baseline - 3"]

    labelL = [0,0,0,0,1,1,1,1]
    lN = {0 : "Fault", 1 : "Baseline"}

    # OutFolderName, listOfFolderToLabel, CorrespondingLabels, Optional Label names
    renameFolderList('FanEnd', folderL, labelL, labelNames=lN)

