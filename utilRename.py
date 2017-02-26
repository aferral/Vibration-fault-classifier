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

    f1b = "/home/andres/Desktop/Untitled Folder/MFPT Data 96x96 ScalogramsOr/1 - Three Baseline Conditions"
    f2b = "/home/andres/Desktop/Untitled Folder/MFPT Data 96x96 ScalogramsOr/2 - Three Outer Race Fault Conditions"
    f3b = "/home/andres/Desktop/Untitled Folder/MFPT Data 96x96 ScalogramsOr/3 - Seven More Outer Race Fault Conditions"
    f4b = "/home/andres/Desktop/Untitled Folder/MFPT Data 96x96 ScalogramsOr/4 - Seven Inner Race Fault Conditions"

    f1 = [os.path.join(f1b,elem) for elem in sorted(os.listdir(f1b))] #GET ALL SUB FOLDER
    f2 = [os.path.join(f2b,elem) for elem in sorted(os.listdir(f2b))] #GET ALL SUB FOLDER
    f3 = [os.path.join(f3b,elem) for elem in sorted(os.listdir(f3b))] #GET ALL SUB FOLDER
    f4 = [os.path.join(f4b,elem) for elem in sorted(os.listdir(f4b))] #GET ALL SUB FOLDER

    folderL = f1 + f2 + f3 + f4
    labelL = [1,1,1,
              0,0,0,
              0,0,0,0,0,0,0,
              2,2,2,2,2,2,2]
    lN = {0 : "Outer", 1 : "Baseline",2 : "Inner"}

    #OutFolderName, listOfFolderToLabel, CorrespondingLabels, Optional Label names
    renameFolderList('MFPT96', folderL, labelL, labelNames=lN)

