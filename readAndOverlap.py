from scipy import io as sio
import os
import time

dataFolder = "data/CWRfeatures"
outFolder = "data/CWRfeatures stride8"

if not os.path.exists(outFolder):
    os.makedirs(outFolder)

dataHeight = 75
timeStep = 15
stride= 8



#for each subfolder a new label (just read folder)
allFolders = filter(lambda x : os.path.isdir(os.path.join(dataFolder,x)), os.listdir(dataFolder))
now = time.strftime('Day%Y-%m-%d-Time%H-%M')

for label,folder in enumerate(allFolders):
	# Read all the mat files inside

	fileList = os.listdir(os.path.join(dataFolder,folder))
	for f in fileList:
		if f.split('.')[-1] == 'mat': #check that the file is mat
			matpath = os.path.join(dataFolder,folder,f)
			fileName = f.split('.')[-2] #Extract the .mat from the filename
			#The .mat is a dictionary and the fileName is the key for the data
			dataDict = sio.loadmat(matpath)
			matrixData = dataDict[fileName]

			i=0
			lsup=0
			while (lsup+timeStep) < matrixData.shape[1]:
				linf = stride*i
				lsup = stride*i+timeStep
				trozo = matrixData[:,linf:lsup]
				assert(trozo.shape == (dataHeight,timeStep))

				#Ahora guardar archivo
				if not os.path.exists(os.path.join(outFolder,folder)):
					os.makedirs(os.path.join(outFolder,folder))
				outDict = {}

				outNamepre = fileName+"S"+str(i)
				outName = outNamepre+'.mat'
				outDict[outNamepre] = trozo
				sio.savemat(os.path.join(outFolder,folder,outName),outDict)
				i+=1
	#Recordar logear archivo para saber que stride time step use, archvo
	outDescName = 'descriptionStride'+str(now)+'.txt'
	outDescPath = os.path.join(outFolder,outDescName)
	with open(outDescPath,'w') as fileToLog:
		fileToLog.write('DataHeight '+str(dataHeight)+" timeStep "+str(timeStep)+" stride "+str(stride)+" NOW "+str(now))





		
	
