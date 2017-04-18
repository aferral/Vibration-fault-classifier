dataFolder = ""

#for each subfolder a new label (just read folder)
allFolders = filter(lambda x : os.path.isdir(os.path.join(dataFolder,x)), os.listdir(dataFolder))

now = time.strftime('Day%Y-%m-%d-Time%H-%M')
log = "\n\n Dataset label log "+str(dataFolder)+str(now)+" \n"

all = []
allL = []

dataHeight = 75
timeStep = 15
stride= 8

outFolder = ""


for label,folder in enumerate(allFolders):
    # Read all the mat files inside

    fileList = os.listdir(os.path.join(dataFolder,folder))
    for f in fileList:
        if f.split('.')[-1] == 'mat': #check that the file is mat
            matpath = os.path.join(dataFolder,folder,f)
            fileName = f.split('.')[-2] #Extract the .mat from the filename
            matrixData = sio.loadmat(matpath)[fileName] #The .mat is a dictionary and the fileName is the key for the data
		
		while lsup < matrixData.shape[1]:
			linf = stride*i
			lsup = stride*i+timeStep
			trozo = matrixData[:,linf:lsup]
			assert(trozo.shape == (dataHeight,timeStep))

			#Ahora guardar archivo
			sio.savemat(trozo,os.path.join(outFolder,dataFolder,folder,f))

	#Recordar logear archivo para saber que stride time step use, archvo
	with open('descriptionStride'+str(dataFolder)+'.txt','w') as fileToLog:
		fileToLog.write('DataHeight '+str(dataHeight)+" timeStep "+str(timeStep)+" stride "+str(stride)+" out "+str(outFolder))





		
	
