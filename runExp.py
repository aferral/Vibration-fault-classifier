import nbrun
import time
import os
note = "reportModelResult"

outReport = 'reports'

if not os.path.exists(outReport):
    os.makedirs(outReport)


# datasets = ["MFPT32","MFPT96","CWRbearings",'MFPTFFT32','MFPTFFT96']

# datasets =["MFPT32Scalograms","MFPT32SpectrogramsV2","MFPT96Scalograms","MFPT96SpectrogramsV2",
#            "CW32Scalograms", "CW32Spectrograms", "CW96Scalograms", "CW96Spectrograms",
#            "CWRHHT32", "CWRHHT96", "MFPT_HHT_32", "MFPT_HHT_96"
#            ]
datasets = ['CW96Scalograms']


for data in datasets:
    # for i in range(3):
    i = 2
    dFolder = os.path.join('data',data)
    alt = int(i)

    now = time.strftime('-day%Y-%m-%d-time%H-%M')
    nb_kwargs = {'dataFolder': dFolder, 'alternativeArc': alt, 'timeNow': now, 'epochs' : 20, "valSplit" : 0.1}

    dname = nb_kwargs['dataFolder'].split("/")[1]
    alt2 = str(nb_kwargs['alternativeArc'])
    print "About to execute ",dname," ",alt2
    nbrun.run_notebook(note,out_path=outReport,timeout=10000000,nb_suffix='-out_%s_%s--%s' % (dname,str(alt2),str(now)),nb_kwargs=nb_kwargs,execute_kwargs={"kernel_name": 'python2' })

