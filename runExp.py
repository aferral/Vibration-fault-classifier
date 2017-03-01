import nbrun
import time
import os
note = "reportModelResult"



datasets = ["MFPT32","MFPT96","CWRbearings",'MFPTFFT32','MFPTFFT96']




for data in datasets:
    for i in range(2):
        dFolder = os.path.join('data',data)
        alt = bool(i)

        now = time.strftime('-day%Y-%m-%d-time%H-%M')
        nb_kwargs = {'dataFolder': dFolder, 'alternativeArc': alt, 'timeNow': now}

        dname = nb_kwargs['dataFolder'].split("/")[1]
        alt = str(nb_kwargs['alternativeArc'])
        print "About to execute ",dname," ",alt
        nbrun.run_notebook(note,timeout=10000000,nb_suffix='-out_%s_%s--'+str(now) % (dname,alt), nb_kwargs=nb_kwargs,execute_kwargs={"kernel_name": 'python2' })
