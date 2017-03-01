import nbrun
import time
note = "reportModelResult"

now=time.strftime('-day%Y-%m-%d-time%H-%M')

nb_kwargs = {'dataFolder': "data/MFPT32", 'alternativeArc' : False, 'timeNow' : now}


dname = nb_kwargs['dataFolder'].split("/")[1]
alt = str(nb_kwargs['alternativeArc'])
print "About to execute ",dname," ",alt
nbrun.run_notebook(note,timeout=10000000,nb_suffix='-out_%s_%s--'+str(now) % (dname,alt), nb_kwargs=nb_kwargs,execute_kwargs={"kernel_name": 'python2' })
