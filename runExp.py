import nbrun
note = "reportModelResult"


nb_kwargs = {'dataFolder': "data/MFPT32", 'alternativeArc' : False}


dname = nb_kwargs['dataFolder'].split("/")[1]
alt = str(nb_kwargs['alternativeArc'])
print "About to execute ",dname," ",alt
nbrun.run_notebook(note,timeout=10000000,nb_suffix='-out_%s_%s' % (dname,alt), nb_kwargs=nb_kwargs,execute_kwargs={"kernel_name": 'python2' })
