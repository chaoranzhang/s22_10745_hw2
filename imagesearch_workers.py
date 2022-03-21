
# source activate tensorflow_p27

#run this code with "celery -A imagesearch_workers worker --loglevel=info" on the worker machines

#Then this machine will become a worker, and will be able to run the app task, 
#i.e., the imagesearch_tasks or upload_data functions, whenever the server requests it.

import celery
import numpy as np
from copy import deepcopy
import json
import pyflann
from pyflann import * # I use pyflann data structure. You any that you like
import time
from scipy.spatial import distance

# change this to your load balancers!!!

app = celery.Celery('imagesearch_workers',
                        broker='amqp://myguest:myguestpwd@10745-hw1-78ed39ec05bd3e11.elb.us-east-2.amazonaws.com',
                        backend='amqp://myguest:myguestpwd@10745-hw1-78ed39ec05bd3e11.elb.us-east-2.amazonaws.com')

#Global variables
#Use any global variables that you will need

flann_kdtree = FLANN()
flann_linear = FLANN()
features_train=[]
params_linear =[]
params_kdtree =[]
mydata_loaded =[]
 
@app.task
def upload_data(**kwargs):
    print("in upload_data")
    global upload_nums
    global params_kdtree
    global params_linear
    global features_train
    global flann_linear
    global flann_kdtree
    global mydata_loaded
    print("Start to receive data")
    json_dump=kwargs['json_dump']
    json_load = json.loads(json_dump)
    mydata_loaded = np.asarray(json_load["mydata"])
    print(len(mydata_loaded))
    print(mydata_loaded[0]['path'])
    
    print('*** data uploaded ***')
    
   
    dim=len(mydata_loaded[0]['features'])
    features_train=np.zeros([len(mydata_loaded),dim])
    for iter in range(len(mydata_loaded)):
        features_train[iter,:]=mydata_loaded[iter]['features']

    # ****************************************
    #    *** you will need to complete this part ***
    # Create a near neighbour data structure here from the feature_train feature vectors!
    # ********************************************
    
    pyflann.set_distance_type("euclidean")
    
    params_kdtree = flann_kdtree.build_index(features_train, algorithm = "kdtree", trees = 4)
    params_linear = flann_linear.build_index(features_train, algorithm = "linear")
    
    print("dataset built")
    return

@app.task
def imagesearch_tasks(**kwargs):
    global upload_nums
    global params_kdtree
    global params_linear
    global features_train
    global flann_linear
    global flann_kdtree
    global mydata_loaded
    
    print("start query")
    
    num_results=5
    
    json_dump=kwargs['json_dump']
    
    print("1")
    json_load = json.loads(json_dump)
    print("1")

    query_feature = np.asarray(json_load["query_feature"])
    print("1")
    
    results=[]
    print("1")
    
    # ****************************************
    #    *** you will need to complete this part ***
    # Find k nearest neighbours that are closest to your query vector
    # ********************************************
    
    t1 = time.time()
    linear_result, dists = flann_linear.nn_index(query_feature, 5, checks = params_linear['checks'])
    t2 = time.time()
    linear_time = t2 - t1
#     results.append([linear_result, linear_time])
    
    t1 = time.time()
    kdtree_result, dists = flann_kdtree.nn_index(query_feature, 5, checks = params_kdtree["checks"])
    t2 = time.time()
    kdtree_time = t2 - t1
    
#     results.append([kdtree_result, kdtree_time])
    
    linear_paths = [mydata_loaded[idx]['path'] for idx in linear_result[0]]
    kdtree_paths = [mydata_loaded[idx]['path'] for idx in kdtree_result[0]]
    
    results.append([linear_paths, linear_time])
    results.append([kdtree_paths, kdtree_time])
    print("end query")
    
    # Naive
    t1 = time.time()
    min5 = 999999999999999999999999999999
    min5_res = []
    for feature in mydata_loaded:
        feature = feature["features"]
        dis = dist(feature, query_feature, 0)
        if len(min5_res) < 5:
            min5_res.append(dis)
        else:
            min5 = max(min5_res)
            if dis < min5:
                min5.sort()
                k = min5.pop(-1)
                min5.append(dis)
    t2 = time.time()
    results.append(t2-t1)
  
    return json.dumps({'results': results},cls=NumpyEncoder)

# Euclidean Distance Caculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)
        
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.int64): 
            return int(obj)
        return json.JSONEncoder.default(self, obj)

