import sys
import os
SPARK_HOME = "/usr/lib/spark" 
os.environ["SPARK_HOME"] = SPARK_HOME 
os.environ["SPARK_LOCAL_IP"] = "127.0.0.1" 
sys.path.append( SPARK_HOME + "/python") 
from pyspark.mllib.regression import LabeledPoint
import numpy as np
import math
from pyspark import SparkConf, SparkContext

def gdTrain(data):
    iter = 1000
    a = 0.0005
    D = 4
    w = np.ones(D)
    for i in range(iter):
        gradient = data.map(lambda p: ((   1/ (1+ math.exp( -p.label * np.dot(w, p.features))) - 1 ) * p.label * p.features )).reduce(lambda x,y:x+y)
        w -= a*gradient
    print("Final w: ",w)
    return w
    
def prediction(model, feature):
    print(round(1/( 1+math.exp(-np.dot(model,feature)) )))
    return round(1/( 1+math.exp(-np.dot(model,feature)) ))


def getSparkContext():
    """
    Gets the Spark Context
    """
    conf = (SparkConf()
        .setMaster("local") 
        .setAppName("Logistic Regression") 
        .set("spark.executor.memory", "1g")) 
    sc = SparkContext(conf = conf) 
    return sc

def mapper(line):
    """
    Mapper that converts an input line to a feature vector
    """    
    feats = line.strip().split(",") 
    label = feats[len(feats) - 1] 
    feats = feats[: len(feats) - 1]
    features = [ float(feature) for feature in feats ]
    return LabeledPoint(float(label), features)

sc = getSparkContext()

# Load and parse the data
data = sc.textFile("/hw4/data")
parsedData = data.map(mapper)

# Train model
model = gdTrain(parsedData)

# prediction
labelsAndPreds = parsedData.map(lambda point: (int(point.label), prediction(model, point.features)))

# error rate
trainErr = labelsAndPreds.filter(lambda (v, p): v != p).count() / float(parsedData.count())
print("Training Error = " + str(trainErr))