from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
spark = SparkSession.builder.appName('ml-classifier').getOrCreate()
from elasticsearch import Elasticsearch
import os
import sys

ES_HOST = {
        "host" : "localhost", 
        "port" : 9200
    }
es = Elasticsearch(hosts = [ES_HOST])

def openfile(path):
    return spark.read.csv('files/' + path, header = True, inferSchema = True)
    
def checkNumericColumns(df):
    return [t[0] for t in df.dtypes if t[1] == 'int']

def checkCategoricalColumns(df):
    return [item[0] for item in df.dtypes if item[1].startswith('string')] 

def allColumns(df):
    return df.columns

#input
def indexInputColumns(categoricalColumns,stages,df):
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]
    return stages,df

#output
def indexOutputColumn(stages,output,df):
    label_stringIdx = StringIndexer(inputCol = output, outputCol = 'label')
    stages += [label_stringIdx]
    return stages,df

def vectorAsFeatures(categoricalColumns,numericCols,stages,df):
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler]
    return stages,df

def pipelane(df,stages,cols):
    pipeline = Pipeline(stages = stages)
    pipelineModel = pipeline.fit(df)
    df = pipelineModel.transform(df)
    selectedCols = ['label', 'features'] + cols
    return selectedCols,df

def splitDataToTrainAndTest(df):
    return df.randomSplit([0.7, 0.3], seed = 69)

def logisticRegression(train,selectedCols, test):
    selectedCols = selectedCols
    lr = LogisticRegression(featuresCol = 'features', labelCol = 'label', maxIter=10)
    lrModel = lr.fit(train)
    trainingSummary = lrModel.summary
    print('Training set areaUnderROC: ' + str(trainingSummary.areaUnderROC))
    predictions = lrModel.transform(test)
    predictions.select('age','label','rawPrediction','prediction','probability').show(10)
    return lrModel,lr,predictions

def binaryClassificationEvaluator(predictions):
    evaluator = BinaryClassificationEvaluator()
    print("Test Area Under ROC: " + str(evaluator.evaluate(predictions, {evaluator.metricName: "areaUnderROC"})))
    evaluator.getMetricName()
    return evaluator

# Create ParamGrid for Cross Validation
def paramGridBuilder(lr):
    return (ParamGridBuilder()
                 .addGrid(lr.regParam, [0.01, 0.5, 2.0])
                 .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
                 .addGrid(lr.maxIter, [1, 5, 10])
                 .build())

def crossValidator(lr,paramGrid,evaluator,train,test):
    cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)
    cvModel = cv.fit(train)
    predictions = cvModel.transform(test)
    print('Test Area Under ROC', evaluator.evaluate(predictions))

def decisionTreeClassifier(train,test):
    dt = DecisionTreeClassifier(featuresCol = 'features', labelCol = 'label', maxDepth = 3)
    dtModel = dt.fit(train)
    predictions = dtModel.transform(test)
    predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
    return predictions

def randomForestClassifier(train,test):
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)
    predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
    return predictions

def gbtClassifier(train,test):
    gbt = GBTClassifier(maxIter=10)
    gbtModel = gbt.fit(train)
    predictions = gbtModel.transform(test)
    predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
    return predictions


def doRandomForestClassification(filename):
    stages=[]
    es.update(index='spark-jobs', doc_type='job', id=task_id, body={
        'doc': { 
            'current': 30,
            'status': 'Reading file..' 
        }
    })
    df=openfile(filename)
    es.update(index='spark-jobs', doc_type='job', id=task_id, body={
        'doc': { 
            'current': 40,
            'status': 'Mapping..' 
        }
    })
    categoricalColumns=checkCategoricalColumns(df)
    numericColumns=checkNumericColumns(df)
    cols=allColumns(df)
    stages,df=indexInputColumns(categoricalColumns,stages,df)
    stages,df = indexOutputColumn(stages,'deposit',df)
    stages,df = vectorAsFeatures(categoricalColumns,numericColumns,stages,df)
    selectedCols,df=pipelane(df,stages,cols)
    es.update(index='spark-jobs', doc_type='job', id=task_id, body={
        'doc': { 
            'current': 50,
            'status': 'Splitting data to train and test..' 
        }
    })
    train,test = splitDataToTrainAndTest(df)
    es.update(index='spark-jobs', doc_type='job', id=task_id, body={
        'doc': { 
            'current': 60,
            'status': 'Training model..' 
        }
    })
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)
    predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability')
    es.update(index='spark-jobs', doc_type='job', id=task_id, body={
        'doc': { 
            'current': 80,
            'status': 'Calculating accuracy..' 
        }
    })
    evaluator = binaryClassificationEvaluator(predictions)
    accuracy = evaluator.evaluate(predictions)
    return accuracy, predictions, rfModel
    # print("Test Error = %g" % (1.0 - accuracy))

# def testModel(filename, rfModel):
#     stages=[]
#     df=openfile(filename)
#     categoricalColumns=checkCategoricalColumns(df)
#     numericColumns=checkNumericColumns(df)
#     cols=allColumns(df)
#     stages,df=indexInputColumns(categoricalColumns,stages,df)
#     stages,df = indexOutputColumn(stages,'',df)
#     stages,df = vectorAsFeatures(categoricalColumns,numericColumns,stages,df)
#     selectedCols,df=pipelane(df,stages,cols)
#     test = df
#     predictions = rfModel.transform(test)
#     predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability').show(10)
#     evaluator = binaryClassificationEvaluator(predictions)
#     accuracy = evaluator.evaluate(predictions)
#     return accuracy, predictions
    
if __name__ == "__main__":
    task_id = sys.argv[1]
    print(task_id)

    es.update(index='spark-jobs', doc_type='job', id=task_id, body={
        'doc': { 
            'current': 10,
            'status': 'Spark job started..' 
        }
    })

    es.update(index='spark-jobs', doc_type='job', id=task_id, body={
        'doc': { 
            'current': 20,
            'status': 'Classification..' 
        }
    })

    accuracy, predictions, rfModel = doRandomForestClassification("bank.csv")

    es.update(index='spark-jobs', doc_type='job', id=task_id, body={
        'doc': { 
            'current': 95,
            'status': 'Converting to csv..' 
        }
    })

    predictions.toPandas().to_csv(str(task_id) + 'output.csv')

    es.update(index='spark-jobs', doc_type='job', id=task_id, body={
        'doc': { 
            'current': 100,
            'status': 'Spark job finished.',
            'result': str(task_id) + 'output.csv'
        }
    })