from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
spark = SparkSession.builder.appName('ml-classifier').getOrCreate()
import pandas

def openfile(path):
    return spark.read.csv(path, header = True, inferSchema = True)
    
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
        encoder = OneHotEncoderEstimator(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
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
    df=openfile(filename)
    categoricalColumns=checkCategoricalColumns(df)
    numericColumns=checkNumericColumns(df)
    cols=allColumns(df)
    stages,df=indexInputColumns(categoricalColumns,stages,df)
    stages,df = indexOutputColumn(stages,'deposit',df)
    stages,df = vectorAsFeatures(categoricalColumns,numericColumns,stages,df)
    selectedCols,df=pipelane(df,stages,cols)
    train,test = splitDataToTrainAndTest(df)
    rf = RandomForestClassifier(featuresCol = 'features', labelCol = 'label')
    rfModel = rf.fit(train)
    predictions = rfModel.transform(test)
    predictions.select('age', 'job', 'label', 'rawPrediction', 'prediction', 'probability')
    evaluator = binaryClassificationEvaluator(predictions)
    accuracy = evaluator.evaluate(predictions)
    predictions = predictions.toPandas()
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
    
