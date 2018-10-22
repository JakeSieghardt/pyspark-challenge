
from pyspark.sql import SQLContext
from pyspark.sql.types import *

sqlContext = SQLContext(sc)

df_data_1 = sqlContext.read.format('com.databricks.spark.csv').\
options(header='true', inferschema='true').load('realestate.csv')


df_data_1.printSchema()





df_data_1.describe('beds','baths','sq__ft').show()

#Filter rows with probably wrong data (number of baths/beds/sqft=0)
filtered_df=df_data_1.where("baths > 0 and beds > 0 and sq__ft >0 ")
filtered_df.cache()
filtered_df.take(5)



#Apply type
from pyspark.sql.types import DoubleType

filtered_typed_df=filtered_df.withColumn("beds", filtered_df["beds"].\
cast("double")).withColumn("baths", filtered_df["baths"].cast("double")).\
withColumn("sq__ft", filtered_df["sq__ft"].cast("double")).\
withColumn("price", filtered_df["price"].cast("double"))

filtered_typed_df.printSchema()
filtered_typed_df.show(5)



# Get some statistics on the relevant features and label (price)

filtered_typed_df.describe('beds','baths','sq__ft','price').show()





# Select all apartment in EL DORADO HILLS and determine the one with maximum SQFT

el_dorado_aptm_df=filtered_typed_df.filter(filtered_df["city"]=="EL DORADO HILLS")
el_dorado_aptm_df.cache()
el_dorado_aptm_df.show(5)
max_sqft= el_dorado_aptm_df.groupBy('city').max('sq__ft').collect()[0][1]

print (max_sqft) 


el_dorado_aptm_df.filter(el_dorado_aptm_df.sq__ft==max_sqft).collect()





#select and scale the relevant features 
from pyspark.ml.feature import VectorAssembler

assembler1 = VectorAssembler(
    inputCols=['beds','baths','sq__ft'],
    outputCol='features')


filtered_features_df = assembler1.transform(filtered_typed_df)


filtered_features_df.printSchema()
filtered_features_df.show(5)
               



from pyspark.ml.feature import MinMaxScaler
#from pyspark.ml.linalg import Vectors


scaler1 = MinMaxScaler(inputCol="features", outputCol="scaled_features")

# Compute summary statistics and generate MinMaxScalerModel
scaler_model1 = scaler1.fit(filtered_features_df)

# rescale each feature to range [min, max].
scaled_data_df = scaler_model1.transform(filtered_features_df)
#print("Features scaled to range: [%f, %f]" % (scaler1.getMin(), scaler1.getMax()))
scaled_data_df.select("features", "scaled_features").show()

scaled_data_df.printSchema()



training_data_df, testing_data_df = scaled_data_df.randomSplit([.8,.2],seed=1234)
training_data_df.take(5)




testing_data_df.take(5)




from  pyspark.ml.regression import LinearRegression

lr = LinearRegression(featuresCol = 'scaled_features', labelCol = 'price')
#lr = LinearRegression(featuresCol = 'features', labelCol = 'price')

# Fit the model
lr_model = lr.fit(training_data_df)


# Print the coefficients and intercept for linear regression
print("Coefficients: %s" % str(lr_model.coefficients))
print("Intercept: %s" % str(lr_model.intercept))




# Summarize the model over the training set and print out some metrics
training_summary = lr_model.summary
print("numIterations: %d" % training_summary.totalIterations)
print("objectiveHistory: %s" % str(training_summary.objectiveHistory))
training_summary.residuals.show()
print("RMSE: %f" % training_summary.rootMeanSquaredError)
print("r2: %f" % training_summary.r2)



testing_summary = lr_model.evaluate(testing_data_df)
testing_summary.predictions.select('price','baths','beds','sq__ft','prediction').show(10)
print("RMSE: %f" % testing_summary.rootMeanSquaredError)
print("r2: %f" % testing_summary.r2)
