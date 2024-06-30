import os
import hashlib
from pyspark.sql.functions import col, array, collect_list, udf, sort_array, reverse
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType, ArrayType
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector
from collections import defaultdict
import numpy as np
import pandas as pd
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler
from sklearn.preprocessing import MinMaxScaler
from pyspark.sql import functions as F

class Clustering:
    
    def __init__(self, spark_session):
        self.spark_session = spark_session


    def _calculate_initial_features_(self, df):

        def calculate_duration(times):
            return int(max(times)) - int(min(times))

        duration_udf = udf(calculate_duration, IntegerType())

        grouped_df = df.groupBy("PID") \
                        .agg(collect_list("Time").alias("Time"),
                             collect_list("Target").alias("Target")) \
                        .withColumn('Duration', duration_udf(col('Time'))) \
                        .withColumn('Length', (F.size(col('Target')) / 2).cast("long")) \
                        .select('PID', 'Duration', 'Length')
        grouped_df.cache()                
        grouped_df.show(n=500)
        return grouped_df
    

    def _prepare_all_features_(self, df, features):

        data = df.join(features, features.PID == df.id)
        grouped_df = data.groupBy('component') \
            .agg(F.avg('Length').alias('average_length'),
                F.avg('Duration').alias('average_duration'),
                F.max("Duration").alias('max_duration'),
                F.min("Duration").alias('min_duration'))
            # .withColumn('Duration', length_udf(col('ID')))
        grouped_df.show(n=50)

        grouped_df.toPandas().to_csv('clustering.csv')
        return grouped_df


    # Kneedle algorithm to determine the elbow point
    def _kneedle_algorithm(self, wssse_list, cluster_range):
        # Normalize the WSSSE values
        wssse_scaled = MinMaxScaler().fit_transform(np.array(wssse_list).reshape(-1, 1)).flatten()
        
        # Calculate the differences between consecutive WSSSE values
        differences = np.diff(wssse_scaled)
        
        # Calculate the second derivative (the change of differences)
        second_derivative = np.diff(differences)
        
        # Find the elbow point: point of maximum curvature (minimum second derivative)
        elbow_point_index = np.argmin(second_derivative) + 2  # +2 to adjust for index shift and original range start

        return elbow_point_index


    def make_predictions(self, data=None):

        if not data:
            data = self.spark_session.read.csv("clustering.csv", header=True, inferSchema=True)

        # Select features for clustering
        feature_columns = ['average_length', 'average_duration', 'max_duration', 'min_duration']
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        data = assembler.transform(data)

        print("Assembled data")

        wssse_list = []
        cluster_range = range(2, 11)  # Try clustering for K from 2 to 10
        
        print("Checking cluster ranges")

        for k in cluster_range:
            kmeans = KMeans().setK(k).setSeed(1)
            model = kmeans.fit(data)
            transformed = model.transform(data)
            
            # Calculate WSSSE
            evaluator = ClusteringEvaluator(predictionCol='prediction', featuresCol='features', metricName='silhouette')
            wssse = evaluator.evaluate(transformed)
            wssse_list.append(wssse)

        print("Calculating elbow point")

        elbow_point = self._kneedle_algorithm(wssse_list, cluster_range)
        print(f"Elbow point: {elbow_point}")
        
        kmeans = KMeans(featuresCol="features", k=elbow_point, seed=1)  # Adjust K as needed
        model = kmeans.fit(data)
        print("Fitting the model")
        # Make predictions
        predictions = model.transform(data)

        # Evaluate clustering by computing Silhouette score
        evaluator = ClusteringEvaluator(featuresCol="features")
        silhouette = evaluator.evaluate(predictions)
        print(f"Silhouette with squared euclidean distance = {silhouette}")

        # Show the result
        centers = model.clusterCenters()
        print("Cluster Centers: ")
        for center in centers:
            print(center)

        # predictions.select("features", "prediction").show(truncate=False)
        test = predictions.groupBy('prediction') \
                    .agg(collect_list(col = 'component').alias('processes'))
        test.toPandas().to_csv("clusters.csv", header=True)
        test.show()
        return predictions
    
    def output_observations(self, data=None):
        data = pd.read_csv("clusterss.csv", header=0)




if __name__ == "__main__":
    import spark_config
    import time
    solution_time = time.time() 

    spark_session = spark_config._create_spark_session()
    c = Clustering(spark_session)
    
    # c.make_predictions()

    c.output_observations()
    print(time.time() - solution_time)
