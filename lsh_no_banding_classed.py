from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import *
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col
import random
import time
from graphframes import GraphFrame


class MinHashLSHProcessor:
    CONFIG = {
        "CoreCount": 8,
        "MinHashSignatureSize": 20,
        "JaccardDistThreshold": 1.0,
        "vector_count": 500,
        "vector_length": 20
    }

    def __init__(self, spark_session, sparse_vector_df):
        self.spark = spark_session
        self.vector_list = None
        self.df = sparse_vector_df
        self.signature_frame = None
        self.similarity_matrix = None
        self.final_similarity_groups = None

    
    def create_signatures(self):
        mh = MinHashLSH(inputCol="sparse_vectors", outputCol="signatures", numHashTables=self.CONFIG["MinHashSignatureSize"], seed=0)
        model = mh.fit(self.df)
        self.signature_frame = model.transform(self.df).cache()

    def compute_similarity(self):
        st = time.time()
        mh = MinHashLSH(inputCol="sparse_vectors", outputCol="signatures", numHashTables=self.CONFIG["MinHashSignatureSize"], seed=0)
        # print("hey")
        # self.signature_frame.select("PID").show(truncate=False)
        model = mh.fit(self.df)
        self.similarity_matrix = model.approxSimilarityJoin(self.signature_frame, self.signature_frame, self.CONFIG["JaccardDistThreshold"], distCol="JaccardDistance")\
            .select(col("datasetA.PID").alias("idA"),
                    col("datasetB.PID").alias("idB"),
                    col("JaccardDistance"))
        self.similarity_matrix = self.similarity_matrix.filter(self.similarity_matrix.idA != self.similarity_matrix.idB)\
            .selectExpr("idA as src", "idB as dst", "JaccardDistance").cache()
        self.similarity_matrix.show(n=1000, truncate=False)
        et = time.time()
        elapsed_time = et - st
        print('Execution no banding:', elapsed_time, 'seconds')

    def create_similarity_groups(self):
        vertices = self.similarity_matrix.selectExpr("src as id").distinct()
        g = GraphFrame(vertices, self.similarity_matrix)
        result = g.connectedComponents()
        nonsimilars = self.signature_frame.selectExpr('PID').subtract(result.selectExpr('id'))
        nonsimilars = nonsimilars.withColumn("component", nonsimilars.PID)
        self.final_similarity_groups = result.union(nonsimilars).cache()
        self.final_similarity_groups.show(n=500, truncate=False)

    def run(self):
        self.create_signatures()
        self.compute_similarity()
        self.create_similarity_groups()


if __name__ == "__main__":

    # Instantiate classes with the shared SparkSession
    # processor = MinHashLSHProcessor(spark)

    # Run the processor
    # processor.run()
    pass
