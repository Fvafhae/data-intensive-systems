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
        "JaccardDistThreshold": 0.3,
        "vector_count": 500,
        "vector_length": 20
    }

    def __init__(self):
        self.spark, self.sc = self._create_spark_session()
        self.vector_list = None
        self.df = None
        self.signature_frame = None
        self.similarity_matrix = None
        self.final_similarity_groups = None

    def _create_spark_session(self):
        conf = SparkConf()
        conf.setAppName("minhash")
        conf.setMaster(f"local[{self.CONFIG['CoreCount']}]")
        conf.set("spark.driver.memory", "1G")
        conf.set("spark.driver.maxResultSize", "1g")
        conf.set("spark.executor.memory", "8G")
        conf.set("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12")
        sc = SparkContext(conf=conf)
        spark = SparkSession.builder.getOrCreate()
        spark.sparkContext.setCheckpointDir('spark-warehouse/checkpoints')
        return spark, sc

    def vector_creator(self, vector_count=None, vector_length=None):
        if vector_count is None:
            vector_count = self.CONFIG["vector_count"]
        if vector_length is None:
            vector_length = self.CONFIG["vector_length"]
        
        vec_list = []
        one_count = random.randint(5, int(vector_length / 2))

        for i in range(vector_count):
            one_list = []
            for k in range(one_count):
                vec_index = random.randint(0, vector_length - 1)
                if vec_index not in one_list:
                    one_list.append(vec_index)
                else:
                    k -= 1
            vec_list.append((i, Vectors.sparse(vector_length, sorted(one_list), [1.0] * len(one_list)),))

        return vec_list

    def create_vectors(self):
        self.vector_list = self.vector_creator()
        self.df = self.spark.createDataFrame(self.vector_list, ["id", "shinglings"])
    
    def create_signatures(self):
        mh = MinHashLSH(inputCol="shinglings", outputCol="signatures", numHashTables=self.CONFIG["MinHashSignatureSize"], seed=0)
        model = mh.fit(self.df)
        self.signature_frame = model.transform(self.df).cache()

    def compute_similarity(self):
        st = time.time()
        mh = MinHashLSH(inputCol="shinglings", outputCol="signatures", numHashTables=self.CONFIG["MinHashSignatureSize"], seed=0)
        model = mh.fit(self.df)
        self.similarity_matrix = model.approxSimilarityJoin(self.signature_frame, self.signature_frame, self.CONFIG["JaccardDistThreshold"], distCol="JaccardDistance")\
            .select(col("datasetA.id").alias("idA"),
                    col("datasetB.id").alias("idB"),
                    col("JaccardDistance"))
        self.similarity_matrix = self.similarity_matrix.filter(self.similarity_matrix.idA != self.similarity_matrix.idB)\
            .selectExpr("idA as src", "idB as dst", "JaccardDistance").cache()
        et = time.time()
        elapsed_time = et - st
        print('Execution no banding:', elapsed_time, 'seconds')

    def create_similarity_groups(self):
        vertices = self.similarity_matrix.selectExpr("src as id").distinct()
        g = GraphFrame(vertices, self.similarity_matrix)
        result = g.connectedComponents()
        nonsimilars = self.signature_frame.selectExpr('id').subtract(result.selectExpr('id'))
        nonsimilars = nonsimilars.withColumn("component", nonsimilars.id)
        self.final_similarity_groups = result.union(nonsimilars).cache()
        self.final_similarity_groups.show(n=500, truncate=False)

    def run(self):
        self.create_vectors()
        self.create_signatures()
        self.compute_similarity()
        self.create_similarity_groups()


if __name__ == "__main__":
    processor = MinHashLSHProcessor()
    processor.run()
