import os
import hashlib
from pyspark.sql.functions import col, array, collect_list, udf, sort_array
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType, ArrayType
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector
from collections import defaultdict

# os.environ['JAVA_HOME'] = r"C:\Program Files\Java\jdk-22"
# os.environ['PYSPARK_PYTHON'] = r"C:\Users\milot\AppData\Local\Programs\Python\Python311\python.exe"

class Shingler:
    def __init__(self, spark_session, df):
        self.spark = spark_session
        self.df = df
        self.bucket_count = 10**8

    def process_data(self):
        grouped_df = self.df.filter(col("EventType") == "Request") \
                            .withColumn("ServerSequence", array(col("Caller"), col("Target"))) \
                            .groupBy("PID") \
                            .agg(collect_list("ServerSequence").alias("ServerSequence"))

        def hash_shingle(shingle):
            shingle_str = ''.join(map(str, shingle))
            return int(hashlib.md5(shingle_str.encode('utf-8')).hexdigest(), 16) % (10**8)

        def generate_hashed_shingles(server_list, k):
            if len(server_list) < k:
                return [hash_shingle(server_list)]
            shingles = [server_list[i:i+k] for i in range(len(server_list) - k + 1)]
            hashed_shingles = [hash_shingle(shingle) for shingle in shingles]
            return hashed_shingles

        hashed_shingle_udf = udf(lambda x: generate_hashed_shingles(x, 3), ArrayType(IntegerType()))
        hashed_df = grouped_df.withColumn("hashed_shingles", hashed_shingle_udf(grouped_df["ServerSequence"]))

        array_to_vector_udf = udf(lambda array: Vectors.dense(array), VectorUDT())
        dense_vectors_df = hashed_df.withColumn("dense_vectors", array_to_vector_udf(sort_array(col("hashed_shingles"), asc=True)))
        # dense_vectors_df.show(truncate=False)

        def dense_to_sparse(dense_vector):
            indices = sorted(set(dense_vector))
            values = [1] * len(indices)
            sparse_vector = SparseVector(10**8, indices, values)
            return sparse_vector
        
        """
        def dense_to_sparse(dense_vector):
            print(dense_vector)
            sparse_vector = SparseVector(10**8, dense_vector, [1] * len(dense_vector))
            return sparse_vector
        """
        

        dense_to_sparse_udf = udf(dense_to_sparse, VectorUDT())
        self.sparse_vectors_df = dense_vectors_df.withColumn("sparse_vectors", dense_to_sparse_udf(col("dense_vectors")))

    def show_result(self):
        self.sparse_vectors_df.select("PID", "sparse_vectors").show(truncate=False)

    def stop(self):
        self.spark.stop()
        self.sc.stop()

    def run(self):
        self.process_data()
        self.show_result()

if __name__ == "__main__":
    processor = Shingler()
    processor.process_data()
    # processor.show_result()
    # processor.stop()
