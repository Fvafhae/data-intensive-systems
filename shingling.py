import os
import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import functions as f
from pyspark.sql import *
from pyspark.sql.functions import *
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, LongType
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector

os.environ['JAVA_HOME'] = r"C:\Program Files\Java\jdk-22"
os.environ['PYSPARK_PYTHON'] = r"C:\Users\milot\AppData\Local\Programs\Python\Python311\python.exe"


conf = SparkConf()
conf.setAppName("Practical")
conf.setMaster("local[*]")
conf.set("spark.driver.memory", "2G")
conf.set("spark.driver.maxResultSize", "2g")
conf.set("spark.executor.memory", "1G")
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()

import hashlib
##### What this code does #####

# 1. Create DataFrame for our LOG.
# 2. Filter out the requests
# 3. Shingle and hash this information per ProcessID
# 4. for each ProcessID, returns a dense vector of the hash value(s)
bucket_size = 10**8

import generator
data= generator.LOG

schema = StructType([StructField("FromServer", StringType(), True), StructField("ToServer", StringType(), True), StructField("time", LongType(), True), StructField("action", StringType(), True), StructField("ProcessId", LongType(), True)])
df = spark.createDataFrame(data, schema=schema)
df.show(truncate=False)

grouped_df = df.filter(col("action") == "Request").withColumn("ServerSequence", array(col("FromServer"), col("ToServer"))).groupBy("ProcessId").agg(collect_list("ServerSequence")).alias("ServerSequence")
                                                                                                    #we only care about requests (requests always results in response)
                                                                                                    #we create a column "ServerSequence", which is an array [ToServer, FromServer], for each request action
                                                                                                    #We aggregate into a list of arrays, per processID
def hash_shingle(shingle):
    shingle_str = ''.join(map(str, shingle))                                                         # Convert k-shingle to string. Example (k=2): [[0,1], [1,3]] -> "[0,1][1,3]"
    return int(hashlib.md5(shingle_str.encode('utf-8')).hexdigest(), 16) % (bucket_size)                 # hash this string, MOD 10**8 for distribution in one of 10**8 buckets

def generate_hashed_shingles(server_list, k):
    if len(server_list) < k:                                                                         #if processlenght<K just hash wathever process does exist
        return [hash_shingle(server_list)]
    shingles = [server_list[i:i+k] for i in range(len(server_list) - k + 1)]                         #else make list of all k-sized shingles.
    hashed_shingles = [hash_shingle(shingle) for shingle in shingles]                                #hash each shingle
    return hashed_shingles

hashed_shingle_udf = udf(lambda x: generate_hashed_shingles(x, 3), ArrayType(IntegerType()))                        #maps our previously defined function
hashed_df = grouped_df.withColumn("hashed_shingles", hashed_shingle_udf(grouped_df["collect_list(ServerSequence)"]))   #executes the hash shingle function over the filtered dataset.



def array_to_vector(array):
    return Vectors.dense(array)

array_to_vector_udf = udf(array_to_vector, VectorUDT())                             #convert to vector that pyspark recognizes
dense_vectors_df = hashed_df.withColumn(
    "dense_vectors", array_to_vector_udf(sort_array(col("hashed_shingles")))
)

def dense_to_sparse(dense_vector):
    sparse_vector = SparseVector(bucket_size, dense_vector, [1] * len(dense_vector)) #convert to sparse
    return sparse_vector

dense_to_sparse_udf = udf(dense_to_sparse, VectorUDT())
sparse_vectors_df = dense_vectors_df.withColumn(
    "sparse_vectors", dense_to_sparse_udf(col("dense_vectors")))


#sparse_vectors_df.show(truncate=False)

sparse_vectors_df.select("ProcessID", "sparse_vectors").show(truncate=False
                      )