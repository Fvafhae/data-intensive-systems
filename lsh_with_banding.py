import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import *
import pyspark.sql.functions as f
import math as m
from pyspark.sql.types import ArrayType, FloatType, StructType, StructField, IntegerType
import pandas as pd
from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors, DenseVector
from pyspark.sql.functions import col
import random
from pyspark.sql.functions import udf
import numpy as np
import xxhash
import time
from graphframes import GraphFrame # Graphframes makes Spark GraphX available in Python.
from functools import reduce  # For Python 3.x
from pyspark.sql import DataFrame

# TODO: See where you can drop which columns.
# TODO: We implemented band bucketing but see speed and accuracy with/without band bucketing


##### What this code does #####

# 1. Create random sparse vectors (the shingling matrix).
# 2. Perform only the signature creation part of minhashing.
# 3. Perform LSH bucketing by using xxHash library and band-ing strategy.
# 4. Calculate similarity scores for vectors that happened to be in the 
    # same bucket for any band.

# Config settings:
CONFIG = {
    "CoreCount": 8,
    "MinHashSignatureSize": 20,
    "BandCount": 3,
    "JaccadDistThreshold": 1.0,
    "vector_count": 10,
    "vector_length": 20
}

### Stop spark if you have created spark session before
#spark.stop()
# create the session
conf = SparkConf()
conf.setAppName("minhash")
conf.setMaster(f"local[{CONFIG['CoreCount']}]")
conf.set("spark.driver.memory", "1G")
conf.set("spark.driver.maxResultSize", "1g")
conf.set("spark.executor.memory", "8G")
conf.set("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12")
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
spark.sparkContext.setCheckpointDir('spark-warehouse/checkpoints')
spark

# 1. Create random sparse vectors (the shingling matrix).
# First we need a function to create random sparse vectors

def vector_creator(vector_count = CONFIG["vector_count"], vector_length = CONFIG["vector_length"]):
    vec_list = []

    # at most half of the vector can be full, at least 1 1.:
    one_count = random.randint(5, int(vector_length / 2))
    # one_count = 15
    
    # for each vector
    for i in range(vector_count):
        # which entries of the vector are going to be 1?
        one_list = []
        for k in range(one_count):
            vec_index = random.randint(0, vector_length - 1)
            if vec_index not in one_list:
                one_list.append(vec_index)
            else:
                k -= 1
        vec_list.append((i, Vectors.sparse(vector_length, sorted(one_list), [1.0] * len(one_list)),))


    return vec_list


# create the sparse vector list
vector_list = vector_creator()

# turn vector list to a dataframe
df = spark.createDataFrame(vector_list, ["id", "shinglings"])
df.show(truncate=False)

# TODO: Do I need to create a key vector?
# TODO: Maybe we should use approxSimilarityJoin only, not neighbors


# 2. Perform only the signature creation part of minhashing.

# Minhash model parameters set.
mh = MinHashLSH(inputCol = "shinglings", outputCol="signatures", numHashTables=CONFIG["MinHashSignatureSize"], seed=0)
# Fit minhash model
model = mh.fit(df)
# Get model results on the data frame: get the signatures and cache, we'll use this frame a lot:
signature_frame = model.transform(df)



# 3. Perform LSH bucketing by using xxHash library and band-ing strategy.

# 3.1. Create the bands from the sparse vecotors.
# There's nothing about slicing sparse vectors in PySpark documentation.
# On the internet the only approach I could find is first creating actual lists from
    # the sparse vectors, then slicing them. So that's what I'll do now.

# We use xxhash for band hashing since it is one of the fastest:
# xxhash benchmark: https://xxhash.com/

# TODO: Its not clear if minhashLSH uses band bucketing!!!
# TODO: We must try with and without it.
# This page says it does not use band bucketing: https://stackoverflow.com/questions/65259348/is-the-number-of-rows-always-1-in-each-band-in-the-spark-implementation-of-minha
# But GPT says it does and numHashTables is actually the band count. This doesnt make sense.
# I strongly believe that the hash values returned by MinHashLSH function are the signatures.
# One way to know is to try both and compare the times.
# This link is Spark team's implementation of LSH. I dont see any band bucketing: 
# https://github.com/apache/spark/blob/master/mllib/src/main/scala/org/apache/spark/ml/feature/MinHashLSH.scala

# function to convert sparse vectors to lists:
@udf (returnType=ArrayType(FloatType()))
def denser(sparse_vector):
    return sparse_vector.toArray().tolist()

signature_frame = signature_frame.withColumn("dense_shinglings", denser(df.shinglings))
# signature_frame.show(truncate=1000)


# TODO: Return a single list, anly hashes, not the tuple here.
# function to split a list into bands:
@udf(returnType=StructType([
    StructField("bands", ArrayType(ArrayType(FloatType()))),
    StructField("band_hashes", ArrayType(IntegerType()))
]))
def band_maker(whole_list):
    band_count = CONFIG["BandCount"]
    bands = []
    band_hashes = []
    band_size = m.ceil(len(whole_list) / band_count)

    # use a seed with the hash function so it does determisitic hashing.
    
    if len(whole_list) > band_count:

        for i in range(band_count):
            start = i * band_size
            end = min(((i+1)*band_size), len(whole_list))
            band = whole_list[start:end]
            bands.append(band)
            hash_func = xxhash.xxh32(seed = 0)
            hash_func.update(str(band).encode('utf-8'))
            band_hashes.append(hash_func.intdigest())

    return (bands, band_hashes)

# Create band, band_hash dicts.
signature_frame = signature_frame.withColumn("bands_hashes", band_maker(signature_frame.dense_shinglings)).drop("dense_shinglings")

# use this line if you want to see the bands itself. I've already checked. Hashing works correctly.
# df = df.withColumn("bands", df["bands_hashes"].getItem("bands"))

# remove band parts, keep only band_hashes.
signature_frame = signature_frame.withColumn("band_hashes", signature_frame["bands_hashes"].getItem("band_hashes")).drop("bands_hashes")

# Now we make each band_hash a different column
band_col_names = []
for i in range(CONFIG["BandCount"]):
    signature_frame = signature_frame.withColumn(f"band_hash_{i}", signature_frame["band_hashes"][i])
    band_col_names.append(f"band_hash_{i}")
signature_frame = signature_frame.drop("band_hashes").cache()

# This is the first action we perform on signature frame (show). All previous commands were transformations.
# They will be executed once we see this show action.
signature_frame.cache()
# signature_frame.show(truncate=False)
print(band_col_names)

st = time.time()


# Now we divide the df into multiple dfs based on buckets
# create the empty grand similarity frame
schema = StructType([
   StructField("idA", IntegerType(), True),
   StructField("idB", IntegerType(), True),
   StructField("JaccardDistance", FloatType(), True)
])

#Creating an empty DataFrame.
grand_similarity = spark.createDataFrame([], schema)

st = time.time()

# for each band field:
for band_col in band_col_names:
    # get distinct values for the column.
    dist_vals = signature_frame.select(band_col).distinct().rdd.flatMap(lambda x: x).collect()
    # print(dist_vals)

    for i, val in enumerate(dist_vals):
        split = signature_frame.filter(signature_frame[band_col] == val).select("id", "shinglings") #.cache()
        # split.show(truncate=False)
        # print(split.count())

        # now for each split we need to make Jaccard comparison
        # TODO: Keep only id and signature columns at this point
        # TODO: Initialize and cache the output frame to append on it

        split_similars = model.approxSimilarityJoin(split, split, CONFIG["JaccadDistThreshold"], distCol="JaccardDistance")\
                 .select(col("datasetA.id").alias("idA"),
                col("datasetB.id").alias("idB"),
                col("JaccardDistance"))
        # eliminate rows matched with itself
        split_similars = split_similars.filter(split_similars.idA != split_similars.idB)
        # add each buckets similars into the grand similarity frame
        # TODO: Check physical plan to see if grand_similarity is already cached.
        grand_similarity = grand_similarity.union(split_similars)

        # if i == 10:
            # break
    # break

grand_similarity = grand_similarity.selectExpr("idA as src", "idB as dst", "JaccardDistance")
grand_similarity.cache()

et = time.time()
elapsed_time = et - st
print('Execution no banding:', elapsed_time, 'seconds')

# Create similarity groups with graph.

# Create vertices DataFrame
vertices = grand_similarity.selectExpr("src as id").distinct()

# Create GraphFrame
g = GraphFrame(vertices, grand_similarity)

# Find connected components
result = g.connectedComponents()

# Get the processes that have no similars
nonsimilars = signature_frame.selectExpr('id').subtract(result.selectExpr('id'))
nonsimilars = nonsimilars.withColumn("component", nonsimilars.id)

final_similarity_groups = result.union(nonsimilars).cache()
final_similarity_groups.show(n=500, truncate=False)
