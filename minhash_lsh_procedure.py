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
    "BandCount": 3
}

### Stop spark if you have created spark session before
#spark.stop()
# create the session
conf = SparkConf()
conf.setAppName("minhash")
conf.setMaster(f"local[{CONFIG['CoreCount']}]")
conf.set("spark.driver.memory", "2G")
conf.set("spark.driver.maxResultSize", "2g")
conf.set("spark.executor.memory", "1G")
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
spark

# 1. Create random sparse vectors (the shingling matrix).
# First we need a function to create random sparse vectors

def vector_creator(vector_count = 100, vector_length = 20):
    vec_list = []

    # at most half of the vector can be full, at least 1 1.:
    one_count = random.randint(1, int(vector_length / 2))
    
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
df.show(truncate=1000)

# TODO: Do I need to create a key vector?
# TODO: Maybe we should use approxSimilarityJoin only, not neighbors


# 2. Perform only the signature creation part of minhashing.

# Minhash model parameters set.
mh = MinHashLSH(inputCol = "shinglings", outputCol="signatures", numHashTables=CONFIG["MinHashSignatureSize"])
# Fit minhash model
model = mh.fit(df)
# Get model results on the data frame: get the signatures:
signature_frame = model.transform(df)
signature_frame.show(truncate=1000)


# 3. Perform LSH bucketing by using xxHash library and band-ing strategy.

# 3.1. Create the bands from the sparse vecotors.
# There's nothing about slicing sparse vectors in PySpark documentation.
# On the internet the only approach I could find is first creating actual lists from
    # the sparse vectors, then slicing them. So that's what I'll do now.

# function to convert sparse vectors to lists:
@udf (returnType=ArrayType(FloatType()))
def denser(sparse_vector):
    return sparse_vector.toArray().tolist()

df = df.withColumn("dense_shinglings", denser(df.shinglings))
df.show(truncate=1000)


# function to split a list into bands:
@udf(returnType=ArrayType(ArrayType(FloatType())))
def band_maker(whole_list):
    band_count = CONFIG["BandCount"]
    bands = []
    hash_values = []
    band_size = m.ceil(len(whole_list) / band_count)
    
    if len(whole_list) > band_count:
        for i in range(band_count):
            start = i * band_size
            end = min(((i+1)*band_size), len(whole_list))
            bands.append(whole_list[start:end])

    return bands

df = df.withColumn("bands", band_maker(df.dense_shinglings)).drop(df.dense_shinglings)
df.show(truncate=1000)


a = [1]
print(band_maker(a))