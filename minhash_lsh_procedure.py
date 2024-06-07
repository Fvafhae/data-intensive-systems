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

# cache the signature_frame. we're gonna use it a lot.
signature_frame.cache()


# 3. Perform LSH bucketing by using xxHash library and band-ing strategy.

# 3.1. Create the bands from the sparse vecotors.
# There's nothing about slicing sparse vectors in PySpark documentation.
# On the internet the only approach I could find is first creating actual lists from
    # the sparse vectors, then slicing them. So that's what I'll do now.

# We use xxhash for band hashing since it is one of the fastest:
# xxhash benchmark: https://xxhash.com/

# function to convert sparse vectors to lists:
@udf (returnType=ArrayType(FloatType()))
def denser(sparse_vector):
    return sparse_vector.toArray().tolist()

signature_frame = signature_frame.withColumn("dense_shinglings", denser(df.shinglings))
signature_frame.show(truncate=1000)


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
for i in range(CONFIG["BandCount"]):
    signature_frame = signature_frame.withColumn(f"band_hash_{i}", signature_frame["band_hashes"][i])
signature_frame = signature_frame.drop("band_hashes")

signature_frame.show(truncate=1000)