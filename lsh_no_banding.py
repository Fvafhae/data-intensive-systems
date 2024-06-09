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
    "vector_count": 500,
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
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
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
# TODO:
# This caching is good, we use signature_frame twice while approx, no need to calculate it twice.
# But also, this not the bottle neck, if we need memory, we might choose to remove this matrix from memory by not caching.
signature_frame = model.transform(df).cache()


st = time.time()
similarity_matrix = model.approxSimilarityJoin(signature_frame, signature_frame, 1.0, distCol="JaccardDistance")\
    .select(col("datasetA.id").alias("idA"),
            col("datasetB.id").alias("idB"),
            col("JaccardDistance"))
similarity_matrix = similarity_matrix.filter(similarity_matrix.idA != similarity_matrix.idB).cache()
similarity_matrix.show()

et = time.time()
elapsed_time = et - st
print('Execution normal:', elapsed_time, 'seconds')