import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import *
import pyspark.sql.functions as f
import math as m
from pyspark.sql.types import FloatType
import pandas as pd

# Config settings:
CONFIG = {
    "CoreCount": 8
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

from pyspark.ml.feature import MinHashLSH
from pyspark.ml.linalg import Vectors
from pyspark.sql.functions import col

dataA = [(0, Vectors.sparse(10, [0, 1, 2, 7], [1.0, 1.0, 1.0, 1.0]),),
         (1, Vectors.sparse(10, [2, 3, 4, 8], [1.0, 1.0, 1.0, 1.0]),),
         (2, Vectors.sparse(10, [0, 2, 4, 9], [1.0, 1.0, 1.0, 1.0]),)]
dfA = spark.createDataFrame(dataA, ["id", "features"])

dfA.show(truncate=100)

dataB = [(3, Vectors.sparse(10, [1, 3, 5, 6], [1.0, 1.0, 1.0, 1.0]),),
         (4, Vectors.sparse(10, [2, 3, 5, 7], [1.0, 1.0, 1.0, 1.0]),),
         (5, Vectors.sparse(10, [1, 2, 4, 8], [1.0, 1.0, 1.0, 1.0]),),
         (6, Vectors.sparse(10, [1, 2, 4, 9], [1.0, 1.0, 1.0, 1.0]),)]
dfB = spark.createDataFrame(dataB, ["id", "features"])

dfB.show(truncate=100)

key = Vectors.sparse(6, [0, 1, 2, 3], [1.0, 1.0, 1.0, 1.0])
print(key)

# numHashTables: how many hash functions we're going to use. 
# As we increase the number of hash functions, number of false negatives decrease,
# Because if 2 sets dont fall into the same bucket for one hash function, they might for another hash function
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=2)
model = mh.fit(dfA)

# Feature Transformation
print("The hashed dataset where hashed values are stored in the column 'hashes':")
model.transform(dfA).show(truncate=1000)
model.transform(dfB).show(truncate=1000)

# Compute the locality sensitive hashes for the input rows, then perform approximate
# similarity join.
# We could avoid computing hashes by passing in the already-transformed dataset, e.g.
# model.approxSimilarityJoin(transformedA, transformedB, 0.6)
# distance = 1 - JaccardSim
print("Approximately joining dfA and dfB on distance smaller than 0.6:")
similars = model.approxSimilarityJoin(dfA, dfB, 1.0, distCol="JaccardDistance")\
    .select(col("datasetA.id").alias("idA"),
            col("datasetB.id").alias("idB"),
            col("JaccardDistance"))
# similars = similars.withColumn("JaccardSimilarity", 1 - similars.JaccardDistance).drop("JaccardDistance")
similars.show(truncate=100)

 
# Compute the locality sensitive hashes for the input rows, then perform approximate nearest
# neighbor search.
# We could avoid computing hashes by passing in the already-transformed dataset, e.g.
# `model.approxNearestNeighbors(transformedA, key, 2)`
# It may return less than 2 rows when not enough approximate near-neighbor candidates are
# found.
print("Approximately searching dfA for 2 nearest neighbors of the key:")
model.approxNearestNeighbors(dfA, key, 2).show()
model.explainParams()



my_df = dfA.union(dfB)
mh = MinHashLSH(inputCol="features", outputCol="hashes", numHashTables=2)
model = mh.fit(my_df)
print("The hashed dataset where hashed values are stored in the column 'hashes':")
model.transform(my_df).show(truncate=1000)

# For a single hash function, we can get different buckets (hashes):
# We can use multiple hash functions if we want to create more buckets with less elements.
# We can apply this to different bands of the vectors. 
# And if any two id happen to be in the same bucket for any of the bands, we make the actual comparison with the whole vector.
"""
+---+--------------------------------+----------------+
| id|                        features|          hashes|
+---+--------------------------------+----------------+
|  0|(10,[0,1,2,7],[1.0,1.0,1.0,1.0])|[[1.05211356E8]]|
|  1|(10,[2,3,4,8],[1.0,1.0,1.0,1.0])|[[9.82158528E8]]|
|  2|(10,[0,2,4,9],[1.0,1.0,1.0,1.0])|[[1.05211356E8]]|
|  3|(10,[1,3,5,6],[1.0,1.0,1.0,1.0])|[[2.59504543E8]]|
|  4|(10,[2,3,5,7],[1.0,1.0,1.0,1.0])|[[2.59504543E8]]|
|  5|(10,[1,2,4,8],[1.0,1.0,1.0,1.0])|[[5.43684942E8]]|
+---+--------------------------------+----------------+

"""

# To make the actual minhashing with the whole vector, we should increase numHashTables
