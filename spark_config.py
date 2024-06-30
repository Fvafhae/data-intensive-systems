from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

def _create_spark_session():
        conf = SparkConf().setAppName("minhash").setMaster("local[8]")
        conf.set("spark.executor.instances", "1")
        conf.set("spark.driver.memory", "4G")
        conf.set("spark.driver.maxResultSize", "6G")
        conf.set("spark.executor.cores", "4")
        conf.set("spark.driver.cores", "4")
        conf.set("spark.default.parallelism", "8")
        conf.set("spark.memory.fraction", "0.8")
        conf.set("spark.memory.storageFraction", "0.2")
        conf.set("spark.rdd.compress", "true")
        conf.set("spark.io.compression.codec", "lz4")
        conf.set("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12")

        sc = SparkContext(conf=conf)
        spark = SparkSession(sc)
        # spark.stop()
        # spark = SparkSession.builder.getOrCreate()
        spark.sparkContext.setCheckpointDir('spark-warehouse/checkpoints')
        # spark.sparkContext.setLogLevel("DEBUG")
        return spark