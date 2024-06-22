from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import *
import pyspark.sql.functions as f
import math as m
from pyspark.sql.types import FloatType

class StringSimilarity:
    def __init__(self, core_count=8):
        self.core_count = core_count
        self.spark = self._create_spark_session()
        self.collapsed_data = None
        self.jaro_th = 0.3

    def _create_spark_session(self):
        conf = SparkConf().setAppName("minhash").setMaster("local[8]")
        conf.set("spark.driver.memory", "1G")
        conf.set("spark.driver.maxResultSize", "1g")
        conf.set("spark.executor.memory", "8G")
        conf.set("spark.jars.packages", "graphframes:graphframes:0.8.2-spark3.1-s_2.12")
        sc = SparkContext(conf=conf)
        spark = SparkSession.builder.getOrCreate()
        spark.sparkContext.setCheckpointDir('spark-warehouse/checkpoints')
        return spark

    @staticmethod
    def jaro_winkler_similarity(s1, s2, winkler_factor=0.1, matching_prefix_length=20):
        if s1 == s2:
            return 1.0
        
        s1_len = len(s1)
        s2_len = len(s2)
        
        match_th = m.floor(max(s1_len, s2_len) / 2) - 1

        s1_matches = [0] * s1_len
        s2_matches = [0] * s2_len

        match_count = 0
        trans_count = 0
        matching_prefix_size = 0

        for i in range(s1_len):
            for j in range(max(0, i - match_th), min(i + match_th + 1, s2_len)):
                if s1[i] == s2[j]:
                    match_count += 1
                    s1_matches[i] = 1
                    s2_matches[j] = 1

                    if i != j and not (i == s1_len - 1 and j == s2_len - 1):
                        trans_count += 1
                    break

        if match_count == 0:
            return float(match_count)

        trans_count = m.floor(trans_count / 2)

        jaro = (1/3) * ((match_count / s1_len) + (match_count / s2_len) + ((match_count - trans_count) / match_count))

        for i in range(min(s1_len, s2_len)):
            if s1[i] == s2[i]:
                matching_prefix_size += 1
            else:
                break

        if matching_prefix_size > matching_prefix_length:
            matching_prefix_size = matching_prefix_length

        jaro_winkler = jaro + (winkler_factor * matching_prefix_size * (1 - jaro))

        return jaro

    def load_data(self):

        # Read CSV file without headers
        df = self.spark.read.csv("./output/log.csv", header=False, inferSchema=True)

        # Assign custom column names
        custom_column_names = ["Caller", "Target", "Time", "EventType", "PID"]
        df = df.toDF(*custom_column_names)

        columns = df.columns

        # Iterate over each column and remove < and > characters
        for column in columns:
            df = df.withColumn(column, f.regexp_replace(df[column], "[<>]", ""))
        return df

    def get_distinct_servers(self, input_data):
        callers = input_data.select("Caller")
        distinct_servers = callers.union(input_data.select("Target")).distinct()
        return distinct_servers

    def calculate_similarity(self, distinct_servers):
        jaro_udf = f.udf(self.jaro_winkler_similarity, FloatType())
        distinct_servers.createOrReplaceTempView("distinct_servers")

        query = """
            select a.Caller as CallerName1, b.Caller as CallerName2
            from distinct_servers as a
            inner join distinct_servers as b
            on 1 = 1 and a.Caller != "None" and b.Caller != "None"
        """

        cross_joined_server_names = self.spark.sql(query)
        cross_joined_server_names = cross_joined_server_names.withColumn("Similarity", jaro_udf(cross_joined_server_names.CallerName1, cross_joined_server_names.CallerName2))
        return cross_joined_server_names

    def filter_similarity(self, cross_joined_server_names):
        return cross_joined_server_names.where(cross_joined_server_names.Similarity > self.jaro_th)

    @staticmethod
    def similarity_assignment(df):
        set_dict = {}
        handled = set()
        out_dict = {}

        i = 0
        for row in df.iterrows():
            if i == 0:
                set_dict[row[1]["CallerName1"]] = set([row[1]["CallerName2"]])
                handled.add(row[1]["CallerName1"])
                handled.add(row[1]["CallerName2"])
                i = 1
            else:
                if row[1]["CallerName1"] in set_dict.keys() and row[1]["CallerName2"] not in set_dict[row[1]["CallerName1"]]:
                    set_dict[row[1]["CallerName1"]].add(row[1]["CallerName2"])
                    handled.add(row[1]["CallerName2"])
                elif row[1]["CallerName1"] not in set_dict.keys() and row[1]["CallerName1"] not in handled and row[1]["CallerName2"] not in handled:
                    set_dict[row[1]["CallerName1"]] = set([row[1]["CallerName2"]])
                    handled.add(row[1]["CallerName1"])
                    handled.add(row[1]["CallerName2"])

        for key in set_dict.keys():
            for value in set_dict[key]:
                out_dict[value] = key

        return out_dict

    def apply_similarity_assignment(self, input_data, cross_joined_server_names):
        collapsed_data = input_data.na.replace(self.similarity_assignment(cross_joined_server_names.toPandas()))
        return collapsed_data

    def run(self):
        input_data = self.load_data()
        input_data.cache()

        distinct_servers = self.get_distinct_servers(input_data)
        # distinct_servers.show()

        cross_joined_server_names = self.calculate_similarity(distinct_servers)
        # cross_joined_server_names.show()

        filtered_similarity = self.filter_similarity(cross_joined_server_names)
        # filtered_similarity.show(truncate=100)

        self.collapsed_data = self.apply_similarity_assignment(input_data, filtered_similarity)
        self.collapsed_data.show(truncate=100)
        # collapsed_data.cache()

if __name__ == "__main__":

    string_similarity = StringSimilarity()
    string_similarity.run()