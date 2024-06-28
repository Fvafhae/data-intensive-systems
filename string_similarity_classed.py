from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import *
import pyspark.sql.functions as f
import math as m
from pyspark.sql.types import FloatType
from collections import defaultdict

class StringSimilarity:
    def __init__(self, core_count=8, jaro_th=0.1, edit_th=10, jaro_or_edit="jaro"):
        self.core_count = core_count
        self.spark = self._create_spark_session()
        self.collapsed_data = None
        self.jaro_or_edit = jaro_or_edit
        if jaro_or_edit == "jaro":
            self.th = jaro_th
        else:
            self.th = edit_th

        self.input_length = 0

    def _create_spark_session(self):
        conf = SparkConf().setAppName("minhash").setMaster("local[8]")
        conf.set("spark.driver.memory", "4G")
        conf.set("spark.driver.maxResultSize", "4g")
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
                if s1[i] == s2[j] and not s2_matches[j]:
                    match_count += 1
                    s1_matches[i] = 1
                    s2_matches[j] = 1
                    break

        if match_count == 0:
            return 0.0

        k = 0
        for i in range(s1_len):
            if s1_matches[i]:
                while not s2_matches[k]:
                    k += 1
                if s1[i] != s2[k]:
                    trans_count += 1
                k += 1

        trans_count /= 2

        jaro = (match_count / s1_len + match_count / s2_len + (match_count - trans_count) / match_count) / 3

        for i in range(min(s1_len, s2_len)):
            if s1[i] == s2[i]:
                matching_prefix_size += 1
            else:
                break

        matching_prefix_size = min(matching_prefix_size, matching_prefix_length)

        return jaro + (winkler_factor * matching_prefix_size * (1 - jaro))

    @staticmethod
    def edit_dist(s1, s2):
        m = len(s1)
        n = len(s2)
    
        # Initialize a matrix to store the edit distances
        dp = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
        # Initialize the first row and column with values from 0 to m and 0 to n respectively
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
    
        # Fill the matrix using dynamic programming to compute edit distances
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i - 1] == s2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
    
        return dp[m][n]

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
        edit_udf = f.udf(self.edit_dist, FloatType())
        distinct_servers.createOrReplaceTempView("distinct_servers")

        query = """
            select a.Caller as CallerName1, b.Caller as CallerName2
            from distinct_servers as a
            inner join distinct_servers as b
            on 1 = 1 and a.Caller != "None" and b.Caller != "None"
        """

        cross_joined_server_names = self.spark.sql(query)
        if self.jaro_or_edit == "jaro":
            cross_joined_server_names = cross_joined_server_names.withColumn("Similarity", jaro_udf(cross_joined_server_names.CallerName1, cross_joined_server_names.CallerName2))
        else:
            cross_joined_server_names = cross_joined_server_names.withColumn("Similarity", edit_udf(cross_joined_server_names.CallerName1, cross_joined_server_names.CallerName2))
        # cross_joined_server_names.show(truncate=False, n=1000)
        return cross_joined_server_names

    def filter_similarity(self, cross_joined_server_names):
        if self.jaro_or_edit == "jaro":
            return cross_joined_server_names.where(cross_joined_server_names.Similarity > self.th)
        else:
            return cross_joined_server_names.where(cross_joined_server_names.Similarity < self.th)

    @staticmethod
    def similarity_assignment(df):
        class UnionFind:
            def __init__(self):
                self.parent = {}

            def find(self, x):
                if self.parent[x] != x:
                    self.parent[x] = self.find(self.parent[x])
                return self.parent[x]

            def union(self, x, y):
                rootX = self.find(x)
                rootY = self.find(y)
                if rootX != rootY:
                    self.parent[rootY] = rootX

            def add(self, x):
                if x not in self.parent:
                    self.parent[x] = x

        uf = UnionFind()

        for _, row in df.iterrows():
            server1 = row["CallerName1"]
            server2 = row["CallerName2"]
        
            uf.add(server1)
            uf.add(server2)
            uf.union(server1, server2)
    
        groups = defaultdict(set)
        for server in uf.parent:
            root = uf.find(server)
            groups[root].add(server)
    
        out_dict = {}
        for root, servers in groups.items():
            for server in servers:
                out_dict[server] = root
    
        # print("Similar servers: \n")
        # print(out_dict)
        return out_dict

    def apply_similarity_assignment(self, input_data, cross_joined_server_names):
        collapsed_data = input_data.na.replace(self.similarity_assignment(cross_joined_server_names.toPandas()))
        # collapsed_data.show(truncate=False, n=1000)
        return collapsed_data

    def run(self):
        input_data = self.load_data()
        input_data.cache()
        self.input_length = input_data.count()

        distinct_servers = self.get_distinct_servers(input_data)

        cross_joined_server_names = self.calculate_similarity(distinct_servers)

        filtered_similarity = self.filter_similarity(cross_joined_server_names)

        self.collapsed_data = self.apply_similarity_assignment(input_data, filtered_similarity)

if __name__ == "__main__":
    string_similarity = StringSimilarity()
    string_similarity.run()
