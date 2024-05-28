import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql import *
import pyspark.sql.functions as f
import math as m
from pyspark.sql.types import FloatType


# implement Jaro-Winkler similarity as a function
# we can pass this function in spark.map

# First we define Jaro similarity which is to be used in Jaro-Winkler
def jaro_winkler_similarity(s1, s2, winkler_factor=0.1, matching_prefix_length = 20):
    
    # if the two strings are equal, return 1
    if s1 == s2:
        return 1.0
    
    # if not, calculate Jaro similarity
    # get string lengths
    s1_len = len(s1)
    s2_len = len(s2)
    

    # get matching threshold. If two characters are the same and they are not
    # further than this threshold, then we say that there is a match
    match_th = m.floor(max(s1_len, s2_len) / 2) - 1

    s1_matches = [0] * s1_len
    s2_matches = [0] * s2_len

    # find the mathces:
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
                    # print(i, j)
                    trans_count += 1
                break

    # if there are no matches then similarity is 0.0
    if match_count == 0:
        return float(match_count)

    trans_count = m.floor(trans_count / 2)

    # if there are mathces, calculate Jaro Similarity:
    jaro = (1/3) * ((match_count / s1_len) + (match_count / s2_len) + ((match_count - trans_count) / match_count))

    # we need the length of the matching prefix
    for i in range(min(s1_len, s2_len)):
        if s1[i] == s2[i]:
            matching_prefix_size += 1
        else:
            break
    
    if matching_prefix_size > matching_prefix_length:
        matching_prefix_size = matching_prefix_length

    # now we can calculate Jaro-Winkler similarity:
    jaro_winkler = jaro + (winkler_factor * matching_prefix_size * (1 - jaro))

    #return jaro_winkler
    return jaro


### Stop spark if you have created spark session before
#spark.stop()
# create the session
conf = SparkConf()
conf.setAppName("string_similarity")
conf.setMaster("local[1]")
conf.set("spark.driver.memory", "2G")
conf.set("spark.driver.maxResultSize", "2g")
conf.set("spark.executor.memory", "1G")
sc = pyspark.SparkContext(conf=conf)
spark = SparkSession.builder.getOrCreate()
### Creating spark session with inline configurations ###
# spark = SparkSession.builder \
#     .appName("DIS-lab-1") \
#     .master("local[*]") \
#     .config("spark.driver.memory", "2G") \
#     .config("spark.driver.maxResultSize", "2g") \
#     .getOrCreate()
spark


# < null, S1, 0, Request, 1821 >
# < S1, S2, 1, Request, 1821 >
# < null, S1, 3, Request, 1978 >
# < S1, S3, 46, Request, 1978 >
# < S2, S1, 51, Response, 1821 > 
# < S1, null, 62, Response, 1821 > 
# < S3, S1, 71, Response, 1978 >

# Lets create dummy data
dummy_data = spark.createDataFrame([
(None, "MainServer1", 0, "Request", 0),
("MainServer1", "Authentication1", 1, "Request", 0),
("Authentication1", "MainServer1", 2, "Response", 0),
(None, "MainServer1", 0, "Request", 1),
("MainServer1", "CreditCardMasterCard1", 3, "Request", 0),
("MainServer1", "Authentication2", 3, "Request", 1),
("Authentication2", "MainServer1", 7, "Response", 0),
("CreditCardMasterCard1", "MainServer1", 8, "Response", 0),
("MainServer1", "CreditCardVisa1", 8, "Request", 0),
("CreditCardVisa1", "MainServer1", 10, "Response", 0),
("MainServer1", "CreditCardVisa2", 11, "Request", 0),
("CreditCardVisa2", "MainServer1", 12, "Response", 0),
("MainServer2", "CreditCardMasterCard2", 20, "Request", 0),
("CreditCardMasterCard2", "MainServer2", 22, "Response", 0)
], ["Caller", "Target", "Time", "EventType", "PID"])

dummy_data.show()


# Get distinct server names from the both columns
callers = dummy_data.select("Caller")

distinct_servers = callers.union(dummy_data.select("Target")).distinct()
distinct_servers.show()

jaro_udf = f.udf(jaro_winkler_similarity, FloatType())

# Create SQL view
distinct_servers.createOrReplaceTempView("distinct_servers")

query = """
    select a.Caller as CallerName1, b.Caller as CallerName2
    from distinct_servers as a
    inner join distinct_servers as b
    on 1 = 1
    where a.Caller != "None" and b.Caller != "None"
"""

cross_joined_server_names = spark.sql(query)
cross_joined_server_names = cross_joined_server_names.withColumn("JaroWinkler", jaro_udf(cross_joined_server_names.CallerName1, cross_joined_server_names.CallerName2))
cross_joined_server_names.show()

# Next, we should replace the similar server names


