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

    # we calculated JaroWinkler too. We can use either.
    # return jaro_winkler
    return jaro


### Stop spark if you have created spark session before
#spark.stop()
# create the session
conf = SparkConf()
conf.setAppName("string_similarity")
conf.setMaster(f"local[{CONFIG['CoreCount']}]")
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
input_data = spark.createDataFrame([
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
("CreditCardMasterCard2", "MainServer2", 22, "Response", 0),
("TravelCardVisa1", "MainServer2", 22, "Request", 0),
("MainServer2", "TravelCardVisa1", 22, "Response", 0)
], ["Caller", "Target", "Time", "EventType", "PID"])

input_data.show()
input_data.cache()


# Get distinct server names from the both columns
# Caller servers
callers = input_data.select("Caller")

# Add target servers too.
distinct_servers = callers.union(input_data.select("Target")).distinct()
distinct_servers.show()

# Assign similarity function as a user defined function (udf) for use in spark
jaro_udf = f.udf(jaro_winkler_similarity, FloatType())

# Create SQL view
distinct_servers.createOrReplaceTempView("distinct_servers")

# Cartesian product of the servers
query = """
    select a.Caller as CallerName1, b.Caller as CallerName2
    from distinct_servers as a
    inner join distinct_servers as b
    on 1 = 1 and a.Caller != "None" and b.Caller != "None"
"""


# This similarity calculation runs in parallel, OK.
cross_joined_server_names = spark.sql(query)
cross_joined_server_names = cross_joined_server_names.withColumn("Similarity", jaro_udf(cross_joined_server_names.CallerName1, cross_joined_server_names.CallerName2))
cross_joined_server_names.show()

# Next, we should replace the similar server names
# We should first filter higher similarity rows based on a threshold
# TODO: We should choose a threshold for server name similarity score
# Since we decrease the cross-product matrix size based on a similaritry threshold, we can turn this into a dataframe with no problem.
string_sim_th = 0.7
cross_joined_server_names = cross_joined_server_names.where(cross_joined_server_names.Similarity > string_sim_th)
cross_joined_server_names.show(truncate=100)


# We create a dictionary of servers, using it we'll call each server type with 1 server's name
# TODO: We can also think if we can do this in a partitioned manner.
# Example:
# CreditCardMasterCard1: CreditCardMasterCard2, CreditCardMasterCard3, CreditCardMaestro1
# All these servers will be replaced with CreditCardMasterCard1 in the input
# It's OK for this function to run sequentially since we've already made the cross-product matrix quite small.
def similarity_assignment(df):
    # main dictionary
    set_dict = {}
    # a set of handled servers
    handled = set()
    # output dict, with this, we wont need to iterate through all the sets while performing the actual replacement
    out_dict = {}

    i = 0
    # For each row in the similarity matrix
    for row in df.iterrows():
        # initialize the dict structure so we can iterate through it
        # if the first row of the matrix
        if i == 0:
            # Add servername as a key and add the second server to its set
            set_dict[row[1]["CallerName1"]] = set([row[1]["CallerName2"]])
            # Add both servers to the handled servers list
            handled.add(row[1]["CallerName1"])
            handled.add(row[1]["CallerName2"])
            i = 1
        else:
            # if the first server exists as a key but the second server is not in its set,
            # Add the second server to the first's set
            if row[1]["CallerName1"] in set_dict.keys() and row[1]["CallerName2"] not in set_dict[row[1]["CallerName1"]]:
                set_dict[row[1]["CallerName1"]].add(row[1]["CallerName2"])
                handled.add(row[1]["CallerName2"])
            # If both servers were never handled create a new entry in the dict
            elif row[1]["CallerName1"] not in set_dict.keys() and row[1]["CallerName1"] not in handled and row[1]["CallerName2"] not in handled:
                set_dict[row[1]["CallerName1"]] = set([row[1]["CallerName2"]])
                handled.add(row[1]["CallerName1"])
                handled.add(row[1]["CallerName2"])

    for key in set_dict.keys():
        for value in set_dict[key]:
            out_dict[value] = key

    return out_dict

# Now that  we've written this function, we need to apply it to the input file.
# Can we send different parts of the input file to different cores (transform workers) and collect the results and append them?

# The replacement runs in parallel too. Hence the huge input file is going to be processed in parallel.
# This is the second time we read the input data, better cache it.
collapsed_data = input_data.na.replace(similarity_assignment(cross_joined_server_names.toPandas()))
collapsed_data.show(truncate=100)
collapsed_data.cache()

print(similarity_assignment(cross_joined_server_names.toPandas()))
for x, y in similarity_assignment(cross_joined_server_names.toPandas()).items():
    print(f"{x}: {y}")
