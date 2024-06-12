

#TODO: define our syntetic data as data

#data = ....

#TODO: INITIALIZE SPARK

from pyspark.ml.linalg import Vectors, VectorUDT
import hashlib

##### What this code does #####

# 1. Create DataFrame for our LOG.
# 2. Filter out the requests
# 3. Shingle and hash this information per ProcessID
# 4. for each ProcessID, returns a dense vector of the hash value(s)

schema = StructType([StructField("FromServer", StringType(), True), StructField("ToServer", StringType(), True), StructField("time", LongType(), True), StructField("action", StringType(), True), StructField("ProcessId", LongType(), True)])
df = spark.createDataFrame(data, schema=schema)                                                     #create df for our LOG (data) file. uses above defined schema.


grouped_df = df.filter(col("action") == "Request").withColumn("ServerSequence", array(col("FromServer"), col("ToServer"))).groupBy("ProcessId").agg(collect_list("ServerSequence")).alias("ServerSequence")
                                                                                                    #we only care about requests (requests always results in response)
                                                                                                    #we create a column "ServerSequence", which is an array [ToServer, FromServer], for each request action
                                                                                                    #We group by processID and then aggregate the results; [ToServer, FromServer] into a list of array.

def hash_shingle(shingle):
    shingle_str = ''.join(map(str, shingle))                                                         # Convert k-shingle to string. Example (k=2): [[0,1], [1,3]] -> "[0,1][1,3]"
    return int(hashlib.md5(shingle_str.encode('utf-8')).hexdigest(), 16) % (10 ** 8)                 # hash this string, MOD 10**8 for distribution in one of 10**8 buckets

def generate_hashed_shingles(server_list, k):
    if len(server_list) < k:                                                                         #if processlenght<K just hash wathever process does exist
        return [hash_shingle(server_list)]
    shingles = [server_list[i:i+k] for i in range(len(server_list) - k + 1)]                         #else make list of all k-sized shingles.
    hashed_shingles = [hash_shingle(shingle) for shingle in shingles]                                #hash each shingle
    return hashed_shingles

hashed_shingle_udf = udf(lambda x: generate_hashed_shingles(x, 3), ArrayType(IntegerType()))                        #maps our previously defined function
hashed_df = grouped_df.withColumn("hashed_shingles", hashed_shingle_udf(grouped_df["collect_list(ServerSequence)"]))   #executes the hash shingle function over the filtered dataset.

Dense_vectors = hashed_df.select(
    col("ProcessId"),                                                                                #Initial idea: Create dense vectors just by sorting the
    sort_array(col("hashed_shingles")).alias("dense_vectors")                                        #Hashed shingled into ascending order.
)

# def array_to_vector(array):
#     return Vectors.dense(array)
#
# array_to_vector_udf = udf(array_to_vector, VectorUDT())                                           #If built in pyspark functions don't accept these sorted shingles
#                                                                                                   #convert to this format. (And probably remove the sorting code above)
# # Apply the UDF to convert the hashed_shingles column to VectorUDT
# dense_vectors_df = hashed_df.withColumn(
#     "dense_vectors", array_to_vector_udf(col("hashed_shingles"))
# )
#Dense_vectors = dense_vectors_df.select(
#     col("ProcessId"),
#     sort_array(col("hashed_shingles")).alias("dense_vectors")
# )

return Dense_vectors