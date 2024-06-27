from pyspark.sql import SparkSession
from pyspark.sql.functions import col, count, desc, when
from pyspark.sql.window import Window
from pyspark.sql import *

class CalculateAccuracy:
    
    def __init__(self, spark_session):
        self.spark = spark_session

    def calculate_accuracy(self, match_df=None):
        print("Starting accuracy calculation...")

        # Read CSV file without headers
        stats = self.spark.read.csv("./output/stats.csv", header=True, inferSchema=True).selectExpr("process_id as id", "pattern_id")

        # Join match_df with stats on id
        joined_df = match_df.join(stats, match_df.id == stats.id).drop(stats.id)
        joined_df_to_work_on = joined_df
        joined_df.cache().count()
        # joined_df.show(truncate=False, n=1000)

        # Initialize an empty DataFrame to store the mappings
        mappings = []

        # Get unique pattern_ids
        pattern_ids = joined_df_to_work_on.select("pattern_id").distinct().rdd.flatMap(lambda x: x).collect()

        for curr_pattern_id in pattern_ids:
                # Find the component with the highest count for this pattern_id
                best_match = joined_df_to_work_on.filter(joined_df_to_work_on.pattern_id == curr_pattern_id).groupBy("component", "pattern_id").agg(count("*").alias("count")) \
                                       .orderBy(desc("count")) \
                                       .first()
                print(best_match)

                if best_match:
                    best_component = best_match["component"]
                    # Append the best match to mappings
                    mappings.append((best_component, curr_pattern_id))

                    # Remove matched component-pattern_id pairs from joined_df
                    joined_df_to_work_on = joined_df_to_work_on.filter(~((joined_df_to_work_on.component == best_component) | (joined_df_to_work_on.pattern_id == curr_pattern_id)))
                    #joined_df_to_work_on.show(truncate=False, n=1000)

        # Convert mappings to a DataFrame
        schema = ["component", "pattern_id"]
        mapping_df = self.spark.createDataFrame(mappings, schema).selectExpr("component", "pattern_id as mapping_pattern_id")
        #print("mapping_df:")
        #mapping_df.show()
        #print("joined_df:")
        #joined_df.show(truncate=False, n=1000)

        # Join the original DataFrame with the inferred mapping
        final_comparison_df = joined_df.join(mapping_df, "component", "left_outer")
        final_comparison_df.cache().count()
        #final_comparison_df.show(truncate=False, n=1000)

        # Compare the component predictions with the actual pattern_id
        final_comparison_df = final_comparison_df.withColumn("best_pattern_id_match", col("pattern_id") == col("mapping_pattern_id"))
        #final_comparison_df.show(truncate=False, n=1000)

        # Show the results
        # final_comparison_df.select("id", "pattern_id", "best_pattern_id_match").show(truncate=False)

        # Optionally, count the number of matching and non-matching records
        match_count = final_comparison_df.filter(col("best_pattern_id_match")).count()
        mismatch_count = final_comparison_df.count() - match_count

        print(f"Number of matching pattern_ids: {match_count}")
        print(f"Number of non-matching pattern_ids: {mismatch_count}")

        self.accuracy = match_count / (match_count + mismatch_count)
        print(f"Accuracy: {self.accuracy}")
