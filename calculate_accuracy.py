from pyspark.sql.functions import col, count, desc, when
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number

class CalculateAccuracy:
    
    def __init__(self, spark_session):
        self.spark = spark_session

    def calculate_accuracy(self, match_df = None):
        print("I'm hereee!")

        # Read CSV file without headers
        stats = self.spark.read.csv("./output/stats.csv", header=True, inferSchema=True).selectExpr("process_id as id", "pattern_id")

        # match_df = self.spark.read.csv("final_similarity_groups.csv", header=True, inferSchema=True)

        match_df.show()

        # Join match_df with stats on id
        joined_df = match_df.join(stats, match_df.id == stats.id).drop(stats.id)
        print("joined df")
        joined_df.show()

        # Calculate total count of elements for each pattern_id
        total_counts_df = joined_df.groupBy("pattern_id").agg(count("*").alias("total_count"))

        # Group by component and pattern_id, and count occurrences
        grouped_df = joined_df.groupBy("component", "pattern_id").agg(count("*").alias("count"))
        print("grouped df")
        grouped_df.show()

        # Join grouped_df with total_counts_df
        comparison_df = grouped_df.join(total_counts_df, "pattern_id")
        print("comparison df")
        comparison_df.show()

        # Filter pairs where component count is more than half of the pattern_id total count
        filtered_df = comparison_df.filter(col("count") > (col("total_count") / 2))
        print("filtered df")
        filtered_df.show()

        # Find the most frequent component-pattern_id pairs
        window = Window.partitionBy("component").orderBy(col("count").desc())

        mapping_df = filtered_df.withColumn("rank", row_number().over(window)) \
                                .filter(col("rank") == 1) \
                                .selectExpr("component", "pattern_id as pattern_id_predicted")
        print("mapping df")
        mapping_df.show()

        # Join the original DataFrame with the inferred mapping
        final_comparison_df = joined_df.join(mapping_df, "component", "left_outer")
        print("final comparison df")
        final_comparison_df.show()

        # Compare the component predictions with the pattern_id
        final_comparison_df = final_comparison_df.withColumn("pattern_id_match", col("pattern_id_predicted") == col("pattern_id"))
        final_comparison_df = final_comparison_df.withColumn("pattern_id_match", when(col("pattern_id_match").isNull(), False).otherwise(col("pattern_id_match"))).orderBy(col("pattern_id"))
        final_comparison_df.show(n=1000)

        # Show the results
        final_comparison_df.select("id", "pattern_id_predicted", "pattern_id", "pattern_id_match")

        # Optionally, count the number of matching and non-matching records
        match_count = final_comparison_df.filter(col("pattern_id_match") == True).count()
        mismatch_count = final_comparison_df.filter(col("pattern_id_match") == False).count()

        print(f"Number of matching pattern_ids: {match_count}")
        print(f"Number of non-matching pattern_ids: {mismatch_count}")
        self.accuracy = match_count / (match_count + mismatch_count)
        print(self.accuracy)
