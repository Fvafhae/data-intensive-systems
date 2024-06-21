from pyspark.sql.functions import col, count
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

        #stats_grouped = stats.groupBy(stats.pattern_id)
        #match_df_grouped = stats.groupBy(match_df.component)

        #stats_grouped.show()

        joined_df = match_df.join(stats, match_df.id == stats.id).drop(stats.id)
        print("joined df")
        joined_df.show()


        # Group by component and pattern_id, and count occurrences
        grouped_df = joined_df.groupBy("component", "pattern_id").agg(count("*").alias("count"))
        print("grouped df")
        grouped_df.show()

        # Find the most frequent component-pattern_id pairs
        # We assume that we predicted more than 50% of the processes for each pattern_id
        window = Window.partitionBy("component").orderBy(col("count").desc())

        mapping_df = grouped_df.withColumn("rank", row_number().over(window)) \
                            .filter(col("rank") == 1) \
                            .selectExpr("component", "pattern_id as pattern_id_predicted")

        print("mapping df")
        mapping_df.show()

        # Join the original DataFrame with the inferred mapping
        comparison_df = joined_df.join(mapping_df, "component")
        comparison_df.show()

        # Compare the component predictions with the pattern_id
        comparison_df = comparison_df.withColumn("pattern_id_match", col("pattern_id_predicted") == col("pattern_id"))
        comparison_df.show()

        # Show the results
        comparison_df.select("id", "pattern_id_predicted", "pattern_id", "pattern_id_match").show()

        # Optionally, count the number of matching and non-matching records
        match_count = comparison_df.filter(col("pattern_id_match") == True).count()
        mismatch_count = comparison_df.filter(col("pattern_id_match") == False).count()

        print(f"Number of matching pattern_ids: {match_count}")
        print(f"Number of non-matching pattern_ids: {mismatch_count}")
        self.accuracy = match_count / (match_count + mismatch_count)
        print(self.accuracy)






