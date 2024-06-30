import os
from pyspark.sql.functions import col, array, collect_list, udf, sort_array, reverse, first


def part1_output(similarity_data, parsed_data):
    types_path = "./output/part1Output.txt"
    os.makedirs(os.path.dirname(types_path), exist_ok=True)

    a = set()

    # data = df.join(features, features.PID == df.id)
    open(types_path, "w").close()
    def f(event):
        with open(types_path, "a") as file:
            file.write(f"<{event.Caller},{event.Target},{event.Time},{event.EventType},{event.component}>\n")

    grouped_df = similarity_data.groupBy('component') \
                    .agg(first('id').alias("id")) \
                    .join(parsed_data, parsed_data.PID == col("id")) \
                    .sort(col("component"))     
    grouped_df.show(n=10)
    grouped_df.foreach(f)


def part1_observations(similarity_data, parsed_data):
    types_path = "./output/part1Observations.txt"
    os.makedirs(os.path.dirname(types_path), exist_ok=True)
    # data = df.join(features, features.PID == df.id)
    open(types_path, "w").close()

    a = set()

    def p1Obs(event):
        with open(types_path, "a") as file:
            file.write(f"\nGroup: {set(event.id)}\n")
            for i, id in enumerate(event.id):
                tmp_l = len(a)
                a.add(id)
                if tmp_l != len(a):
                    file.write(f"{id}\n")
            # file.write(f"{event.Caller}.txt\n")

                file.write(f"\t\t<{event.Caller[i]},{event.Target[i]},{event.Time[i]},{event.EventType[i]},{event.id[i]}>\n")

    # duration_udf = udf(f, IntegerType())
    grouped_df = similarity_data \
                    .join(parsed_data, parsed_data.PID == col("id")) \
                    .groupBy('component') \
                    .agg(collect_list('id').alias("id"),
                        collect_list('Caller').alias("Caller"),
                        collect_list('Target').alias("Target"),
                        collect_list('Time').alias("Time"),
                        collect_list('EventType').alias("EventType")) \
                    .sort(col("component"))             
    grouped_df.show(n=10)
    grouped_df.foreach(p1Obs)


def part2_observations(similarity_data, parsed_data, clusters):
    types_path = "./output/part2Observations.txt"
    os.makedirs(os.path.dirname(types_path), exist_ok=True)
    # data = df.join(features, features.PID == df.id)
    open(types_path, "w").close()

    a = set()

    def p2Obs(event):
        with open(types_path, "a") as file:
            file.write(f"\nCluster: {set(event.component)}\n")
            for i, id in enumerate(event.component):
                tmp_l = len(a)
                a.add(id)
                if tmp_l != len(a):
                    file.write(f"{id}\n")
            # file.write(f"{event.Caller}.txt\n")

                file.write(f"\t\t<{event.Caller[i]},{event.Target[i]},{event.Time[i]},{event.EventType[i]},{event.component[i]}>\n")

    # duration_udf = udf(f, IntegerType())
    grouped_df = clusters \
                    .join(similarity_data, similarity_data.component == clusters.component) \
                    .join(parsed_data, parsed_data.PID == col("id")) \
                    .groupBy('prediction') \
                    .agg(collect_list(clusters.component).alias("component"),
                        collect_list('Caller').alias("Caller"),
                        collect_list('Target').alias("Target"),
                        collect_list('Time').alias("Time"),
                        collect_list('EventType').alias("EventType"))
    grouped_df.show(n=10)
    grouped_df.foreach(p2Obs)