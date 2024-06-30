import generator as g
import time

import string_similarity_classed as s
from shingling import Shingler
from lsh_no_banding_classed import MinHashLSHProcessor
from calculate_accuracy import CalculateAccuracy

from clusterring import Clustering
import output
import spark_config

# TODO: Turn the output into a json for easier evaluation
# TODO: Create a scheme for automatic runs:
    # Create data once, traverse the parameter values.
    # Repeat many times.
    # parameters to be set: jaro_th, jaccard_th, length of the signature matrix.

def main():
    test = True

    # Data generation
    st = time.time()
    generator = g.Generator()
    generator._set_()
    generator.generate_process_patterns()
    generator.populate_processes()
    print("--- %s seconds ---" % (time.time() - st))

    spark = spark_config._create_spark_session()
    st_solution = time.time()
    
    # string similarity is used
    string_sim = s.StringSimilarity(spark=spark, jaro_or_edit="jaro", edit_th=5, jaro_th=0.8)
    string_sim.run()

    # string_sim.collapsed_data.show(truncate=False)
    print("Start Clustering...")
    clustering = Clustering(spark_session=spark)

    features = clustering._calculate_initial_features_(string_sim.collapsed_data)
    print("Start Shingling...")

    shing = Shingler(spark_session = string_sim.spark, df = string_sim.collapsed_data)
    shing.run()

    print("Start MinHashing...")

    minhasher = MinHashLSHProcessor(spark_session=string_sim.spark, sparse_vector_df=shing.sparse_vectors_df, jaccard_th=0.2, signature_size=20)
    minhasher.run()
    minhasher.final_similarity_groups.toPandas().to_csv('final_similarity_groups.csv')

    solution_time = time.time() - st_solution
    print(solution_time)

    # features = ss._merge_selection_(minhasher.final_similarity_groups, features)
    
    minhasher.final_similarity_groups.show(n=50)

    print("Calculate Accuracy\n")

    acc_calculator = CalculateAccuracy(spark_session=string_sim.spark)
    acc_calculator.calculate_accuracy(match_df = minhasher.final_similarity_groups)

    solution_time = time.time() - st_solution
    final_acc = acc_calculator.accuracy
    print("Part 1 Output\n")

    output.part1_output(minhasher.final_similarity_groups, string_sim.collapsed_data)
    output.part1_observations(minhasher.final_similarity_groups, string_sim.collapsed_data)

    print("Calculate clustering features\n")

    data = clustering._prepare_all_features_(minhasher.final_similarity_groups, features=features)
    print("Predict clusters\n")

    clusters = clustering.make_predictions(data)

    print("Part 2 Output\n")
    output.part2_observations(minhasher.final_similarity_groups, string_sim.collapsed_data, clusters)


"""
def main():
    string_sim = s.StringSimilarity()
    acc_calculator = CalculateAccuracy(spark_session=string_sim.spark)
    acc_calculator.calculate_accuracy()
"""

if __name__ == "__main__":
    main()