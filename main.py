import generator as g
import time

import string_similarity_classed as s
from shingling import Shingler
from lsh_no_banding_classed import MinHashLSHProcessor
from calculate_accuracy import CalculateAccuracy

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

    st_solution = time.time()

    # string similarity is used
    string_sim = s.StringSimilarity(jaro_th=0.3)
    string_sim.run()
    string_sim.collapsed_data.show(truncate=False)

    shing = Shingler(spark_session = string_sim.spark, df = string_sim.collapsed_data)
    shing.run()

    minhasher = MinHashLSHProcessor(spark_session=string_sim.spark, sparse_vector_df=shing.sparse_vectors_df, jaccard_th=0.3)
    minhasher.run()
    minhasher.final_similarity_groups.toPandas().to_csv('final_similarity_groups.csv')

    acc_calculator = CalculateAccuracy(spark_session=string_sim.spark)
    acc_calculator.calculate_accuracy(match_df = minhasher.final_similarity_groups)

    solution_time = time.time() - st_solution
    final_acc = acc_calculator.accuracy

    print(solution_time)

"""
def main():
    string_sim = s.StringSimilarity()
    acc_calculator = CalculateAccuracy(spark_session=string_sim.spark)
    acc_calculator.calculate_accuracy()
"""

if __name__ == "__main__":
    main()