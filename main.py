import generator as g
import time

import string_similarity_classed as s
from shingling import Shingler
from lsh_no_banding_classed import MinHashLSHProcessor
from calculate_accuracy import CalculateAccuracy


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
    string_sim = s.StringSimilarity()
    string_sim.run()
    string_sim.collapsed_data.show(truncate=False)

    shing = Shingler(spark_session = string_sim.spark, df = string_sim.collapsed_data)
    shing.run()

    minhasher = MinHashLSHProcessor(spark_session=string_sim.spark, sparse_vector_df=shing.sparse_vectors_df)
    minhasher.run()
    minhasher.final_similarity_groups.toPandas().to_csv('final_similarity_groups.csv')

    acc_calculator = CalculateAccuracy(spark_session=string_sim.spark)
    acc_calculator.calculate_accuracy(match_df = minhasher.final_similarity_groups)

    solution_time = time.time() - st_solution
    final_acc = acc_calculator.accuracy

    jaro_th = string_sim.jaro_th
    jaccard_th = minhasher.CONFIG["JaccardDistThreshold"]

    minhash_signature_size = minhasher.CONFIG["MinHashSignatureSize"]

    if test:
        with open("config.yaml", 'r') as file:
            CONFIG = g.yaml.safe_load(file)

            process_max_depth = CONFIG["PROCESS_MAX_DEPTH"]
            process_max_length = CONFIG["PROCESS_MAX_LENGTH"]
            number_of_gold_patterns = CONFIG["PROCESS_PATTERN_NUMBER"]
            number_of_processes = CONFIG["PROCESSES_TO_GENERATE"]

        result_st = f"process_max_depth: {process_max_depth}, process_max_length: {process_max_length}, number_of_gold_patterns: {number_of_gold_patterns}, number_of_processes: {number_of_processes}, solution_time: {solution_time}, accuracy: {final_acc}, jaro_th: {jaro_th}, jaccard_th: {jaccard_th}, minhash_signature_size: {minhash_signature_size}"
        with open("./output/test_results.txt", "a") as file:
            file.write(result_st)


"""
def main():
    string_sim = s.StringSimilarity()
    acc_calculator = CalculateAccuracy(spark_session=string_sim.spark)
    acc_calculator.calculate_accuracy()
"""

if __name__ == "__main__":
    main()