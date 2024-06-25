import generator as g
import time

import string_similarity_classed as s
from shingling import Shingler
from lsh_no_banding_classed import MinHashLSHProcessor
from calculate_accuracy import CalculateAccuracy
import json
import os

# TODO: Turn the output into a json for easier evaluation
# TODO: Create a scheme for automatic runs:
    # Create data once, traverse the parameter values.
    # Repeat many times.
    # parameters to be set: jaro_th, jaccard_th, length of the signature matrix.

def tester(file_path):
    test_case_count = 3
    for i in range(0, test_case_count):
        
        # Data generation once for each case
        st = time.time()
        generator = g.Generator()
        generator._set_()
        generator.generate_process_patterns()
        generator.populate_processes()

        with open("config.yaml", 'r') as file:
            CONFIG = g.yaml.safe_load(file)

            process_max_depth = CONFIG["PROCESS_MAX_DEPTH"]
            process_max_length = CONFIG["PROCESS_MAX_LENGTH"]
            number_of_gold_patterns = CONFIG["PROCESS_PATTERN_NUMBER"]
            number_of_processes = CONFIG["PROCESSES_TO_GENERATE"]

        for jaro_loop in range(1, 11):
            jaro_th = jaro_loop * 0.1
            for jaccard_loop in range(1, 11):
                jaccard_th = jaccard_loop * 0.1
                st_solution = time.time()

                # string similarity is used
                string_sim = s.StringSimilarity(jaro_th=jaro_th)
                string_sim.run()
                string_sim.collapsed_data.show(truncate=False)

                shing = Shingler(spark_session = string_sim.spark, df = string_sim.collapsed_data)
                shing.run()

                minhasher = MinHashLSHProcessor(spark_session=string_sim.spark, sparse_vector_df=shing.sparse_vectors_df, jaccard_th=jaccard_th)
                minhasher.run()
                minhasher.final_similarity_groups.toPandas().to_csv('final_similarity_groups.csv')

                acc_calculator = CalculateAccuracy(spark_session=string_sim.spark)
                acc_calculator.calculate_accuracy(match_df = minhasher.final_similarity_groups)

                solution_time = time.time() - st_solution
                final_acc = acc_calculator.accuracy

                minhash_signature_size = minhasher.CONFIG["MinHashSignatureSize"]

                result = {
                        "id": str(i) + str(jaro_loop) + str(jaccard_loop),
                        "process_max_depth": process_max_depth,
                        "process_max_length": process_max_length,
                        "number_of_gold_patterns": number_of_gold_patterns,
                        "number_of_processes": number_of_processes,
                        "solution_time": solution_time,
                        "accuracy": final_acc,
                        "jaro_th": jaro_th,
                        "jaccard_th": jaccard_th,
                        "minhash_signature_size": minhash_signature_size
                }
                    
                # Function to read existing data from the JSON file
                def read_json_file(file_path):
                    if os.path.exists(file_path):
                        with open(file_path, 'r') as json_file:
                            return json.load(json_file)
                    else:
                        return {}
                        
                existing_data = read_json_file(file_path)

                with open(file_path, 'w') as file:

                    existing_data[str(i) + str(jaro_loop) + str(jaccard_loop)] = result
                       
                    # file_data[str(test_case_count) + str(jaro_loop) + str(jaccard_loop)] = result
                    # convert back to json.
                    json.dump(existing_data, file, indent=4)
                        
                string_sim.spark.stop()

tester(file_path = "./output/test_results.json")
