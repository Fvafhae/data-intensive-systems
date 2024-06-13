import generator as g
import time

import string_similarity_classed as s
from shingling import Shingler
from lsh_no_banding_classed import MinHashLSHProcessor

def main():
    # Data generation
    st = time.time()
    generator = g.Generator()
    generator._set_()
    generator.generate_process_patterns()
    generator.populate_processes()
    print("--- %s seconds ---" % (time.time() - st))

    # string similarity is used
    string_sim = s.StringSimilarity()
    string_sim.run()
    string_sim.collapsed_data.show(truncate=False)

    shing = Shingler(spark_session = string_sim.spark, df = string_sim.collapsed_data)
    shing.run()

    minhasher = MinHashLSHProcessor(spark_session=string_sim.spark, sparse_vector_df=shing.sparse_vectors_df)
    minhasher.run()





if __name__ == "__main__":
    main()