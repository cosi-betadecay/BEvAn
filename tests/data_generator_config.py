import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.synthetic_data_generator import SyntheticDataGenerator

gen_1000_2_true = SyntheticDataGenerator(max_hits_per_sequence=2, seed=0).generate(num_samples=1000, mode=1)
gen_2000000_4_true = SyntheticDataGenerator(max_hits_per_sequence=4, seed=0).generate(
    num_samples=2000000, mode=1
)
gen_200_8_true = SyntheticDataGenerator(max_hits_per_sequence=8, seed=0).generate(num_samples=200, mode=1)
gen_100000_24_true = SyntheticDataGenerator(max_hits_per_sequence=24, seed=0).generate(
    num_samples=100000, mode=1
)

gen_1000_2_false = SyntheticDataGenerator(max_hits_per_sequence=2, seed=0).generate(num_samples=1000, mode=2)
gen_2000000_4_false = SyntheticDataGenerator(max_hits_per_sequence=4, seed=0).generate(
    num_samples=2000000, mode=2
)
gen_200_8_false = SyntheticDataGenerator(max_hits_per_sequence=8, seed=0).generate(num_samples=200, mode=2)
gen_100000_24_false = SyntheticDataGenerator(max_hits_per_sequence=24, seed=0).generate(
    num_samples=100000, mode=2
)
