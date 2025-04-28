import argparse

def parse_args():
    parser = argparse.ArgumentParser(
        description='EA input parameters',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--design_space_file', type=str, required=True,
                        help='Path to YAML file for the design space')
    parser.add_argument('--csv_file', type=str, required=True,
                        help='Path to CSV file with genome data')
    parser.add_argument('--population_size', type=int, default=5,
                        help='Population size')
    parser.add_argument('--fixed_length', type=int, default=22,
                        help='Fixed length for genome encoding')
    return parser.parse_args()