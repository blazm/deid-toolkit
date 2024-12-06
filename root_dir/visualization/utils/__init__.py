import argparse
import pandas as pd

RESULTS_DIR = "./root_dir/results"
LABELS_DIR = "./root_dir/datasets/labels"

def read_args():
    """
    Function to parse command-line arguments for visualization metrics.
    """
    def list_of_strings(arg):#custom argument
        return arg.split(',')
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Args to graphic metrics")
    
    # Positional arguments
    parser.add_argument('--evaluations', type=list_of_strings,required=True,  help='List of datasets')
    parser.add_argument("--datasets",  type=list_of_strings, required=True, help='List of datasets')
    parser.add_argument("--techniques", type=list_of_strings, required=True, help='List of techniques')
    parser.add_argument('--path_save', type=str,required=True , help='Paths of save pdf')
    #TODO check if the path to save contains extension
    #TODO: check if the path of the scores is a csv

    # Parse arguments
    args = parser.parse_args()
#    assert os.path.exists(args.aligned_path)
#    assert os.path.exists(args.deidentified_path)
#    if args.impostor_pairs_filepath is not None: 
#        assert os.path.exists(args.impostor_pairs_filepath)
#    if args.genuine_pairs_filepath is not None:
#        assert os.path.exists(args.genuine_pairs_filepath)
    return args


        
