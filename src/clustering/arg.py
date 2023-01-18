import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--output_dir", default=None, type=str, required=True,
        help="The output directory where the conll files and evaluate results will be written.",
    )
    parser.add_argument("--test_golden_filepath", default=None, type=str, required=True, 
        help="golden test set file path.",
    )
    parser.add_argument("--test_pred_filepath", default=None, type=str, required=False, 
        help="predicted coref file path.",
    )
    parser.add_argument("--test_cluster_filepath", default=None, type=str, required=False, 
        help="predicted cluster file path.",
    )
    parser.add_argument("--golden_conll_filename", default=None, type=str, required=True)
    parser.add_argument("--pred_conll_filename", default=None, type=str, required=True)
    
    # Other parameters
    parser.add_argument("--do_evaluate", action="store_true", help="Whether to evaluate conll files.")
    
    args = parser.parse_args()
    return args