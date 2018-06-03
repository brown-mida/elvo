from argparse import ArgumentParser

from preprocessing.preprocessor import preprocess

# TODO: This doesn't work
if __name__ == '__main__':
    parser = ArgumentParser(description='Preprocesses the ELVO scans')
    parser.add_argument(
        'ct_dir',
        help='Path to the directory holding anonymized folders of CT scans',
    )
    parser.add_argument(
        'roi_dir',
        help='Path to the directory holding'
             ' anonymized folders of ROI annotations',
    )
    parser.add_argument(
        'output_dir',
        help='Path to write the processed data to',
    )
    args = parser.parse_args()
    preprocess(args.ct_dir, args.roi_dir, args.output_dir)
