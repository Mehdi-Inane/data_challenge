import zipfile
import argparse

def main():
    parser = argparse.ArgumentParser(description='Extract datasets for training and testing.')
    parser.add_argument('--train_zip_path', type=str, required=True, help='Path to the training zip file')
    parser.add_argument('--test_zip_path', type=str, required=True, help='Path to the testing zip file')
    parser.add_argument('--extract_folder', type=str, required=True, help='Folder where the contents will be extracted')
    # Parse arguments
    args = parser.parse_args()
    # Unzipping the training dataset
    with zipfile.ZipFile(args.train_zip_path, 'r') as zip_ref:
        print(f'Extracting {args.train_zip_path} to {args.extract_folder}/train')
        zip_ref.extractall(args.extract_folder+'/train')
    # Unzipping the testing dataset
    with zipfile.ZipFile(args.test_zip_path, 'r') as zip_ref:
        print(f'Extracting {args.test_zip_path} to {args.extract_folder}/test')
        zip_ref.extractall(args.extract_folder+'/test')


if __name__ == '__main__':
    main()
