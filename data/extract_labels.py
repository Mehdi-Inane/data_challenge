
import argparse
import pandas as pd
import os
from PIL import Image
import numpy as np

def main():

    parser = argparse.ArgumentParser(description='Extract datasets for training and testing.')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label csv file')
    args = parser.parse_args()
    labels_path = args.label_path
    header = pd.read_csv(labels_path, nrows=0)  # reads only the header row
    output_dir = 'data/train/y_train'
    os.makedirs(output_dir,exist_ok = True)
    # Generate a list of column names excluding the first one
    cols_to_use = header.columns[1:]  # Skip the first column name
    print(header.columns[0])
    # Now, read the CSV excluding the first column
    df = pd.read_csv(labels_path, usecols=cols_to_use,dtype="int8")
    for col in df.columns:
        # Convert column to numpy array
        data = df[col].to_numpy()
        # Check if the data can be reshaped to 512x512
        if data.size == 512*512:
            # Reshape data
            reshaped_data = data.reshape((512, 512))

            # Convert to uint8 if necessary (assuming your data is in a compatible range, you might need to scale it)
            reshaped_data = reshaped_data.astype(np.uint8)

            # Save as PNG
            img = Image.fromarray(reshaped_data)
            img.save(f'{output_dir}/{col}')
        else:
            print(f"Column {col} does not have enough data to reshape to 512x512. It has {data.size} elements.")




if __name__ == '__main__':
    main()
