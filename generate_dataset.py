import h5py
import argparse
import os
from skimage import io
from tqdm import tqdm 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Builds an HDF5 dataset from a CSV with paths.')

    parser.add_argument('--csv_path',dest='csv_path',
                        action='store',type=str,default=None,
                        help="Path to csv file with one image path per line.")
    parser.add_argument('--output_path',dest='output_path',
                        action='store',type=str,default=None,
                        help="Path to output HDF5.")

    args = parser.parse_args()

    with open(args.csv_path,'r') as o:
        lines = [x.strip().split(',') for x in o.readlines()]
        image_and_classes = {l[0]:int(l[1]) for l in lines}

    with h5py.File(args.output_path,'w') as F:
        for k in tqdm(image_and_classes):
            k_ = k.split(os.sep)[-1]
            g = F.create_group(k_)
            g.create_dataset(
                'image',data=io.imread(k),dtype='uint8')
            g.create_dataset(
                'class',data=image_and_classes[k],dtype='uint16')