import argparse
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras

from quality_net_utilities import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Trains a feature-aware normalization model.')

    parser.add_argument('--dataset_path',dest='dataset_path',
                        action='store',type=str,default=None,
                        help="Path to hdf5 dataset.")
    parser.add_argument('--checkpoint_path',dest='checkpoint_path',
                        action='store',type=str,default=None,
                        help="Path to hdf5 dataset.")
    parser.add_argument('--input_height',dest = 'input_height',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')
    parser.add_argument('--input_width',dest = 'input_width',
                        action = 'store',type = int,default = 256,
                        help = 'The file extension for all images.')

    parser.add_argument('--batch_size',dest = 'batch_size',
                        action = 'store',type = int,default = 4,
                        help = 'Size of mini batch.')

    args = parser.parse_args()

    print("Setting up network...")
    quality_net = keras.models.load_model(args.checkpoint_path)

    print("Setting up data generator...")
    data_generator = DataGenerator(args.dataset_path,None)
    def load_generator():
        for image,label in data_generator.generate():
            yield image,label

    generator = load_generator
    output_types = (tf.float32,tf.float32)
    output_shapes = (
        tf.TensorShape((args.input_height,args.input_width,3)),
        tf.TensorShape((1)))
    tf_dataset = tf.data.Dataset.from_generator(
        generator,output_types=output_types,output_shapes=output_shapes)
    tf_dataset = tf_dataset.batch(args.batch_size)
    tf_dataset = tf_dataset.prefetch(args.batch_size*5)

    print("Setting up testing...")
    auc = tf.keras.metrics.AUC()
    acc = Accuracy()
    rec = Recall()
    prec = Precision()

    print("Testing...")
    for image,c in tqdm(tf_dataset):
        prediction = quality_net(image)
        auc.update_state(c,prediction)
        acc.update_state(c,prediction)
        rec.update_state(c,prediction)
        prec.update_state(c,prediction)

    print('{},{}'.format(
        'AUC',float(auc.result().numpy())))
    print('{},{}'.format(
        'Accuracy',float(acc.result().numpy())))
    print('{},{}'.format(
        'Recall',float(rec.result().numpy())))
    print('{},{}'.format(
        'Precision',float(prec.result().numpy())))