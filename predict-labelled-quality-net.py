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
                        help="Path to checkpoint.")
    parser.add_argument('--output_path',dest='output_path',
                        action='store',type=str,default=None,
                        help="Path to output HDF5.")
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
        for image,label,path in data_generator.generate(with_path=True):
            yield image,label,path

    generator = load_generator
    output_types = (tf.float32,tf.float32,tf.string)
    output_shapes = (
        tf.TensorShape((args.input_height,args.input_width,3)),
        tf.TensorShape((1)),tf.TensorShape((1)))
    tf_dataset = tf.data.Dataset.from_generator(
        generator,output_types=output_types,output_shapes=output_shapes)
    tf_dataset = tf_dataset.batch(args.batch_size)
    tf_dataset = tf_dataset.prefetch(args.batch_size*5)

    print("Predicting...")
    auc = tf.keras.metrics.AUC()
    acc = Accuracy()
    rec = Recall()
    prec = Precision()

    with h5py.File(args.output_path,'w') as F:
        for images,labels,Ps in tqdm(tf_dataset):
            predictions = quality_net.predict(images)
            auc.update_state(labels,predictions)
            acc.update_state(labels,predictions)
            rec.update_state(labels,predictions)
            prec.update_state(labels,predictions)
            zipped_output = zip(
                images.numpy(),labels.numpy(),Ps.numpy(),predictions)
            for image,label,P,prediction in zipped_output:
                P = str(P[0].decode('utf-8'))
                g = F.create_group(P)
                g['image'] = np.uint8(image*255)
                g['label'] = np.uint8(label)
                g['prediction'] = prediction

    print('{},{}'.format(
        'AUC',float(auc.result().numpy())))
    print('{},{}'.format(
        'Accuracy',float(acc.result().numpy())))
    print('{},{}'.format(
        'Recall',float(rec.result().numpy())))
    print('{},{}'.format(
        'Precision',float(prec.result().numpy())))