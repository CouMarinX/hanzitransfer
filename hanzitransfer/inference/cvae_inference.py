import argparse
import numpy as np
import tensorflow as tf
from PIL import Image

from . import load_cvae_decoder

LATENT_DIM = 16

def main():
    parser = argparse.ArgumentParser(description="Generate images from trained CVAE decoder")
    parser.add_argument("--radical", type=int, required=True, help="Radical index for conditioning")
    parser.add_argument("--num_radicals", type=int, required=True, help="Total number of radical classes")
    parser.add_argument("--output", type=str, default="generated.png", help="Output image file")
    args = parser.parse_args()

    decoder = load_cvae_decoder()
    radical_vec = tf.keras.utils.to_categorical(args.radical, num_classes=args.num_radicals)
    z = np.random.normal(size=(1, LATENT_DIM))
    decoder_input = np.concatenate([z, radical_vec.reshape(1, -1)], axis=1)
    img = decoder.predict(decoder_input)[0, :, :, 0]
    Image.fromarray((img * 255).astype(np.uint8)).save(args.output)
    print(f"Saved generated image to {args.output}")

if __name__ == "__main__":
    main()
