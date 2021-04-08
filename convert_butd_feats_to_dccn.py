import numpy as np
import torch
import os, sys, argparse


def convert_to_dccn(objects_vocab, img_attrs, model_vocab_file, imglist, input_folder):
    # objects_vocab : list of object/class names (len(objects_vocab) == img_attr.shape[2])
    # img_attrs : (num_images, num_regions, num_classes, 2)
    # model_vocab_file : torch file containing the processed dictionary for the model
    # imglist : list of images for this split
    # input_folder : folder containing the npz files

    # Read from processed vocabulary file
    model_vocab_src = torch.load(model_vocab_file)[0][1] # default dict from words to indices
    # Convert to indexes into the DCCN vocab list
    objects_model_vocab_id = np.vectorize(lambda x: model_vocab_src[x])(objects_vocab)
    # objects_model_vocab_mask = objects_model_vocab_id == 0

    img_attrs[:,:,:,0] = objects_model_vocab_id.reshape((1,1,img_attrs.shape[2]))
    
    for i in range(len(imglist)):        
        z = np.load(os.path.join(input_folder, imglist[i] + ".npz"))
        score = z["objects_score"]
        # if there is no corresponding object name in the model name, then set conf to 0
        # use bpe encoding for objects_vocab?
        # score[:,objects_model_vocab_mask] = 0
        img_attrs[i, :, :, 1] = score

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert from bottom-up-attention.pytorch output to DCCN output.")

    parser.add_argument(
        "-f", "--file_names", help="File containing list of file names for this split")
    parser.add_argument(
        "-i", "--input_folder", help="Folder containing input .npz files")
    parser.add_argument(
        "-o", "--output_prefix", help="Output file name prefix")
    parser.add_argument(
        "-v", "--vocab_file", help="File containing vocabulary for the model")

    args = parser.parse_args()

    # Read the image list
    with open(args.file_names, "r") as f:
        imglist = list(filter(lambda x: len(x) > 0, map(
            lambda x: os.path.splitext(x.strip())[0], f.readlines())))

    os.makedirs(os.path.dirname(args.output_prefix), exist_ok=True)

    # Read info file containing class names and number of regions
    info = np.load(os.path.join(args.input_folder, "info.npz"), allow_pickle=True)
    num_classes = len(info["classes"])
    num_images = len(imglist)
    min_regions, max_regions = info["args"].item().min_max_boxes.split(",")
    assert min_regions == max_regions
    num_regions = int(min_regions)

    objects_vocab = info["classes"] # class names

    img_attrs = np.lib.format.open_memmap(
        args.output_prefix + "-img_attr.npy", mode="w+",
        shape=(num_images, num_regions, num_classes, 2), dtype=np.float16)
    img_masks = np.lib.format.open_memmap(
        args.output_prefix + "-img_mask.npy", mode="w+",
        shape=(num_images, num_regions), dtype=np.uint8)

    img_masks[:] = 0
    convert_to_dccn(objects_vocab, img_attrs, args.vocab_file, imglist, args.input_folder)

    img_attrs.flush()
    img_masks.flush()
