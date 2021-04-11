import cv2
from glob import glob
import random
import os
import re
import shutil

# trainA
files = sorted(glob("./denoising-dirty-documents/train/*.png"))
total_trainA = len(files) * 8
if len(glob('./datasets/noisytext/trainA/*.png')) < total_trainA:
    for idx, file in enumerate(files):
        img_orig = cv2.imread(file)
        img_rot90 = cv2.rotate(img_orig, cv2.ROTATE_90_CLOCKWISE)
        img_rot180 = cv2.rotate(img_orig, cv2.ROTATE_180)
        img_rot270 = cv2.rotate(img_orig, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_flip = cv2.flip(img_orig, 1)
        img_flip_rot90 = cv2.rotate(img_flip, cv2.ROTATE_90_CLOCKWISE)
        img_flip_rot180 = cv2.rotate(img_flip, cv2.ROTATE_180)
        img_flip_rot270 = cv2.rotate(img_flip, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for orient, img in enumerate([img_orig, img_rot90, img_rot180, img_rot270, img_flip, img_flip_rot90, img_flip_rot180, img_flip_rot270]):
            new_file = './datasets/noisytext/trainA/img_%04d.png' % (idx * 8 + orient)
            cv2.imwrite(new_file, img)
            print(file + ' -> ' + new_file)

# trainB
files = sorted(glob("./denoising-dirty-documents/train_cleaned/*.png"))
total_trainB = len(files) * 8
if len(glob('./datasets/noisytext/trainB/*.png')) < total_trainB:
    for idx, file in enumerate(files):
        img_orig = cv2.imread(file)
        img_rot90 = cv2.rotate(img_orig, cv2.ROTATE_90_CLOCKWISE)
        img_rot180 = cv2.rotate(img_orig, cv2.ROTATE_180)
        img_rot270 = cv2.rotate(img_orig, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_flip = cv2.flip(img_orig, 1)
        img_flip_rot90 = cv2.rotate(img_flip, cv2.ROTATE_90_CLOCKWISE)
        img_flip_rot180 = cv2.rotate(img_flip, cv2.ROTATE_180)
        img_flip_rot270 = cv2.rotate(img_flip, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for orient, img in enumerate(
                [img_orig, img_rot90, img_rot180, img_rot270, img_flip, img_flip_rot90, img_flip_rot180, img_flip_rot270]):
            new_file = './datasets/noisytext/trainB/img_%04d.png' % (idx * 8 + orient)
            cv2.imwrite(new_file, img)
            print(file + ' -> ' + new_file)


# Noisy Office dataset

files = sorted(glob("./NoisyOffice/SimulatedNoisyOffice/simulated_noisy_images_grayscale/*.png"))
total_trainA += len(files) * 8
total_trainB += len(files) * 8
curr_trainA = len(glob('./datasets/noisytext/trainA/*.png'))
if curr_trainA < total_trainA:
    for idx, file in enumerate(files, curr_trainA // 8):
        basename = os.path.basename(file)
        clean_file = './NoisyOffice/SimulatedNoisyOffice/clean_images_grayscale/' + re.sub(r'Noise[a-z]', 'Clean', basename)

        # trainA
        img_orig = cv2.imread(file)
        img_rot90 = cv2.rotate(img_orig, cv2.ROTATE_90_CLOCKWISE)
        img_rot180 = cv2.rotate(img_orig, cv2.ROTATE_180)
        img_rot270 = cv2.rotate(img_orig, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_flip = cv2.flip(img_orig, 1)
        img_flip_rot90 = cv2.rotate(img_flip, cv2.ROTATE_90_CLOCKWISE)
        img_flip_rot180 = cv2.rotate(img_flip, cv2.ROTATE_180)
        img_flip_rot270 = cv2.rotate(img_flip, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for orient, img in enumerate([img_orig, img_rot90, img_rot180, img_rot270, img_flip, img_flip_rot90, img_flip_rot180, img_flip_rot270]):
            new_file = './datasets/noisytext/trainA/img_%04d.png' % (idx * 8 + orient)
            cv2.imwrite(new_file, img)
            print(file + ' -> ' + new_file)

        # trainB
        img_orig = cv2.imread(clean_file)
        img_rot90 = cv2.rotate(img_orig, cv2.ROTATE_90_CLOCKWISE)
        img_rot180 = cv2.rotate(img_orig, cv2.ROTATE_180)
        img_rot270 = cv2.rotate(img_orig, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_flip = cv2.flip(img_orig, 1)
        img_flip_rot90 = cv2.rotate(img_flip, cv2.ROTATE_90_CLOCKWISE)
        img_flip_rot180 = cv2.rotate(img_flip, cv2.ROTATE_180)
        img_flip_rot270 = cv2.rotate(img_flip, cv2.ROTATE_90_COUNTERCLOCKWISE)

        for orient, img in enumerate([img_orig, img_rot90, img_rot180, img_rot270, img_flip, img_flip_rot90, img_flip_rot180, img_flip_rot270]):
            new_file = './datasets/noisytext/trainB/img_%04d.png' % (idx * 8 + orient)
            cv2.imwrite(new_file, img)
            print(file + ' -> ' + new_file)

# Random sample for test dataset
random.seed(69)
test_perc = 0.15
test_quant = int(total_trainA // 8 * test_perc)
test_sample = sorted(random.sample(list(range(total_trainA // 8)), test_quant))

if len(glob('./datasets/noisytext/testA/*.png')) < test_quant:
    for idx in test_sample:
        for i in range(8):
            shutil.move("./datasets/noisytext/trainA/img_%04d.png" % (idx*8+i), "./datasets/noisytext/testA/img_%04d.png" % (idx*8+i))
            shutil.move("./datasets/noisytext/trainB/img_%04d.png" % (idx*8+i), "./datasets/noisytext/testB/img_%04d.png" % (idx*8+i))