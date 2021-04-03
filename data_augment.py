import cv2
from glob import glob

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

# testA
files = sorted(glob("./denoising-dirty-documents/test/*.png"))
total_testA = len(files)
if len(glob('./datasets/noisytext/testA/*.png')) < total_testA:
    for idx, file in enumerate(files):
        img_orig = cv2.imread(file)
        new_file = './datasets/noisytext/testA/img_%04d.png' % idx
        cv2.imwrite(new_file, img_orig)
        print(file + ' -> ' + new_file)


# Noisy Office dataset

# trainA
files = sorted(glob("./NoisyOffice/SimulatedNoisyOffice/simulated_noisy_images_grayscale/*.png"))
total_trainA += len(files) * 8
curr_trainA = len(glob('./datasets/noisytext/trainA/*.png'))
if curr_trainA < total_trainA:
    for idx, file in enumerate(files, curr_trainA):
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
files = sorted(glob("./NoisyOffice/SimulatedNoisyOffice/clean_images_grayscale/*.png"))
total_trainB += len(files) * 8
curr_trainB = len(glob('./datasets/noisytext/trainB/*.png'))
if curr_trainB < total_trainB:
    for idx, file in enumerate(files, curr_trainB):
        img_orig = cv2.imread(file)
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

# testA
files = sorted(glob("./NoisyOffice/RealNoisyOffice/real_noisy_images_grayscale/*.png"))
total_testA += len(files)
curr_testA = len(glob('./datasets/noisytext/testA/*.png'))
if curr_testA < total_testA:
    for idx, file in enumerate(files, curr_testA):
        img_orig = cv2.imread(file)
        new_file = './datasets/noisytext/testA/img_%04d.png' % idx
        cv2.imwrite(new_file, img_orig)
        print(file + ' -> ' + new_file)
