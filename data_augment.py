import cv2
from glob import glob

# trainA
files = sorted(glob("./denoising-dirty-documents/train/*.png"))
total = len(files)
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
        new_file = './datasets/kaggle/trainA/img_%04d.png' % (idx * 8 + orient)
        cv2.imwrite(new_file, img)
        print(file + ' -> ' + new_file)

# trainB
files = sorted(glob("./denoising-dirty-documents/train_cleaned/*.png"))
total = len(files)
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
        new_file = './datasets/kaggle/trainB/img_%04d.png' % (idx * 8 + orient)
        cv2.imwrite(new_file, img)
        print(file + ' -> ' + new_file)

# testA
files = sorted(glob("./denoising-dirty-documents/test/*.png"))
total = len(files)
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
        new_file = './datasets/kaggle/testA/img_%04d.png' % (idx * 8 + orient)
        cv2.imwrite(new_file, img)
        print(file + ' -> ' + new_file)
