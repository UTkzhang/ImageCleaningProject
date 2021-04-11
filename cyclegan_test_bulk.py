import os

os.chdir('./CycleGAN')

epochs = list(range(50, 55))

for epoch in epochs:
    os.system("python test.py --dataroot ../datasets/noisytext --name noisytext_cyclegan --model cycle_gan --preprocess none --results_dir ../results --num_test 500 --epoch %d" % epoch)
