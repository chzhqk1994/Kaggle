import numpy as np
import pandas as pd  # Data processing
from skimage.data import imread  # imread >> 이미지 파일을 읽어들인다.
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

train = os.listdir("/Users/song/Documents/input/train")
print(len(train))

test = os.listdir("/Users/song/Documents/input/test")
print(len(test))

submission = pd.read_csv("/Users/song/Documents/input/sample_submission.csv")
# print(submission.head())


def rle_decode(mask_rle, shape=(768, 768)):
    '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
    '''

    s = mask_rle.split()
    starts, lengths = [np.array(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # print('starts : ', starts)
    # print('lengths : ', lengths)
    starts -= 1
    ends = starts + lengths
    # print('ends : ', ends)
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


masks = pd.read_csv("/Users/song/Documents/input/train_ship_segmentations.csv")
# print(masks.head())



ImageId = '0005d01c8.jpg'

img = imread('/Users/song/Documents/input/train/' + ImageId)

# loc : 라벨 값 기반의 2차원 인덱싱
# df.loc[행 인덱스, 열 인덱스] 와 같은 형태로 사용한다.   [행 인덱스, 열 인덱스] 에 해당하는 값을 구한다
# 즉 여기선 masks.loc['ImageId', 'EncodedPixels'].tolist() 가 된다.  'ImageId' 에는 미리 지정해둔 파일 이름이 들어간다
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
print(masks)
print(img_masks)

# Take the individual ship masks and create a single mask array for all ships
all_masks = np.zeros((768, 768))
for mask in img_masks:
    all_masks += rle_decode(mask)

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('off')
axarr[1].axis('off')
axarr[2].axis('off')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
# plt.show()