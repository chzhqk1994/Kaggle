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


def rle_decode(mask_rle, shape=(768, 768)): # mask_rle : 인코딩 된 에어버스 마스크 정보
    '''
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of array to return
        Returns numpy array, 1 - mask, 0 - background
    '''

    # split() >> 파라미터가 없으면 공백을 기준으로 쪼개어 리스트에 저장
    # 리스트로 정리 되지 않은 mask_rle 을 리스트로 정리    365871 1 366638 3 367405 6  >>  s ['365871', '1', '366638', '3', '367405'
    s = mask_rle.split()
    # s 는 리스트로 정리 된 에어버스 마스크 정보

    # s[0:][::2] >> 0번째 요소부터 건너뛰어 선택 ex 0 2 4 6 ...   , s[1:][::2] >> 1번째 요소부터 건너뛰어 선택 ex 1 3 5 7 ...
    # 선택된 항목을 int 형 numpy array 로 starts, lengths 변수에 저장
    starts, lengths = [np.array(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    # print('starts : ', starts)
    # print('lengths : ', lengths)

    # 여기서 1을 빼는 이유는 뭘까
    starts -= 1
    ends = starts + lengths
    # print('ends : ', ends)

    # 여기서 shape[0], shape[1] 는 이 함수의 파라미터로 지정되어 있는 shape=(768, 768)
    # 768 * 768 길이의 0행렬을 만듦
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    print('img', img)

    # 1차원 형태의 리스트에 인코드 된 정보를 디코드하여 덮어씌우는 과정이다.
    # zip 함수는 동일한 개수로 이루어진 자료형을 묶어주는 역할을 하는 내장함수
    # x= [1, 2, 3], y= [4, 5, 6]   >>   zip(z, y)   >>   (1, 4), (2, 5), (3, 6)
    # starts 는 기준점, ends 는 기준점으로 부터의 마스크 길이를 나타낸다.  ex) (61380, 16) 의 의미는 61380번째 픽셀로 부터 16개 픽셀까지 마스크한다는 의미
    # 지정된 길이만큼 0에서 1로 바꾸어 마스킹 한다.
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    print('processed img', img)

    # 1차원 형태인 디코드된 img 리스트를 shape 모양 (768, 768) 형으로 reshape 한 후 전치 시킨다.
    # .T >> Transpose 리스트를 전치한다 ( 뒤집는다 )
    return img.reshape(shape).T


masks = pd.read_csv("/Users/song/Documents/input/train_ship_segmentations.csv")
# print(masks.head())



ImageId = '0005d01c8.jpg'

img = imread('/Users/song/Documents/input/train/' + ImageId)

# loc : 라벨 값 기반의 2차원 인덱싱
# df.loc[행 인덱스, 열 인덱스] 와 같은 형태로 사용한다.   [행 인덱스, 열 인덱스] 에 해당하는 값을 구한다
# 즉 여기선 masks.loc['ImageId', 'EncodedPixels'].tolist() 가 된다.  'ImageId' 에는 미리 지정해둔 파일 이름이 들어간다
# 중복되는 행이 있다면 (같은 파일명을 가진 행이 있다면) 여러 대의 배가 있는 것 이다.
img_masks = masks.loc[masks['ImageId'] == ImageId, 'EncodedPixels'].tolist()
print(masks)
print(img_masks)

# Take the individual ship masks and create a single mask array for all ships
# 768 x 768 사이즈의 0행렬을 만든다.
all_masks = np.zeros((768, 768))
for mask in img_masks:
    all_masks += rle_decode(mask)
    print(all_masks)

fig, axarr = plt.subplots(1, 3, figsize=(15, 40))
axarr[0].axis('on')  # 프레임 격자의 표시 여부 결정 on / off
axarr[1].axis('on')
axarr[2].axis('on')
axarr[0].imshow(img)
axarr[1].imshow(all_masks)

# 이미지와 마스크를 동시에 띄워 오버랩 시킴  >>  이런것도 가능하구나
axarr[2].imshow(img)
axarr[2].imshow(all_masks, alpha=0.4)
plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()
