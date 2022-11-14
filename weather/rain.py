import numpy as np
import cv2
import os

def create_noise(img, value=10):
    value *= 0.01
    noise = np.random.uniform(0, 256, img.shape[0:2])
    noise[np.where(noise<(256-value))] = 0

    kernel = np.array([[0, 0.1, 0], [0.1, 8, 0.1], [0, 0.1, 0]])
    noise = cv2.filter2D(noise, -1, kernel)
    return noise

def create_rain(noise, length=10, angle=0, w=1):
    dig = np.eye(length)
    trans = cv2.getRotationMatrix2D((length/2, length/2), angle-45, 1-length/100.0)

    kernel = cv2.warpAffine(dig, trans, (length, length))
    kernel = cv2.GaussianBlur(kernel, (w,w), 0)
    blurred = cv2.filter2D(noise, -1, kernel)

    cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
    blurred - np.array(blurred, dtype = np.uint8)

    return blurred

def add_rain(img, rain, alpha=0.9, beta=0.8):
    rain = np.expand_dims(rain, 2)
    rain_effect = np.concatenate((img, rain), axis=2)

    rainny = img.copy()
    rain = np.array(rain, dtype=np.float32)
    for i in range(3):
        rainny[:,:,i] = alpha * rainny[:,:,i]*(255-rain[:,:,0])/255.0 + beta*rain[:,:,0]

    return rainny

def get_rain(img):
    img = cv2.imread(img)
    # print(img.shape)
    value = np.random.randint(50, 5000)
    value = 4000
    l = np.random.randint(10, 60)
    l = 60
    angle = np.random.randint(-45, 45)
    angle = -30
    # w = np.random.randint(1, 3)
    w = 1
    alpha = 1-np.random.random()*0.15
    beta = 1-np.random.random()*0.25

    noise = create_noise(img, value)
    blur = create_rain(noise, l, angle, w)
    rain = add_rain(img, blur, alpha, beta)

    return rain

if __name__ == "__main__":
    f = "./KITTI_10/image_left/"
    frain = "./KITTI_10/"
    g = os.walk(f)

    for path, dir_l, fl in g:
        i = 0
        for ff in fl:
            print(ff)
            rain = get_rain(f+ff)
            cv2.imwrite(frain+ff, rain)
            if i==0:
                exit()