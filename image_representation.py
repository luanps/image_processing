import sys
import cv2
import pdb
import numpy as np
import matplotlib.pyplot as plt


def random_coords(shape, size):
    x = np.random.randint(shape[1] - size)
    y = np.random.randint(shape[0] - size)
    p1 = (x,y)
    p2 = (x+size,y+size)
    return (p1, p2)


def draw_square(img, coords):
    p1, p2 = coords
    img = cv2.rectangle(img, p1, p2,(0,127,127),2)
    return img


def crop_square(img, coord):
    p1, p2 = coord
    return img[p1[1]:p2[1]+1,p1[0]:p2[0]+1,:]


def draw_pixel_values(img, size):
    fig = plt.figure(figsize=(size, size))
    ax = fig.add_subplot(111)
    ax.imshow(img, cmap='gray')
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            val = round(img[x][y], 2) if img[x][y] != 0 else 0
            ax.annotate(str(val), xy=(y, x),
                        horizontalalignment='center',
                        verticalalignment='center',
                        color='white' if img[x][y] < thresh else 'black')
    plt.savefig('pixels.png')


def split_channels(img):
    blue = img.copy()
    blue[:,:,1] = 0
    blue[:,:,2] = 0

    green = img.copy()
    green[:,:,0] = 0
    green[:,:,2] = 0

    red = img.copy()
    red[:,:,0] = 0
    red[:,:,1] = 0
    splitted_channels = cv2.hconcat([blue,green,red])
    cv2.imwrite('color_channels.png', splitted_channels)


if __name__ == '__main__':

    
    img = cv2.imread(sys.argv[1])
    square_size = 100
    coords = random_coords(img.shape, square_size)
    img_with_square = draw_square(img.copy(), coords)
    cv2.imwrite('img_square.png',img_with_square)

    square = crop_square(img, coords) 

    #reduce square to 10% for better pixel visualization
    resize_ratio = .1
    y,x = square.shape[:2]
    square_resized = cv2.resize(square,(int(y*resize_ratio),int(x*resize_ratio)))

    #convert to grayscale and draw pixel values
    gray_square = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY)
    draw_pixel_values(gray_square, int(square_size*resize_ratio))

    split_channels(img)
    #b,g,r = cv2.split(gray_square)
