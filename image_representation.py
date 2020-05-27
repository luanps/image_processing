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
    cv2.imwrite('img_square.png',img)


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


def split_channels(img, offset):
    rows, cols = img.shape[:2]
    img = cv2.resize(img,(int(cols*.3),int(rows*.3)))
    rows, cols = img.shape[:2]

    #split colors
    blue = img.copy()
    blue[:,:,1] = 0
    blue[:,:,2] = 0

    green = img.copy()
    green[:,:,0] = 0
    green[:,:,2] = 0

    red = img.copy()
    red[:,:,0] = 0
    red[:,:,1] = 0

    #create a blank background
    bgd = 255 * np.ones((rows+offset*2,cols+offset*2,3), np.uint8)

    #plot the color channels
    bgd[0:rows, 0:cols] = red
    bgd[offset:rows+offset, offset:cols+offset] = green
    bgd[offset*2:rows+offset*2, offset*2:cols+offset*2] = blue
    cv2.imwrite('channel_colors.png', bgd)

    # channel intensities
    b,g,r = cv2.split(img)
    channel_intensities = cv2.vconcat([b,g,r])
    cv2.imwrite('channel_intensities.png', channel_intensities)


def edge_detection(img):
    #convolution matrix
    edge_convolution = np.array(([-1,-1,-1],[-1,8,-1],[-1,-1,-1]),dtype='int')

    #convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #apply filter
    transformed_img = cv2.filter2D(gray_img,-1, edge_convolution)
    cv2.imwrite('transformed_img.png',transformed_img)
    cv2.imwrite('grayscale_img.png',gray_img)


if __name__ == '__main__':

    img = cv2.imread(sys.argv[1])
    square_size = 100
    offset = 50


    #select a random squared region
    coords = random_coords(img.shape, square_size)
    draw_square(img.copy(), coords)
    square = crop_square(img, coords) 

    #reduce square to 10% for better pixel visualization
    resize_ratio = .1
    rows,cols = square.shape[:2]
    square_resized = cv2.resize(square,(int(rows*resize_ratio),int(cols*resize_ratio)))

    #convert square to grayscale and draw pixel values on it
    gray_square = cv2.cvtColor(square_resized, cv2.COLOR_BGR2GRAY)
    draw_pixel_values(gray_square, int(square_size*resize_ratio))

    # split RGB channels
    split_channels(img, offset)

    #edge detector
    edge_detection(img)
