import cv2
import numpy as np
from os import path

file_path = path.dirname(__file__)
img_path = path.join(file_path, "img")


code_for_dark = "`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
code_for_white = '$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/\\|()1{}[]?-_+~i!lI;:,"^`'

char_size = 5


def scale_image(img, width=200):
    #     INTER_NEAREST - a nearest-neighbor interpolation
    #     INTER_LINEAR - a bilinear interpolation (used by default)
    #     INTER_AREA - resampling using pixel area relation. It may be a preferred method for image     decimation, as it gives moire’-free results. But when the image is zoomed, it is similar to   the INTER_NEAREST method.
    #     INTER_CUBIC - a bicubic interpolation over 4x4 pixel neighborhood
    #     INTER_LANCZOS4 - a Lanczos interpolation over 8x8 pixel neighborhood

    (w, h) = img.shape
    aspect_ratio = w / h
    height = int(width * aspect_ratio)

    return cv2.resize(img, dsize=(width, height), interpolation=cv2.INTER_CUBIC)


def scale(x, out_range=(0, 64)):
    f_min, f_max = 0, 255
    t_min, t_max = out_range[0], out_range[1]

    y = ((x - f_min) * ((t_max - t_min) / float(f_max - f_min))) + t_min

    return int(y)


v_scale = np.vectorize(scale)


def text_matrix(x, string):
    return string[x]


v_text_matrix = np.vectorize(text_matrix)


# FONT_HERSHEY_SIMPLEX = 0
# FONT_HERSHEY_PLAIN = 1
# FONT_HERSHEY_DUPLEX = 2
# FONT_HERSHEY_COMPLEX = 3
# FONT_HERSHEY_TRIPLEX = 4
# FONT_HERSHEY_COMPLEX_SMALL = 5
# FONT_HERSHEY_SCRIPT_SIMPLEX = 6
# FONT_HERSHEY_SCRIPT_COMPLEX = 7

# the image on which to draw
# the text to be written
# coordinates of the text start point
# font to be used
# font size
# text color
# text thickness
# the type of line used

font = cv2.FONT_HERSHEY_DUPLEX
img_np = np.ones(shape=[111, 111])


def show_result(matrix, w, h):
    matrix = matrix.T
    blank_image = np.zeros(shape=[w*char_size, h*char_size])
    blank_image = cv2.resize(img_np, (w*char_size + 10, h*char_size + 10))

    for i in range(0, w):
        for j in range(0, h):
            blank_image = cv2.putText(
                blank_image, matrix[i][j], (i*char_size, j*char_size), font, 0.2, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.imshow('Result', blank_image)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_img(img):
    pixel_matrix = np.asarray(img)
    print("Successfully constructed pixel matrix!")
    print("----------------------------")
    print("----------------------------")

    index_matrix = v_scale(pixel_matrix)
    print("Successfully constructed index_matrix!!")
    print("Matrix size :-", index_matrix.shape)
    print("----------------------------")

    char_matrix = v_text_matrix(index_matrix, code_for_white)
    print("Successfully constructed char_matrix!!")
    print("Matrix size :-", char_matrix.shape)
    print("----------------------------")

    (w, h) = char_matrix.shape

    print("Showing result!!")
    print("----------------------------")

    show_result(char_matrix, h, w)


def main():
    try:
        image = cv2.imread('./img/test.jpg')
    except FileNotFoundError:
        print("Couldn't find test.*** image to process.")
    else:
        print("Loaded image successfully!!!")
        print("Image size :-", image.size)
        print("----------------------------")

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        scaled_img = scale_image(gray_image)

        process_img(scaled_img)


if __name__ == "__main__":
    main()
