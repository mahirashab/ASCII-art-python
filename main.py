from PIL import Image
from os import path
import numpy as np

file_path = path.dirname(__file__)
img_path = path.join(file_path, "img")

f = open("converted.txt", "w+")

code = "`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$qq"


def scale_image(image, WIDTH=100):
    (w, h) = image.size
    ratio = h/float(w)
    HEIGHT = int(ratio * WIDTH)

    new_image = image.resize((WIDTH, HEIGHT))
    return new_image


def scale(x, out_range=(0, 64)):
    f_min, f_max = 0, 255
    t_min, t_max = out_range[0], out_range[1]

    try:
        y = ((x - f_min) * ((t_max - t_min) / float(f_max - f_min))) + t_min
    except:
        y = 0

    return y


v_scale = np.vectorize(scale)


def convert_to_text(matrix):
    line = ''
    for x in range(len(matrix)):
        for y in range(len(matrix[x])):
            # f.write(code[int(matrix[x][y])])
            # print(matrix[x][y])
            line += code[int(matrix[x][y])]

        # f.write("\r\n")
        print(line)


def process_img(img):
    pixel_matrix = np.asarray(img)
    print("Successfully constructed pixel matrix!")

    pixel_matrix = v_scale(pixel_matrix)
    print("Successfully set char index!!")

    print(pixel_matrix.shape)

    convert_to_text(pixel_matrix)
    # im = Image.fromarray(pixel_matrix)
    # im.save(path.join(img_path, "img.png"))


def main():
    try:
        image = Image.open(path.join(img_path, "test.jpg")).convert("L")
    except FileNotFoundError:
        print("Couldn't find test.*** image to process.")
    else:
        print("Loaded image successfully!!!")
        print("Image size :-", image.size)

        resized_image = scale_image(image)
        print("Resized image successfully!!!")
        print("Image size :-", resized_image.size)

        # resized_image.save(path.join(img_path, "img.png"))

        process_img(resized_image)


if __name__ == "__main__":
    main()
