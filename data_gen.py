"""
Скрипт читает картинку и нарезает из нее датесет монеток, если монета слишком близко к краю изображения он ее проигнорирутет!!
Желательно давать на вход фото сразу с большим колличеством монет
Каждая монету разворачиватет с шагом 20 градусов
"""

import cv2 as cv
import numpy as np
from os import listdir
from os.path import isfile, join
import os


class names():
    global img_save_num
    global img_directory


# Загружаем изображение
def get_image(image_name):
    img = cv.imread(image_name)
    return img


# Ищем все кружки и возвращаем их координаты
def get_points(image):
    img = cv.medianBlur(image, 5)
    gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    circles = cv.HoughCircles(gimg, cv.HOUGH_GRADIENT, 1, \
            160, param1=60, param2=40, minRadius=140, maxRadius=180)  # если плохо находит монеты менять эти параметры 
    circles = np.uint16(np.around(circles))
#    print(circles)
    for i in circles[0, :]:
        cv.circle(img, (i[0], i[1]), i[2], (40, 150, 40), 2)
    height, width, channels = img.shape    
    img = cv.resize(img, (int(width/3), int(height/3)))
    cv.imshow('Circles',img)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return circles


# Обрезаем кружочки и приводим к одному квадратному размеру
def get_images_by_point(image, circles):
    croped_images = []
    height, width, channels = image.shape
    for c in circles[0, :]:
        w = int(c[2] * 2 * 3)
        h = int(c[2] * 2 * 3)
        x = int(c[0] - w / 2)
        y = int(c[1] - h / 2)
        if ((y + h) > height) or ((x + w) > width) or ( y < 0) or ( x < 0):
            print('w=', w, 'h=', h, 'x=', x, 'y=', y)
            continue
            
        crop_img = image[y:y + h, x:x + w]
        print('w=', w, 'h=', h, 'x=', x, 'y=', y)
#        cv.imshow('cr', crop_img)
        crop_img = cv.resize(crop_img, (512, 512))
#        cv.imshow('cr', crop_img)
#        cv.waitKey(0)
#        cv.destroyAllWindows()
        croped_images.append(crop_img)
    return croped_images


# уродуем каждую монетку
def get_more_img(image, names):
    after_filter = []
    # вертим пикчу по 20 градусов за раз, так как получаемое изображение
    # в 2 раза больше то черных полос точно не будет 
    for angle in range(0, 360, 20):
        image_center = tuple(np.array(image.shape)/2)
#        print(image_center)
        rot_mat = cv.getRotationMatrix2D((image_center[0],image_center[1]), angle, 1)
        img = cv.warpAffine(image, rot_mat, (image.shape[0],image.shape[1]), flags=cv.INTER_LINEAR)  # собственно поворот
        # дальше издеваемся
        w = int(image_center[0])
        h = int(image_center[1])
        x = int(image_center[0] / 2)
        y = int(image_center[1] / 2)
        crop_center = img[y - 5:y + h + 5, x - 5:x + w + 5]  # по размеру кружка + 5 пикс с каждой стороны
        cv.imwrite(names.img_directory+str(names.img_save_num)+'.jpg', cv.resize(crop_center, (512, 512)))
        names.img_save_num = names.img_save_num + 1
        crop_right_up=img[y + 5:y + h + 10, x + 5:x + w + 10]  # двигаем вправо вверх
        cv.imwrite(names.img_directory+str(names.img_save_num)+'.jpg', cv.resize(crop_right_up, (512, 512)))
        names.img_save_num = names.img_save_num + 1
        #after_filter.append(cv.resize(crop_right_up, (512, 512)))
        crop_left_down=img[y - 10:y + h - 5, x - 10:x + w - 5]  # двигаем влево вниз
        cv.imwrite(names.img_directory+str(names.img_save_num)+'.jpg', cv.resize(crop_left_down, (512, 512)))
        names.img_save_num = names.img_save_num + 1
        #after_filter.append(cv.resize(crop_left_down, (512, 512)))
    return after_filter

def save_images_from_list(images, PATH):
    for i, img in enumerate(images):
        name = PATH + str(i) + ".jpg"
#        print(img)
        cv.imwrite(name, img)

if __name__ == "__main__":
    x = names()
    x.img_save_num = 1
    x.img_directory = '1r\\'
    imgs = os.listdir(x.img_directory)
    out = []
    for img_name in imgs:
        image = get_image(x.img_directory+img_name)
        circles=get_points(image)
        images=get_images_by_point(image, circles)
        for img in images:
            out.extend(get_more_img(img, x))
