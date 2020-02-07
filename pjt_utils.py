import xlsxwriter
import xml.etree.ElementTree as ET
import xlrd
import csv
import os
from PIL import Image, ImageEnhance
from tqdm import tqdm # optional: to see progress
import random
import matplotlib.pyplot as plt
import urllib.request
import time
import subprocess
import shutil
import cv2
import sys

def PKLot_xml_to_yolo_txt(data_path):
    # Read full directory and file name in the folder
    for path, dirs, files in os.walk(data_path):
        for file in tqdm(files):
            if os.path.splitext(file)[1].lower() == '.xml':  # filtering only for .xml files
                full_path = os.path.join(path, file)  # path + file name
                # print(full_path) # prints path + file name
                # print(path) # prints path only
                # print(file) # prints file name only
                name = os.path.splitext(file)[0]  # remove file format, get name only
                # print(name)

                # Get original image size in pixel
                jpg = path + '/' + name + '.jpg'
                img = Image.open(jpg)
                width, height = img.size

                # Create new .xlsx files
                xlsx_path_name = path + '/' + name + '.xlsx'  # names of the .xlsx files to be created
                workbook = xlsxwriter.Workbook(xlsx_path_name)  # create file
                worksheet = workbook.add_worksheet()  # create worksheet

                i = -1
                j = -1
                k = -1

                # Parsing data from .xml files and write on .xlsx files
                tree = ET.parse(full_path)  # read .xml files
                root = tree.getroot()

                """for space in root.iter('occupied'): # find all child 'space' and iterate
                    car = space.attrib['occupied'] # get the value of the attribute 'occupied'
                    car_int = int(car)
                    i += 1
                    worksheet.write(i, 0, '%.f' % car_int) # write on the first colum"""
                # This code occurs KeyError:'occupied' due to lack of attribute 'occupied' in some .xml files.
                # alternative: include 'if' under 'for' to ignore the files which do not include attribute 'occupied' and continue to iterate.

                for space in root:
                    if 'occupied' in space.attrib:  # check, if there is attribute 'occupied'. This code added because sometimes this attribute does not exist and it causes error.
                        car = space.attrib['occupied']  # if there is, get the value of the attribute 'occupied'
                        car_int = int(car) - 1
                        i += 1
                        worksheet.write(i, 0, '%.f' % car_int)  # write on the first column
                    else:
                        i += 1
                        worksheet.write(i, 0, 'false')  # write 'false', if there is no attribute 'occupied'

                for center in root.iter('center'):  # find all child 'center' and iterate
                    x_coor = center.attrib['x']  # get the value of attribute 'x'
                    y_coor = center.attrib['y']  # get the value of attribute 'y'
                    x_int = int(x_coor)  # change data format from string to integer
                    y_int = int(y_coor)  # change data format from string to integer
                    x_conv = x_int / width  # convert the value to yolo format. Devide by the width of the image
                    y_conv = y_int / height  # convert the value to yolo format. Devide by the height of the image
                    j += 1
                    worksheet.write(j, 1, '%.6f' % x_conv)  # write on the second column
                    worksheet.write(j, 2, '%.6f' % y_conv)  # write on the third column

                for size in root.iter('size'):  # find all child 'size'
                    w_size = size.attrib['w']  # get the value of attribute 'w'
                    h_size = size.attrib['h']  # get the value of attribute 'h'
                    w_int = int(w_size)  # change data format from string to integer
                    h_int = int(h_size)  # change data format from string to integer
                    w_conv = w_int / width  # convert the value to yolo format. Devide by the width of the image
                    h_conv = h_int / height  # convert the value to yolo format. Devide by the height of the image
                    k += 1
                    worksheet.write(k, 3, '%.6f' % w_conv)  # write on the fourth column
                    worksheet.write(k, 4, '%.6f' % h_conv)  # write on the fifth column

                workbook.close()

                # Create text files
                txt_path_name = path + '/' + name + '.txt'  # names of the .txt files to be created
                with open(txt_path_name, 'w') as file:
                    wr = csv.writer(file, delimiter=" ")

                    xlsx = xlrd.open_workbook(xlsx_path_name)  # read data from .xlsx files
                    sheet = xlsx.sheet_by_index(0)
                    for rownum in range(sheet.nrows):
                        value = sheet.cell_value(rownum, 0)  # get the values of the first column as string
                        # print(type(value))
                        # if value != 'false':
                        if value == '0':
                            # print(value)
                            wr.writerow(sheet.row_values(rownum))  # write .txt if the value of the first column is not 'false'

#PKLot_xml_to_yolo_txt(sys.argv[1])


def crop_image(data_path, save_to):
    for path, dirs, files in os.walk(data_path):
        for file in tqdm(files):
            if os.path.splitext(file)[1].lower() == '.jpg':  # filtering only for .jpg files
                full_path = os.path.join(path, file)  # path + file name
                name = os.path.splitext(file)[0]

                # Get original image size in pixel: this is not mandatory
                img = Image.open(full_path)
                width, height = img.size
                # print(width, height)

                # Crop and save the cropped image
                # img.crop((left, top, right, bottom)) # pixel number counts from top-left to bottom-right (top-left: 0,0)
                img = img.crop((0, 0, width, height - 16)).save(save_to + '/' + name + '_crop.jpg')
                # crop from the bottom but not top because the coordinate is measured from the top-left

#crop_image(sys.argv[1], sys.argv[2])


def remove_xml_xlsx(data_path):
    i = 0
    for path, dirs, files in os.walk(data_path):
        for file in tqdm(files):
            # check, if the file format is .xml or .xlsx
            if os.path.splitext(file)[1].lower() == '.xml' or os.path.splitext(file)[1].lower() == '.xlsx':
                file_to_remove = os.path.join(path, file)  # path + file name to be deleted
                os.remove(file_to_remove)  # delete .xlsx and .xml files

#remove_xml_xlsx(sys.argv[1])


def split_dataset(data_path):
    # source: https://github.com/spmallick/learnopencv.git

    f_val = open("test.txt", 'w')
    f_train = open("train.txt", 'w')

    path, dirs, files = next(os.walk(data_path))
    data_size = len(files)

    ind = 0
    data_test_size = int(0.1 * data_size)
    test_array = random.sample(range(data_size), k=data_test_size)

    for f in os.listdir(data_path):
        if (f.split(".")[1] == "jpg"):
            ind += 1

            if ind in test_array:
                f_val.write(data_path + '/' + f + '\n')
            else:
                f_train.write(data_path + '/' + f + '\n')

#split_dataset(sys.argv[1])


def resize_image(data_path, save_to):
    for path, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':  # filtering only for certain files
                # print(file)
                full_path = os.path.join(path, file)  # path + file name
                name = os.path.splitext(file)[0]  # remove file format, get name only
                # print(name)

                basewidth = 1024
                img = Image.open(full_path)
                wpercent = (basewidth / float(img.size[0]))
                hsize = int((float(img.size[1]) * float(wpercent)))
                img = img.resize((basewidth, hsize), Image.ANTIALIAS)
                img.save(save_to + '/' + name + '_re.jpg')

#resize_image(sys.argv[1], sys.argv[2])


def rename_file(data_path):
    i = 0
    for path, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':
                name = os.path.splitext(file)[0]
                i += 1

                old_jpg = path + '/' + name + '.jpg'
                old_txt = path + '/' + name + '.txt'
                new_jpg = path + '/' + '30-1c-' + str(i) + "-re.jpg"
                new_txt = path + '/' + '30-1c-' + str(i) + "-re.txt"

                os.rename(old_jpg, new_jpg)
                os.rename(old_txt, new_txt)

                #full_path = os.path.join(path, file)
                #os.rename(full_path, path + '/' + '%d.jpg' % i)

#rename_file(sys.argv[1])


def plot_train_loss(data_path):
    # source: https://github.com/spmallick/learnopencv.git

    lines = []
    for line in open(data_path):
        if "avg" in line:
            lines.append(line)

    iterations = []
    avg_loss = []

    print('Retrieving data and plotting training loss graph...')
    for i in range(len(lines)):
        lineParts = lines[i].split(',')
        iterations.append(int(lineParts[0].split(':')[0]))
        avg_loss.append(float(lineParts[1].split()[0]))

    fig = plt.figure()
    for i in range(0, len(lines)):
        plt.plot(iterations[i:i + 2], avg_loss[i:i + 2], 'r.-')

    plt.xlabel('Batch Number')
    plt.ylabel('Avg Loss')
    fig.savefig('training_loss_plot.png', dpi=1000)

    print('Done! Plot saved as training_loss_plot.png')

#plot_train_loss(sys.argv[1])


def webcam_capture(url, save_to):
    i = 0
    num_image = 2 # put number of images want to capture
    cap_int = 30 # put time interval of capture (in second)

    for i in tqdm(range(num_image)):
        i += 1

        urllib.request.urlretrieve(url, save_to + '/cap_%d' % i)

        if i < num_image:
            time.sleep(cap_int)

#webcam_capture(sys.argv[1], sys.argv[2])


def image_aug(data_path, save_to):
    deg_rotate = -2
    cont = -2
    bright = 2
    color = 4
    sharp = 4

    for path, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':  # filtering only for .jpg files
                full_path = os.path.join(path, file)  # path + file name
                name = os.path.splitext(file)[0]  # remove file format, get name only

                # Sharpen
                im = Image.open(full_path)
                enhancer = ImageEnhance.Sharpness(im)
                enhanced_im = enhancer.enhance(sharp)
                enhanced_im.save(save_to + '/' + name + '_sharp_%d.jpg' % sharp)

                # Color
                im = Image.open(full_path)
                enhancer = ImageEnhance.Color(im)
                enhanced_im = enhancer.enhance(color)
                enhanced_im.save(save_to + '/' + name + '_color_%d.jpg' % color)

                # Brightness
                im = Image.open(full_path)
                enhancer = ImageEnhance.Brightness(im)
                enhanced_im = enhancer.enhance(bright)
                enhanced_im.save(save_to + '/' + name + '_bright_%d.jpg' % bright)

                # Contrast
                im = Image.open(full_path)
                enhancer = ImageEnhance.Contrast(im)
                enhanced_im = enhancer.enhance(cont)
                enhanced_im.save(save_to + '/' + name + '_contrast_%d.jpg' % cont)

                # Rotate
                image_obj = Image.open(full_path)
                rotated_image = image_obj.rotate(deg_rotate)
                rotated_image.save(save_to + '/' + name + '_rotate_%d.jpg' % deg_rotate)

                # Flip left-right
                image_obj = Image.open(full_path)
                rotated_image = image_obj.transpose(Image.FLIP_LEFT_RIGHT)
                rotated_image.save(save_to + '/' + name + '_flip.jpg')

#image_aug(sys.argv[1], sys.argv[2])


def getDataFromOpenimages(path_aws):
    # source: https://github.com/spmallick/learnopencv.git

    '''
    Need to do before use:
    1. install aws
    2. put these files in the same folder:
     - class-descriptions-boxable.csv
     - train-annotations-bbox.csv
    '''

    runMode = "train"
    classes = ["Car"]

    with open('class-descriptions-boxable.csv', mode='r') as infile:
        reader = csv.reader(infile)
        dict_list = {rows[1]: rows[0] for rows in reader}

    subprocess.run(['rm', '-rf', 'JPEGImages'])
    subprocess.run([ 'mkdir', 'JPEGImages'])

    subprocess.run(['rm', '-rf', 'labels'])
    subprocess.run([ 'mkdir', 'labels'])

    for ind in range(0, len(classes)):

        # new_ind = 1  # added to make motocycle class index 1

        className = classes[ind]
        print("Class " + str(ind) + " : " + className)

        commandStr = "grep " + dict_list[className] + " " + runMode + "-annotations-bbox.csv"
        print(commandStr)
        class_annotations = subprocess.run(commandStr.split(), stdout=subprocess.PIPE).stdout.decode('utf-8')
        class_annotations = class_annotations.splitlines()

        totalNumOfAnnotations = len(class_annotations)
        print("Total number of annotations : " + str(totalNumOfAnnotations))

        cnt = 0
        for line in class_annotations[0:totalNumOfAnnotations]:
            cnt = cnt + 1
            print("annotation count : " + str(cnt))
            lineParts = line.split(',')
            subprocess.run([path_aws, 's3', '--no-sign-request', '--only-show-errors', 'cp', 's3://open-images-dataset/'+runMode+'/'+lineParts[0]+".jpg", 'JPEGImages/'+lineParts[0]+".jpg"])
            with open('labels/%s.txt'%(lineParts[0]),'a') as f:
                f.write(' '.join([str(ind), str((float(lineParts[5]) + float(lineParts[4])) / 2),
                                  str((float(lineParts[7]) + float(lineParts[6])) / 2),
                                  str(float(lineParts[5]) - float(lineParts[4])),
                                  str(float(lineParts[7]) - float(lineParts[6]))]) + '\n')

#getDataFromOpenimages(sys.argv[1])


def seperate_empty_occupied(data_path, emp_save, occ_save):
    i = 0
    j = 0

    # Read full directory and file name in the folder
    for path, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':  # filtering only for .jpg files
                full_path = os.path.join(path, file)  # path + file name

                if 'Empty' in full_path:
                    i += 1
                    shutil.copy(full_path, emp_save)

                if 'Occupied' in full_path:
                    j += 1
                    shutil.copy(full_path, occ_save)

    print('no. empty images: ', i)
    print('no. occupied images: ', j)

#seperate_empty_occupied(sys.argv[1], sys.argv[2], sys.argv[3])


def random_subset(data_path, copy_to, num_subset):
    i = 0
    jpeg = []
    for path, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':  # filtering only for .jpg files
                full_path = os.path.join(path, file)  # path + file name.jpg
                jpeg.append(full_path)  # create list which consist of full_path of all .jpg files

    # print(jpeg) # prints list
    # print(len(jpeg)) # prints length of list
    selected_jpgs = random.sample(jpeg, num_subset)
    # print(selected_jpgs)
    for selected_jpg in selected_jpgs:
        shutil.copy(selected_jpg, copy_to)  # copy selected 50000 .jpg files to designated directory

        # copying corresponding .txt files
        name = os.path.splitext(selected_jpg)[0] # remove file format, get name only
        selected_txt = name + '.txt' # # names of the .txt files to be copied along with copied .jpg files
        shutil.copy(selected_txt, copy_to) # copy selected 500 .txt files to designated directory

#random_subset(sys.argv[1], sys.argv[2], sys.argv[3])


def SegmentedYoloTxt(data_path):
    i = 0

    # Read full directory and file name in the folder
    for path, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':
                full_path = os.path.join(path, file)  # path + file name

                i += 1

                name = os.path.splitext(file)[0]  # remove file format, get name only
                txt_name = path + '/' + name + '.txt'

                file = open(txt_name, 'w')
                file.write('0 0.500000 0.500000 1.000000 1.000000')
                file.close()

#SegmentedYoloTxt(sys.argv[1])


def check_image_size(data_path, w_tobe, h_tobe):
    i = 0

    # Read full directory and file name in the folder
    for path, dirs, files in os.walk(data_path):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.jpg':  # filtering only for .jpg files
                full_path = os.path.join(path, file)  # path + file name

                # Get original image size in pixel
                img = Image.open(full_path)
                width, height = img.size
                # print(width, height)

                # Check, if there exists different image size
                if width != w_tobe or height != h_tobe:
                    i += 1
                    print(i, width, height, full_path)

#check_image_size(sys.argv[1], sys.argv[2], sys.argv[3])


def check_yolo_annotation(data_path):
    # Reading an image in default mode
    image = cv2.imread(data_path)

    # Window name in which image is displayed
    window_name = 'Image'

    # Image size
    img = Image.open(data_path)
    width, height = img.size

    # Read coordinate from corresponding .txt
    name = os.path.splitext(data_path)[0]
    file = open(name + '.txt', 'r')
    lines = file.readlines()
    for line in lines:
        x = float(line.split(' ')[1])
        y = float(line.split(' ')[2])
        w = float(line.split(' ')[3])
        h = float(line.split(' ')[4])

        # Convert the coordinate value
        x1, y1 = int((x - w / 2) * width), int((y - h / 2) * height)
        x2, y2 = int((x + w / 3) * width), int((y + h / 2) * height)

        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (x1, y1)

        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (x2, y2)

        # Blue color in BGR
        color = (0, 255, 0)

        # Line thickness of 2 px
        thickness = 2

        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

    # Displaying the image
    cv2.imshow(window_name, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#check_yolo_nnotation(sys.argv[1])

