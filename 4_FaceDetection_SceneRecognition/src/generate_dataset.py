import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import cv2
import tqdm
import pickle

FACE_IM_PATH = "../dataset/fddb/originalPics/"
OUTPUT_PATH = "../dataset/curated/"
NON_FACE_IM_PATH = "../dataset/cifar-10/"

PROBLEM = 0
IM_COUNT = 0
TEST_IM_COUNT = 0
TRAIN_ARRAY = []
TEST_ARRAY = []
CROP_TEST_ARRAY = []


def get_bb_from_ellipse(im, major_axis_radius, minor_axis_radius, angle, center_x, center_y):
    
    '''
    Angle is always with respect to the major axis. If angle == 0 then
    major axis is parallel to the y axis (rows). Therefore, if angle > 180
    only then we need to worry about axis flip and all
    '''

    if angle > 180 : 
        PROBLEM += 1 #axis flipped cases

    # im  = cv2.ellipse(im, (int(center_x), int(center_y)), (int(minor_axis_radius), int(major_axis_radius)),
    #        angle, 0, 360, (0, 0, 255), 4)

    angle_rad = np.deg2rad(angle)
    b = major_axis_radius
    a = minor_axis_radius
    x = int(np.sqrt((a**2) * np.cos(angle_rad)**2 + (b**2) * np.sin(angle_rad)**2)) 
    y = int(np.sqrt((b**2) * np.cos(angle_rad)**2 + (a**2) * np.sin(angle_rad)**2)) 

    tlx = max(0, int(center_x - x))
    tly = max(0, int(center_y - y))
    brx = max(0, int(center_x + x))
    bry = max(0, int(center_y + y))


    #im = cv2.rectangle(im,(tlx, tly),(brx, bry),(0,255,0),2)
    # print(x,y)
    # print(tlx, tly)
    # print(brx, bry)
    # print()

    try:
        if tlx >= 0 and tly >=0 and brx >= 0 and bry >=0:
            face_im = im[tly: bry , tlx: brx]
        else:
            im  = cv2.ellipse(im, (int(center_x), int(center_y)), (int(minor_axis_radius), int(major_axis_radius)),
            angle, 0, 360, (0, 0, 255), 4)
            face_im = im
        #print(face_im.shape)
    except:
        face_im = im
        print("problemmmm")
        exit(0)

    face_im = cv2.resize(face_im, (24,24), interpolation = cv2.INTER_AREA)
    return face_im
    #plt.imsave(FFDB_FACES_IM_PATH+str(IM_COUNT)+".jpg", face_im)
    # implot = plt.imshow(face_im)
    # plt.scatter([center_x, x2, y1], [center_y, x1, y2])
    # plt.show()
    #print(major_axis_radius)
    #print(minor_axis_radius)
    #print(angle)
    #print(center_x)
    #print(center_y)


def get_image(path):

    im = cv2.imread(path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return im.astype('uint8')


def generate_face_images():

    global IM_COUNT
    global TEST_IM_COUNT
    global FACE_IM_PATH
    global OUTPUT_PATH
    global PROBLEM
    global IM_COUNT
    global TEST_IM_COUNT

    files = glob.glob("../dataset/fddb/FDDB-folds/*ellipseList.txt")

    for index, file in tqdm.tqdm(enumerate(sorted(files))):

        with open(file, 'r') as f:

            line = f.readline().strip()
            image_path = FACE_IM_PATH + line + ".jpg"

            while(line):
                
                im = get_image(image_path)
                face_count = int(f.readline().strip())
                face_anno = []

                for i in range(face_count):
                    
                    major_axis_radius, minor_axis_radius, angle, center_x, center_y = np.asarray(f.readline().strip().split(" "))[:-2].astype(np.float)
                    crop_im = [get_bb_from_ellipse(im, major_axis_radius, minor_axis_radius, angle, center_x, center_y), 1]

                    if index != len(files) - 1:
                        IM_COUNT += 1
                        TRAIN_ARRAY.append(crop_im)
                    
                    else:
                        CROP_TEST_ARRAY.append(crop_im)

                if index == len(files) - 1:
                    TEST_ARRAY.append([im, face_count])
                    TEST_IM_COUNT += 1

                #break
                line = f.readline().strip()
                image_path = FACE_IM_PATH + line + ".jpg"      
        #break

    with open(OUTPUT_PATH + "fddb_faces_train.pkl", "wb") as f1:
        print(np.asarray(TRAIN_ARRAY).shape, "trainn")
        pickle.dump(np.asarray(TRAIN_ARRAY), f1)
    with open(OUTPUT_PATH + "fddb_faces_test.pkl", "wb") as f2:
        print(np.asarray(TEST_ARRAY).shape, "test")
        pickle.dump(np.asarray(TEST_ARRAY), f2)
    with open(OUTPUT_PATH + "fddb_faces_test_cropped.pkl", "wb") as f3:
        print(np.asarray(CROP_TEST_ARRAY).shape, "test")
        pickle.dump(np.asarray(CROP_TEST_ARRAY), f3)

    print("\nProblematic Cases = ", PROBLEM)


def generate_non_face_images(train_size, test_size):

    all_ims = []
    i = 0
    for file in tqdm.tqdm(sorted(glob.glob(NON_FACE_IM_PATH + "*"))):
        im = cv2.cvtColor(cv2.imread(file), cv2.COLOR_RGB2GRAY)
        im = cv2.resize(im, (24,24), interpolation = cv2.INTER_AREA)
        all_ims.append([im, 0])

    all_ims = np.asarray(all_ims)
    print(all_ims.shape)

    train_ind = np.random.choice(all_ims.shape[0], size=train_size, replace=False)
    train_data = all_ims[train_ind]
    np.delete(all_ims, train_ind)

    test_data = all_ims[np.random.choice(all_ims.shape[0], size=test_size, replace=False)] 

    
    with open(OUTPUT_PATH + "cifar_nonfaces_train.pkl", "wb") as f:
        pickle.dump(train_data, f)

    with open(OUTPUT_PATH + "cifar_nonfaces_test.pkl", "wb") as f:
        pickle.dump(test_data, f)


def convert_to_pkl(path, filename):

    face = []
    for file in sorted(glob.glob(path+"face/*")):
        im = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (24,24))
        face.append([im,1])
    print(len(face))

    face = np.asarray(face)
    with open(filename+"faces.pkl", 'wb') as f:
        pickle.dump(face, f)

    nonface = []
    for file in sorted(glob.glob(path+"non-face/*")):
        im = cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY)
        im = cv2.resize(im, (24,24))
        nonface.append([im,0])
    print(len(nonface))
    
    nonface = np.asarray(nonface)
    with open(filename+"nonfaces.pkl", 'wb') as f:
        pickle.dump(nonface, f)

    #data = np.concatenate((face, nonface), axis=0)





def sanity_check(path):

    with open (path, "rb") as f:
        arr = pickle.load(f)

    print(arr.shape)
    plt.imshow(arr[200][0])
    plt.show()
    print("Label:", arr[20][1])


if  __name__ == '__main__':

    #generate_face_images()
    #generate_non_face_images(4000, 500)
    #sanity_check("../dataset/curated/data_train_faces.pkl")
    convert_to_pkl("../dataset/faces/face.train/train/", "../dataset/curated/data_train_")

