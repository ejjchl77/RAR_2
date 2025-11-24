#!/usr/bin/env python
# coding: utf-8

'''
Test action recognition on
(1) a video, (2) a folder of images, (3) or web camera.

Input:
    model: model/trained_classifier.pickle

Output:
    result video:    output/${video_name}/video.avi
    result skeleton: output/${video_name}/skeleton_res/XXXXX.txt
    visualization by cv2.imshow() in img_displayer
'''

'''
Example of usage:

(1) Test on video file:
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type video \
    --data_path data_test/exercise.avi \
    --output_folder output
    
(2) Test on a folder of images:
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type folder \
    --data_path data_test/apple/ \
    --output_folder output

(3) Test on web camera:
python src/s5_test.py \
    --model_path model/trained_classifier.pickle \
    --data_type webcam \
    --data_path 0 \
    --output_folder output
    
'''


import numpy as np
import cv2
import argparse
import datetime
if True:  # Include project path
    import sys
    import os
    ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
    CURR_PATH = os.path.dirname(os.path.abspath(__file__))+"/"
    sys.path.append(ROOT)

    import utils.lib_images_io as lib_images_io
    import utils.lib_plot as lib_plot
    import utils.lib_commons as lib_commons
    from utils.lib_openpose import SkeletonDetector
    from utils.lib_tracker import Tracker
    from utils.lib_classifier import ClassifierOnlineTest
    from utils.lib_classifier import *  # Import all sklearn related libraries
    
    #from utils.EJ_dbupload import dp_upload

def par(path):  # Pre-Append ROOT to the path if it's not absolute
    return ROOT + path if (path and path[0] != "/") else path


# -- Command-line input


def get_command_line_arguments():

    def parse_args():
        parser = argparse.ArgumentParser(
            description="Test action recognition on \n"
            "(1) a video, (2) a folder of images, (3) or web camera.")
        parser.add_argument("-m", "--model_path", required=False,
                            default='model/trained_classifier.pickle')
        parser.add_argument("-t", "--data_type", required=False, default='webcam',
                            choices=["video", "folder", "webcam"])
        parser.add_argument("-p", "--data_path", required=False, default="",
                            help="path to a video file, or images folder, or webcam. \n"
                            "For video and folder, the path should be "
                            "absolute or relative to this project's root. "
                            "For webcam, either input an index or device name. ")
        parser.add_argument("-o", "--output_folder", required=False, default='output/',
                            help="Which folder to save result to.")
        
        args = parser.parse_args()
        return args
    args = parse_args()
    if args.data_type != "webcam" and args.data_path and args.data_path[0] != "/":
        # If the path is not absolute, then its relative to the ROOT.
        args.data_path = ROOT + args.data_path
    return args


def get_dst_folder_name(src_data_type, src_data_path):
    ''' Compute a output folder name based on data_type and data_path.
        The final output of this script looks like this:
            DST_FOLDER/folder_name/vidoe.avi
            DST_FOLDER/folder_name/skeletons/XXXXX.txt
    '''

    assert(src_data_type in ["video", "folder", "webcam"])

    if src_data_type == "video":  # /root/data/video.avi --> video
        start_time = datetime.datetime.now()
        folder_name = os.path.basename(src_data_path).split(".")[-2]

    elif src_data_type == "folder":  # /root/data/video/ --> video
        start_time = datetime.datetime.now()
        folder_name = src_data_path.rstrip("/").split("/")[-1]

    elif src_data_type == "webcam":
        # month-day-hour-minute-seconds, e.g.: 02-26-15-51-12
        start_time, folder_name = lib_commons.get_time()

    return start_time, folder_name


args = get_command_line_arguments()

SRC_DATA_TYPE = args.data_type
SRC_DATA_PATH = args.data_path
SRC_MODEL_PATH = args.model_path

Start_time, DST_FOLDER_NAME = get_dst_folder_name(SRC_DATA_TYPE, SRC_DATA_PATH)

# -- Settings

cfg_all = lib_commons.read_yaml(ROOT + "config/config.yaml")
cfg = cfg_all["s5_test.py"]

CLASSES = np.array(cfg_all["classes"])
SKELETON_FILENAME_FORMAT = cfg_all["skeleton_filename_format"]

# Action recognition: number of frames used to extract features.
WINDOW_SIZE = int(cfg_all["features"]["window_size"])

# Output folder
DST_FOLDER = args.output_folder +"/output/" + DST_FOLDER_NAME + "/" # ej "DST_FOLDER_NAME" : current time ex)'02-26-15-51-12-106'
DST_SKELETON_FOLDER_NAME = cfg["output"]["skeleton_folder_name"]
DST_VIDEO_NAME = cfg["output"]["video_name"]
# framerate of output video.avi
DST_VIDEO_FPS = float(cfg["output"]["video_fps"])

#EJ
ORIGINAL_DST_FOLDER = args.output_folder + "/image/" + DST_FOLDER_NAME + "/" 

# Video setttings

# If data_type is webcam, set the max frame rate.
SRC_WEBCAM_MAX_FPS = float(cfg["settings"]["source"]
                           ["webcam_max_framerate"])

# If data_type is video, set the sampling interval.
# For example, if it's 3, then the video will be read 3 times faster.
SRC_VIDEO_SAMPLE_INTERVAL = int(cfg["settings"]["source"]
                                ["video_sample_interval"])

# Openpose settings
OPENPOSE_MODEL = cfg["settings"]["openpose"]["model"]
OPENPOSE_IMG_SIZE = cfg["settings"]["openpose"]["img_size"]

# Display settings
img_disp_desired_rows = int(cfg["settings"]["display"]["desired_rows"])


# -- Function


def select_images_loader(src_data_type, src_data_path):
    if src_data_type == "video":
        images_loader = lib_images_io.ReadFromVideo(
            src_data_path,
            sample_interval=SRC_VIDEO_SAMPLE_INTERVAL)

    elif src_data_type == "folder":
        images_loader = lib_images_io.ReadFromFolder(
            folder_path=src_data_path)

    elif src_data_type == "webcam":
        if src_data_path == "":
            webcam_idx = 0
        elif src_data_path.isdigit():
            webcam_idx = int(src_data_path)
        else:
            webcam_idx = src_data_path
        images_loader = lib_images_io.ReadFromWebcam(
            SRC_WEBCAM_MAX_FPS, webcam_idx)
    return images_loader


class MultiPersonClassifier(object):
    ''' This is a wrapper around ClassifierOnlineTest
        for recognizing actions of multiple people.
    '''

    def __init__(self, model_path, classes):

        self.dict_id2clf = {}  # human id -> classifier of this person

        # Define a function for creating classifier for new people.
        self._create_classifier = lambda human_id: ClassifierOnlineTest(
            model_path, classes, WINDOW_SIZE, human_id)

    def classify(self, dict_id2skeleton):
        ''' Classify the action type of each skeleton in dict_id2skeleton '''

        # Clear people not in view
        old_ids = set(self.dict_id2clf)
        cur_ids = set(dict_id2skeleton)
        humans_not_in_view = list(old_ids - cur_ids)
        for human in humans_not_in_view:
            del self.dict_id2clf[human]

        # Predict each person's action
        id2label = {}
        for id, skeleton in dict_id2skeleton.items():

            if id not in self.dict_id2clf:  # add this new person
                self.dict_id2clf[id] = self._create_classifier(id)

            classifier = self.dict_id2clf[id]
            id2label[id] = classifier.predict(skeleton)  # predict label
            # print("\n\nPredicting label for human{}".format(id))
            # print("  skeleton: {}".format(skeleton))
            # print("  label: {}".format(id2label[id]))

        return id2label

    def get_classifier(self, id):
        ''' Get the classifier based on the person id.
        Arguments:
            id {int or "min"}
        '''
        if len(self.dict_id2clf) == 0:
            return None
        if id == 'min':
            id = min(self.dict_id2clf.keys())
        return self.dict_id2clf[id]


def remove_skeletons_with_few_joints(skeletons):
    ''' Remove bad skeletons before sending to the tracker '''
    good_skeletons = []
    for skeleton in skeletons:
        px = skeleton[2:2+13*2:2]
        py = skeleton[3:2+13*2:2]
        num_valid_joints = len([x for x in px if x != 0])
        num_leg_joints = len([x for x in px[-6:] if x != 0])
        total_size = max(py) - min(py)
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # IF JOINTS ARE MISSING, TRY CHANGING THESE VALUES:
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if num_valid_joints >= 5 and total_size >= 0.1 and num_leg_joints >= 0:
            # add this skeleton only when all requirements are satisfied
            good_skeletons.append(skeleton)
    return good_skeletons


def draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,
                    skeleton_detector, multiperson_classifier):
    ''' Draw skeletons, labels, and prediction scores onto image for display '''

    # Resize to a proper size for display
    r, c = img_disp.shape[0:2]
    desired_cols = int(1.0 * c * (img_disp_desired_rows / r))
    img_disp = cv2.resize(img_disp,
                          dsize=(desired_cols, img_disp_desired_rows))

    # Draw all people's skeleton
    skeleton_detector.draw(img_disp, humans)

    # Draw bounding box and label of each person
    if len(dict_id2skeleton):
        for id, label in dict_id2label.items():
            skeleton = dict_id2skeleton[id]
            # scale the y data back to original
            skeleton[1::2] = skeleton[1::2] / scale_h
            # print("Drawing skeleton: ", dict_id2skeleton[id], "with label:", label, ".")
            #### EJ 수정 
            if label == 'sitting': new_label = '1.0 met (sitting)'
            elif label == 'stand': new_label = '1.2 met (stand)'
            elif label == 'walking': new_label = '1.7 met (walking)'
            elif label == 'exercise': new_label = '3.5 met (exercise)'
            elif label == 'sleeping': new_label = '0.7 met (sleeping)'
            else: new_label=label
            
            lib_plot.draw_action_result(img_disp, id, skeleton, new_label)

    # Add blank to the left for displaying prediction scores of each class
    img_disp = lib_plot.add_white_region_to_left_of_image(img_disp)

    ##cv2.putText(img_disp, "Frame:" + str(ith_img),
    #            (20, 20), fontScale=1.5, fontFace=cv2.FONT_HERSHEY_PLAIN,
    #            color=(0, 0, 0), thickness=2)

    # Draw predicting score for only 1 person
    if len(dict_id2skeleton):
        classifier_of_a_person = multiperson_classifier.get_classifier(
            id='min')
        #classifier_of_a_person.draw_scores_onto_image(img_disp)
    return img_disp


def get_the_skeleton_data_to_save_to_disk(dict_id2skeleton):
    '''
    In each image, for each skeleton, save the:
        human_id, label, and the skeleton positions of length 18*2.
    So the total length per row is 2+36=38
    '''
    skels_to_save = []
    for human_id in dict_id2skeleton.keys():
        label = dict_id2label[human_id]
        skeleton = dict_id2skeleton[human_id]
        skels_to_save.append([[human_id, label] + skeleton.tolist()])
    return skels_to_save

# EJ
def file_name(SRC_DATA_PATH):
    import glob, os
    f_name_list=[]
    pp=SRC_DATA_PATH+'*.jpg'
    files=glob.glob(pp)
    files.sort()
    for i in files:
        f_name_list.append(os.path.basename(i))
    return f_name_list

# -- Main
if __name__ == "__main__":

    # -- Detector, tracker, classifier

    skeleton_detector = SkeletonDetector(OPENPOSE_MODEL, OPENPOSE_IMG_SIZE)

    multiperson_tracker = Tracker()

    multiperson_classifier = MultiPersonClassifier(SRC_MODEL_PATH, CLASSES)

    # -- Image reader and displayer
    images_loader = select_images_loader(SRC_DATA_TYPE, SRC_DATA_PATH)
    img_displayer = lib_images_io.ImageDisplayer()

    # -- Init output

    # video writer
    video_writer = lib_images_io.VideoWriter(
        DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)
    
    # EJ start time setting
    Start_time=(Start_time+datetime.timedelta(minutes=1)).replace(second=0, microsecond=0)
    Start_trial=0
    # -- Read images and process
    try:
        ith_img = -1
        
        while images_loader.has_image():
            print("now:", datetime.datetime.now(), "start_time:", Start_time)
            
            '''
            This performed at the first time model execute
            if: 촬영 처음 시작(trial==0) and 현재 시간이 Start_time 보다 전일 때 pass
            elif: 촬영 처음 시작(trial==0) and 현재 시간이 Start_time 보다 후일때 그 시점의 폴더 생성 and break
            else: 촬영이 계속 실행(trial==1)되고 있을때 다음으로 넘어가기(break)
            
            while True:
                if datetime.datetime.now() < Start_time: pass
                elif Start_trial==0:     
                    _, DST_FOLDER_NAME = get_dst_folder_name(SRC_DATA_TYPE, SRC_DATA_PATH)
                    DST_FOLDER = args.output_folder +"/output/" + DST_FOLDER_NAME + "/"
                    ORIGINAL_DST_FOLDER = args.output_folder +"/image/" + DST_FOLDER_NAME + "/"
                    
                    os.makedirs(DST_FOLDER, exist_ok=True)
                    os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True)
                    os.makedirs(ORIGINAL_DST_FOLDER, exist_ok=True)
                    Start_trial=1
                    break
                else: break
            '''
            # 현재 시간이 Start_time 시간 보다 20초 이후 인지 확인 
            # (if) 20초가 지나지 않을 경우 같은 폴더에 이미지 저장 / (else) 20초가 지나면 새로 폴더 생성             
            if (datetime.datetime.now()-Start_time).seconds <=20:

                # -- Read image
                img = images_loader.read_image()
                #img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) #1116 NH 수정
                ith_img += 1
                img_disp = img.copy()
                img_origin=img.copy()
                print(f"\nProcessing {ith_img}th image ...")

                # -- Detect skeletons
                humans = skeleton_detector.detect(img)
                skeletons, scale_h = skeleton_detector.humans_to_skels_list(humans)
                skeletons = remove_skeletons_with_few_joints(skeletons)

                # -- Track people
                dict_id2skeleton = multiperson_tracker.track(skeletons)  # int id -> np.array() skeleton
                print(dict_id2skeleton)
                
                # -- Recognize action of each person
                if len(dict_id2skeleton):
                    dict_id2label = multiperson_classifier.classify(dict_id2skeleton)

                # -- Draw
                img_disp = draw_result_img(img_disp, ith_img, humans, dict_id2skeleton,skeleton_detector, multiperson_classifier)
                
                # Print label of a person
                if len(dict_id2skeleton):
                    min_id = min(dict_id2skeleton.keys())
                    print("prediced label is :", dict_id2label[min_id])
                    #print('dict_id2label: ', dict_id2label, min_id)  #EJ
                    #print("src_data_path: ", SRC_DATA_PATH) #EJ
                    
                # -- Get skeleton data and save to file
                skels_to_save = get_the_skeleton_data_to_save_to_disk(
                    dict_id2skeleton)
                    
                lib_commons.save_listlist(
                    DST_FOLDER + DST_SKELETON_FOLDER_NAME +
                    SKELETON_FILENAME_FORMAT.format(ith_img+1),
                    skels_to_save)
                
                if skels_to_save :
                # -- Display image, and write to video.avi
                    save_path = DST_FOLDER+lib_commons.get_time_string()+'.jpg' #EJ
                    img_displayer.display(img_disp, wait_key_ms=1)
                    #img_displayer.save(save_path, img_disp)  #EJ
                    video_writer.write(img_disp)
                    
                    # original image EJ
                    #os.makedirs(ORIGINAL_DST_FOLDER, exist_ok=True)
                    #original_img_sav_path = ORIGINAL_DST_FOLDER + lib_commons.get_time_string() + '.jpg' #EJ
                    #img_displayer.save(original_img_sav_path, img_origin)

            else:
                # Start_time: current time save, DST_FOLDER_NAME: folder name set with the Start_time
                Start_time, DST_FOLDER_NAME = get_dst_folder_name(SRC_DATA_TYPE, SRC_DATA_PATH)
                DST_FOLDER = args.output_folder +"/output/" + DST_FOLDER_NAME + "/"
                ORIGINAL_DST_FOLDER = args.output_folder +"/image/" + DST_FOLDER_NAME + "/"

                # output folder make every 20 secs after
                os.makedirs(DST_FOLDER, exist_ok=True)
                os.makedirs(DST_FOLDER + DST_SKELETON_FOLDER_NAME, exist_ok=True) 
                
                #os.makedirs(ORIGINAL_DST_FOLDER, exist_ok=True)
                
                video_writer = lib_images_io.VideoWriter(DST_FOLDER + DST_VIDEO_NAME, DST_VIDEO_FPS)
                
                ith_img=-1
                print("----- new saving started------")     
                
                
    finally:
        video_writer.stop()
        print("Program ends")
