""""
The main file for the fall detection task 

Written by Hassan Keshvari Khojasteh

"""

""" Fist let import the necessary library"""

import math
import numpy as np
import cv2.cv2 as cv2
from sklearn.externals import joblib
from scipy.ndimage.interpolation import shift
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC



class falldetection():
    """
    Feature Extracting for training SVM

    """

    def features_extractor(self,path_to_video, path_to_save):
        """
        This Function save Extracted features from  desired video with shape of (number_of_fall or not_fall ,3*25)

        path_to_video---the path to desired video for extracting features
        path_to_save---the path to save extracted features

        return
        """


        """ define the necessary parameters"""
        count = 0
        framenumber = 0
        mat3 = 0
        cnt = 30
        mat1 = np.zeros(10)
        mat2 = np.zeros(10)
        features = np.zeros([3, 25])
        features_temp = []
        features_all = []
        font = cv2.FONT_HERSHEY_SIMPLEX

        """capturing video from Webcam or from Computer"""

        cap = cv2.VideoCapture(path_to_video)
        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=5000, nmixtures=5, backgroundRatio=0.1)

        while True:
            # Reading the famres of Video
            ret, frame = cap.read()
            if ret == False:
                break
            if cnt < 25:
                cv2.putText(frame, 'feature extraction', (150, 50), font, 1, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.imshow('frame', frame)

            # Blur the readed frame with lowpass median filter
            frame1 = cv2.medianBlur(frame, 11)
            # apply MOG background subtractor
            fgmask = fgbg.apply(frame1)
            kernel = np.ones((5, 5), np.uint8)
            # apply dilation filter
            dilation = cv2.dilate(fgmask, kernel, iterations=14)
            a = cv2.dilate(fgmask, kernel, iterations=14)
            framenumber = framenumber + 1
            ret, threshed_img = cv2.threshold(a, 127, 255,
                                              cv2.THRESH_BINARY)
            # find contours in the frame
            contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            # if the z key pressed, start feature extraction
            p = cv2.waitKey(1) & 0xff
            if p == ord('z'):
                cnt = 0

            if len(contours) != 0 and cnt < 25:

                # find the biggest area
                c = max(contours, key=cv2.contourArea)
                # Extract the height to width ratio and append the value to features_temp
                x, y, w, h = cv2.boundingRect(c)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                hw = h / w
                features_temp.append(hw)
                # Extract the angle and append absolute value of it to features_temp
                rows, cols = a.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)

                if (lefty - righty) != 0:
                    x = math.atan(1 / ((cols - 1) / (lefty - righty)))
                    x1 = 180 * x / math.pi
                    features_temp.append(abs(x1))
                # Extract the momentum and add it's value to features_temp
                M = cv2.moments(c)

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                mat1[framenumber % 10] = cx
                mat2[framenumber % 10] = cy
                mat4 = mat2[5] - mat2[0]
                for i in range(9):
                    mat3 = mat3 + np.sqrt(((mat1[i + 1] - mat1[i]) * (mat1[i + 1] - mat1[i])) + (
                            (mat2[i + 1] - mat2[i]) * (mat2[i + 1] - mat2[i])))

                features_temp.append(mat3)
                # put features_temp in features columns until the 25'th frame
                if cnt < 25 and len(features_temp) != 0:
                    features[:, cnt] = np.reshape(np.asarray(features_temp), [3])
                cnt += 1
            # append the extracted features of 25 frames to features_all
            if cnt == 25:
                features_all.append(features)
                features = np.zeros([3, 25])
                cnt = 30

            # back the value of features_temp and mat3 to their initial values
            features_temp = []
            mat3 = 0

        cap.release()
        cv2.destroyAllWindows()

        # Convert the list to numpy array and the final array will have the shape of (number_of_fall,3,25) and is suitable to train SVM
        features_all = np.asarray(features_all)
        features_all = features_all.reshape([len(features_all), 3 * 25])
        # Saving the extracted features
        np.savetxt(path_to_save, features_all, delimiter=',')
        return


    """"
    Training svm with the extracted features of our dataset that is created by Mr.Farahnejad
    
    """

    def train_svm(self,path_to_old_fall_features, path_to_old_not_fall_features, path_to_new_fall_features,
                  path_to_new_not_fall_features, path_to_save_trained_SVM):
        """
        path_to_old_fall_features---the path to old fall features extracted
        path_to_old_not_fall_features--- the path to old not fall features extracted
        path_to_new_fall_features---the path to new fall features extracted
        path_to_new_not_fall_features--- the path to new not fall features extracted
        path_to_save_trained_SVM---the path to save trained svm
        return
        this function will print the accuracy in Train set data and the accuracy in Test set data and finally
        the cross_validation_accuracy
        """


        # loading the new extracted features of fall and not_fall windows

        fall_new = np.loadtxt(path_to_new_fall_features, delimiter=',')
        label_new_fall = np.ones([np.shape(fall_new)[0], 1], dtype=np.int16)

        not_fall_new = np.loadtxt(path_to_new_not_fall_features, delimiter=',')
        label_new_not_fall = np.zeros([np.shape(not_fall_new)[0], 1], dtype=np.int16)

        # loading the old extracted features of fall and not_fall windows

        fall_old = np.loadtxt(path_to_old_fall_features, delimiter=',')
        label_old_fall = np.ones([np.shape(fall_old)[0], 1], dtype=np.int16)

        not_fall_old = np.loadtxt(path_to_old_not_fall_features, delimiter=',')
        label_old_not_fall = np.zeros([np.shape(not_fall_old)[0], 1], dtype=np.int16)

        # Concatenate the new and old features to train svm
        var = np.concatenate([fall_old, not_fall_old, fall_new, not_fall_new], axis=0)
        targ = np.concatenate([label_old_fall, label_old_not_fall, label_new_fall, label_new_not_fall], axis=0)

        # split data to train and test
        var_train, var_test, targ_train, targ_test = train_test_split(var, targ, train_size=0.7,
                                                                      random_state=0, shuffle=True)

        # hard margin,linearkernel

        svclassifier = SVC(kernel='linear', C=10000)
        svclassifier.fit(var_train, targ_train)
        y_pred = svclassifier.predict(var_train)
        print(classification_report(targ_train, y_pred))

        # test error
        y_pred_test_li = svclassifier.predict(var_test)
        print(classification_report(targ_test, y_pred_test_li))

        # cross validation  score

        svclassifier_c = SVC(kernel='linear', C=10000)
        print(np.mean(cross_val_score(svclassifier_c, var, targ, cv=2)))

        # save the trained svm
        _ = joblib.dump(svclassifier, path_to_save_trained_SVM, compress=9)
        return


    """
    Fall detection using the trained SVM

    """

    def fall_webcam(self,path_svm, path_video):
        """
        inputs

        path_svm---the path to trained svm
        path_video---the path to the  video

        return
        """


        """ import the trained SVM"""
        svm = joblib.load(path_svm)
        """ define the necessary parameters"""
        count = 0
        features_temp = []
        labels = []
        features = np.zeros([3, 25])
        framenumber = 0
        mat3 = 0
        cnt = 0
        h = 0
        mat1 = np.zeros(10)
        mat2 = np.zeros(10)

        """capturing video from from Computer"""

        cap = cv2.VideoCapture(path_video)

        fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=5000, nmixtures=5, backgroundRatio=0.1)
        font = cv2.FONT_HERSHEY_SIMPLEX

        while True:
            # Reading the famres of Video
            ret, frame = cap.read()

            if ret == False:
                break
            # Blur the readed frame with lowpass median filter
            frame1 = cv2.medianBlur(frame, 11)
            if h == 1:
                cv2.putText(frame, 'FALL', (180, 80), font, 2, (0, 0, 255), 10, cv2.LINE_AA)
                h = 0
            # apply MOG background subtractor
            fgmask = fgbg.apply(frame1)
            kernel = np.ones((5, 5), np.uint8)
            # apply dilation filter
            dilation = cv2.dilate(fgmask, kernel, iterations=14)
            a = cv2.dilate(fgmask, kernel, iterations=14)
            framenumber = framenumber + 1
            ret, threshed_img = cv2.threshold(a, 127, 255,
                                              cv2.THRESH_BINARY)
            # find contours in the frame
            contours, hierarchy = cv2.findContours(threshed_img, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            if len(contours) != 0:

                # find the biggest area
                c = max(contours, key=cv2.contourArea)
                # Extract the height to eidth ratio and append the value to features_temp
                x, y, w, h = cv2.boundingRect(c)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                hw = h / w
                features_temp.append(hw)
                # Extract the angle and append absolute value of it to features_temp
                rows, cols = a.shape[:2]
                [vx, vy, x, y] = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
                lefty = int((-x * vy / vx) + y)
                righty = int(((cols - x) * vy / vx) + y)

                if (lefty - righty) != 0:
                    x = math.atan(1 / ((cols - 1) / (lefty - righty)))
                    x1 = 180 * x / math.pi
                    features_temp.append(abs(x1))
                else:
                    features_temp.append(0)

                # Extract the momentum and add it's value to features_temp
                M = cv2.moments(c)

                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                mat1[framenumber % 10] = cx
                mat2[framenumber % 10] = cy
                mat4 = mat2[5] - mat2[0]
                for i in range(9):
                    mat3 = mat3 + np.sqrt(((mat1[i + 1] - mat1[i]) * (mat1[i + 1] - mat1[i])) + (
                            (mat2[i + 1] - mat2[i]) * (mat2[i + 1] - mat2[i])))

                features_temp.append(mat3)
                # put features_temp in features columns until the 25'th frame
                if cnt < 25 and len(features_temp) != 0:
                    features[:, cnt] = np.reshape(np.asarray(features_temp), [3])
                cnt += 1
            # shift the features to left and add the new features_temp in last colum of features
            if cnt > 25 and len(features_temp) != 0:
                features = shift(features, [0, -1])
                features[:, 24] = np.reshape(np.asarray(features_temp), [3])
            # predict the label of 25 frames by trained svm
            if cnt > 24 and len(contours) != 0:
                features_input = np.reshape(features, [1, 3 * 25])
                label = svm.predict(features_input)
                labels.append(label)
                # if predicted label is 1(fall) the put text fall in the current frame
                if label == 1:
                    cv2.putText(frame, 'FALL', (180, 80), font, 2, (0, 0, 255), 10, cv2.LINE_AA)
                    h = label
            # show the current frame
            cv2.imshow('frame1', frame)
            # out.write(frame)
            # back the value of features_temp and mat3 to their initial values
            features_temp = []
            mat3 = 0
            k = cv2.waitKey(10) & 0xff
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
        return



def main():

    fall = falldetection()

    ############## First Extract the defined features of video ############################################
    path_to_save = './etracted_new_fall.txt'
    path_to_video = './fall1.mp4'
    fall.features_extractor(path_to_video, path_to_save)
    
    path_to_save = './etracted_new_not_fall.txt'
    path_to_video = './fall1.mp4'
    fall.features_extractor(path_to_video, path_to_save)
    
    ############## Train the SVM with extracted features ##################################################
    path_to_new_fall_features = './etracted_new_fall.txt'
    path_to_new_not_fall_features = './etracted_new_not_fall.txt'
    path_to_old_fall_features = './etracted_old_fall.txt'
    path_to_old_not_fall_features = './etracted_old_not_fall.txt'
    path_to_save_trained_SVM = './trained_svm.joblib1.pkl'

    fall.train_svm(path_to_old_fall_features, path_to_old_not_fall_features, path_to_new_fall_features,
              path_to_new_not_fall_features, path_to_save_trained_SVM)

    ############## Detect the Fall with trined SVM ##################################################
    path_svm = './trained_svm.joblib.pkl'
    path_video = './fall2.mp4'

    fall.fall_webcam(path_svm, path_video)

if __name__=='__main__':
 main ()







