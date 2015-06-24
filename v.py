#!/usr/bin/env python

import numpy as np
import cv2
import ransacLaneFit
import sys
import os

if not '/Users/Xixuan/Documents/ppg/' in sys.path:
    sys.path.append('/Users/Xixuan/Documents/ppg/')

# == template ==

pieceTemplate = np.array([[0, 0, 0, 255, 255, 255, 0, 0, 0],
                          [0, 0, 0, 255, 255, 255, 0, 0, 0],
                          [0, 0, 0, 255, 255, 255, 0, 0, 0],
                          [0, 0, 0, 255, 255, 255, 0, 0, 0],
                          [0, 0, 0, 255, 255, 255, 0, 0, 0],
                          [0, 0, 0, 255, 255, 255, 0, 0, 0],
                          [0, 0, 0, 255, 255, 255, 0, 0, 0],
                          [0, 0, 0, 255, 255, 255, 0, 0, 0]],
                         dtype='uint8')

temp = np.ones((pieceTemplate.shape[0], pieceTemplate.shape[1], 3), dtype='uint8')

temp[:, :, 0] = pieceTemplate/260
temp[:, :, 1] = pieceTemplate
temp[:, :, 2] = pieceTemplate

tempHeight, tempWidth, tempChannel = temp.shape

# == perspective ==

M = np.float32([[  1.05123932e+01,   3.06117249e+01,  -3.06117249e+03],
                [ -6.10622664e-16,   1.24148718e+01,  -1.90247863e+02],
                [ -1.25767452e-17,   9.51239316e-02,   1.00000000e+00]])

# == video reader ==

cap = cv2.VideoCapture('/Users/Xixuan/Documents/ppg/highway.mp4')

videoHeight, videoWidth, videoChannel = cap.read()[1].shape

# == video writer ==
fps = 30
capSize = (int(cap.get(3)), int(cap.get(4))) # this is the size of my source video
fourcc = cv2.cv.CV_FOURCC('m', 'p', '4', 'v') # note the lower case
out = cv2.VideoWriter()
success = out.open('output_{0}.mov'.format(int(os.times()[4]*100)), fourcc, fps, capSize, True)
# == parameters ==

params = {'roiTop': 230,
          'roiBottom': 340,
          'thresholdMatchTemp': -60,
          'blurParam': 5,
          'laneColor': (0, 0, 255),
          'dilateParam': 10,
          'brightness': 70,
          'longCoef': 7,
          'leastNoPointForLane': 800,
          'laneExtraTop': 5,
          'regularizationRansac': 30,
          'narrowCoef': 2,
          'ransacPercent': 0.7
          }

# == text frame ==

staticTextFrame = np.zeros((videoHeight, videoWidth, videoChannel), dtype='uint8')
positionText = 20
for key, value in params.items():
    cv2.putText(staticTextFrame, key + ': ' + str(value), (10, positionText), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    positionText += 20

# == roi display in original frame
roiHeight = params['roiBottom'] - params['roiTop']

roiOriginal = np.ones((roiHeight, videoWidth, 3), dtype = 'uint8')
roiOriginal[:, :, 1] *= 255

roiOriginal = cv2.warpPerspective(roiOriginal, M, (videoWidth, roiHeight), roiOriginal, cv2.WARP_INVERSE_MAP)

roiBackground = np.zeros((videoHeight, videoWidth, 3), dtype='uint8')
roiBackground[params['roiTop']:params['roiBottom'], :, :] = roiOriginal

# == lane display in original frame
laneHeight = videoHeight - params['roiTop'] + params['laneExtraTop']

laneOriginal = np.zeros((laneHeight, videoWidth), dtype = 'uint8')

# == main loop ==

meanChannel = [0, 0, 0]
frameNo = 0
black = np.zeros((roiHeight, videoWidth), dtype = 'uint8')

while True:

    # == count frame No. ==

    frameNo += 1

    # == read frame ==

    _, original = cap.read()

    # == roi ==

    roadPart = original[params['roiTop']:params['roiBottom'], :, :]

    meanChannel[0] = roadPart[:, :, 0].mean()
    meanChannel[1] = roadPart[:, :, 1].mean()
    meanChannel[2] = roadPart[:, :, 2].mean()

    power = (np.log(params['brightness']) - np.log(roadPart.max())) / (np.log(roadPart.mean()) - np.log(roadPart.max()))

    roadPart = (((roadPart.astype('float32') / roadPart.max()) ** power) * 255).astype('uint8')

    roadComp = roadPart.astype('float32')
    roadComp = (roadComp[:, :, 0]*(-0.8) + roadComp[:, :, 1]*0.9 + roadComp[:, :, 2]*0.9)
    roadComp[np.where(roadComp < 0)] = 0
    roadComp[np.where(roadComp > 255)] = 255
    roadComp = roadComp.astype('uint8')

    # == warp & select GR channel & resize ==
    laneOriginal[params['laneExtraTop']:(params['laneExtraTop']+roiHeight), :] = roadComp

    roadPartWarped = cv2.warpPerspective(laneOriginal, M, (videoWidth, laneHeight))

    roadPartWarpedLong = cv2.resize(roadPartWarped, (videoWidth/params['narrowCoef'], laneHeight*params['longCoef']))

    # == match & unresize ==

    roadMatch = cv2.matchTemplate(roadPartWarpedLong, pieceTemplate, cv2.TM_CCORR)
    roadMatch -= roadMatch.min()
    roadMatch /= roadMatch.max()

    road = cv2.adaptiveThreshold((roadMatch*255).astype('uint8'), 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 81, params['thresholdMatchTemp'])
    road = road.astype('float32')
    #roadBlured = cv2.erode(road, np.ones((np.floor(params['dilateParam']/2), np.floor(params['dilateParam']/2)), np.uint8), iterations=1)
    roadBlured = cv2.medianBlur(road, params['blurParam'])
    roadBlured = cv2.resize(roadBlured, (videoWidth/params['narrowCoef'], laneHeight*params['longCoef']))

    # == ransac lane fit ==

    #laneBackground = roadBlured
    laneBackground = ransacLaneFit.rlf(roadBlured, params, videoWidth, laneHeight, 100, 10, 10, params['ransacPercent'], True)

    # == unwarp & road mask in road part==
    road = cv2.resize(laneBackground, (videoWidth, laneHeight))
    road = cv2.dilate(road, np.ones((params['dilateParam'], params['dilateParam']), np.uint8), iterations=1)

    roadPWU = np.zeros((laneHeight, videoWidth, videoChannel), dtype='uint8')

    roadMask = cv2.warpPerspective(road, M, (videoWidth, laneHeight), roadPWU, cv2.WARP_INVERSE_MAP)

    # == frame mask for road ==

    frameMask = np.zeros((videoHeight, videoWidth), dtype='float32')
    frameMask[(params['roiTop'] - params['laneExtraTop']):, :] = roadMask

    # == put color to the frame mask ==

    roadFrame = np.zeros((videoHeight, videoWidth, videoChannel), dtype='uint8')

    roadFrame[:, :, 0] = frameMask * params['laneColor'][0]
    roadFrame[:, :, 1] = frameMask * params['laneColor'][1]
    roadFrame[:, :, 2] = frameMask * params['laneColor'][2]

    # == put raodFrame and original together==

    backGround = cv2.bitwise_and(original, original, mask=cv2.bitwise_not((frameMask*255).astype('uint8')))
    final = cv2.add(roadFrame, backGround)

    # == brightness and frame No. ==

    text = {'power': str(power),
            'mean': (np.rint(meanChannel[0]), np.rint(meanChannel[1]), np.rint(meanChannel[2])),
            'frameNo': frameNo
            }

    positionTextSecondPart = positionText
    for key, value in text.items():
        cv2.putText(staticTextFrame, key + ': ' + str(value), (10, positionTextSecondPart), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        positionTextSecondPart += 20

    # == show final result ==

    original = cv2.addWeighted(original, 0.9, roiBackground, 0.1, 0.5)

    cv2.imshow('result', np.hstack((np.vstack((original, final)), np.vstack((staticTextFrame, roadFrame)))))

    # == keyboard operations ==

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if cv2.waitKey(1) & 0xFF == ord('p'):
        cv2.imwrite('rpo_' + str(frameNo) + '.png', roadPart)

    # == put the words flying white ==
    positionTextSecondPart = positionText
    for key, value in text.items():
        cv2.putText(staticTextFrame, key + ': ' + str(value), (10, positionTextSecondPart), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        positionTextSecondPart += 20

    # == record video ==
    out.write(final)

cap.release()
out.release()
cv2.destroyAllWindows()
