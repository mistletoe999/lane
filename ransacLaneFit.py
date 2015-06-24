import cv2
import numpy as np


def ransacFit(data, n, k, t, d, l):
    iterations = 0
    bestfit = None
    besterr = np.inf
    while iterations < k:
        maybe_idxs, test_idxs = random_partition(n, data.shape[0])
        maybeinliers = data[maybe_idxs,:]
        test_points = data[test_idxs, :]
        maybemodel = np.poly1d(np.polyfit(maybeinliers[:, 0], maybeinliers[:, 1], 2))
        test_err = np.sqrt((test_points[:, 1] - maybemodel(test_points[:, 0]))**2)
        also_idxs = test_idxs[test_err < t] # select indices of rows with accepted points
        alsoinliers = data[also_idxs,:]
        if len(also_idxs)+len(maybe_idxs) > np.floor(data.shape[0]*d):
            betterdata = np.concatenate( (maybeinliers, alsoinliers) )
            bettermodel = np.poly1d(np.polyfit(betterdata[:, 0], betterdata[:, 1], 2))
            thiserr = np.mean(np.sqrt((betterdata[:, 1] - bettermodel(betterdata[:, 0]))**2)) + sum(np.abs(bettermodel.coeffs[:2])) * l
            if thiserr < besterr:
                besterr = thiserr
                bestfit = bettermodel
        iterations+=1
    return bestfit


def random_partition(n, n_data):
    all_idxs = np.arange( n_data )
    np.random.shuffle(all_idxs)
    idxs1 = all_idxs[:n]
    idxs2 = all_idxs[n:]
    return idxs1, idxs2


def rlf(laneOriginal, params, videoWidth, laneHeight, n = 50, k = 10, t = 8, d = 0.5, display = False):
    left = laneOriginal[:, :videoWidth/(params['narrowCoef']*2)]
    right = laneOriginal[:, (videoWidth/(params['narrowCoef']*2)):]

    leftData = np.where(left > 0.5)
    leftData = np.array([leftData[0], leftData[1]]).transpose((1, 0))
    rightData = np.where(right > 0.5)
    rightData = np.array([rightData[0], rightData[1]]).transpose((1, 0))

    laneX = np.linspace(1, laneHeight*params['longCoef'], laneHeight*params['longCoef']).astype('int')

    leftLane = np.zeros_like(left)
    if leftData.shape[0] > params['leastNoPointForLane']:
        leftModel = ransacFit(leftData, n, k, t, d, params['regularizationRansac'])
        if leftModel:
            leftLaneCoordRaw = leftModel(laneX).astype('int')
            where = np.where(np.logical_or(leftLaneCoordRaw < 0, leftLaneCoordRaw > videoWidth/(params['narrowCoef']*2)))[0]
            leftLaneCoordY = np.delete(leftLaneCoordRaw, where, 0)
            leftLaneCoordX = np.delete(laneX, where, 0)
            leftLane[(leftLaneCoordX - 1, leftLaneCoordY - 1)] = 1

    rightLane = np.zeros_like(right)
    if rightData.shape[0] > params['leastNoPointForLane']:
        rightModel = ransacFit(rightData, n, k, t, d, params['regularizationRansac'])
        if rightModel:
            rightLaneCoordRaw = rightModel(laneX).astype('int')
            where = np.where(np.logical_or(rightLaneCoordRaw < 0, rightLaneCoordRaw > videoWidth/(params['narrowCoef']*2)))[0]
            rightLaneCoordY = np.delete(rightLaneCoordRaw, where, 0)
            rightLaneCoordX = np.delete(laneX, where, 0)
            rightLane[(rightLaneCoordX - 1, rightLaneCoordY - 1)] = 1

    laneBackground = np.hstack((leftLane[:, :videoWidth/(params['narrowCoef']*2)], rightLane[:, :videoWidth/(params['narrowCoef']*2)])).transpose((1, 0)).transpose((1, 0))

    if display:
        bird = cv2.addWeighted(laneOriginal, 0.5, laneBackground.astype('float32'), 0.5, 0.1)
        cv2.imshow('bird', bird)

    return laneBackground