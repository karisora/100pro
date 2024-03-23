# -*- coding: utf-8 -*-
import numpy as np
import cv2
import sys
import logging

logging.basicConfig(level=logging.DEBUG)

aruco = cv2.aruco #arucoライブラリ
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

client = cv2.VideoCapture(0)

def main():
    cnt=0
    marker_length = 0.1 # [m]
    camera_matrix = np.array([[9.31357583e+03 ,0.00000000e+00 ,1.61931898e+03],
                              [0.00000000e+00 ,9.64867367e+03 ,1.92100899e+03],
                              [0.00000000e+00 ,0.00000000e+00 ,1.00000000e+00]])

    #カメラキャリブレーションでパラメータを求める↑

    distortion_coeff = np.array( [[ 0.22229833, -6.34741982,  0.01145082,  0.01934784, -8.43093571]] )
    while True:
        ret, frame = client.read()  # 新しい画像を取得

        corners, ids, rejectedImgPoints = aruco.detectMarkers(frame, dictionary)

        aruco.drawDetectedMarkers(frame, corners, ids, (0,255,255))

        if len(corners) > 0:
            for i, corner in enumerate(corners):
                # rvec -> rotation vector, tvec -> translation vector
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corner, marker_length, camera_matrix, distortion_coeff)

                tvec = np.squeeze(tvec)
                rvec = np.squeeze(rvec)
                rvec_matrix = cv2.Rodrigues(rvec)
                rvec_matrix = rvec_matrix[0]
                transpose_tvec = tvec[np.newaxis, :].T
                proj_matrix = np.hstack((rvec_matrix, transpose_tvec))
                euler_angle = cv2.decomposeProjectionMatrix(proj_matrix)[6] # [deg]
                print("x : " + str(tvec[0]))
                print("y : " + str(tvec[1]))
                print("z(distance) : " + str(tvec[2]))
                print("roll : " + str(euler_angle[0]))
                print("pitch: " + str(euler_angle[1]))
                print("yaw  : " + str(euler_angle[2]))

                draw_pole_length = marker_length/2 # 現実での長さ[m]
                #aruco.drawDetectedMarkers(frame, camera_matrix, distortion_coeff, rvec, tvec, draw_pole_length)

        cv2.imwrite('distance'+str(cnt)+'.png',frame)
        cnt+=1

        key = cv2.waitKey(50)
        if key == 27: # ESC
            break
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
