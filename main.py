import cv2
import Region, feature
from validity import *
from average import *


def main():
    print('------------------------------------------')
    levels = int(input("Enter number of levels: "))
    choice = input("Select method [SSD, SAD, NCC]: ")
    type = int(input("Select format [1. Region-based, 2. Feature-based]: "))
    template = int(input("Enter template size: "))
    window = int(input("Enter window size: "))
    print('------------------------------------------')

    if type == 1:
        # region functions
        if choice == 'SSD':
            left, right = Region.ssd(levels, template, window)
            cv2.imwrite('Results/Scores/Left to Right Disparity (Region) - SSD.jpg', left)
            cv2.imwrite('Results/Scores/Right to Left Disparity (Region) - SSD.jpg', right)
        
            l, r = validity(left, right)
            cv2.imwrite('Results/Validated/Left to Right Disparity (Region) - SSD.jpg', l)
            cv2.imwrite('Results/Validated/Right to Left Disparity (Region) - SSD.jpg', r)

            l, r = averaging(l, r)
            cv2.imwrite('Results/Averaged/Left to Right Disparity (Region) - SSD.jpg', l)
            cv2.imwrite('Results/Averaged/Right to Left Disparity (Region) - SSD.jpg', r)

        elif choice == 'SAD':
            left, right = Region.sad(levels, template, window)
            cv2.imwrite('Results/Scores/Left to Right Disparity (Region) - SAD.jpg', left)
            cv2.imwrite('Results/Scores/Right to Left Disparity (Region) - SAD.jpg', right)
            
            l, r = validity(left, right)
            cv2.imwrite('Results/Validated/Left to Right Disparity (Region) - SAD.jpg', l)
            cv2.imwrite('Results/Validated/Right to Left Disparity (Region) - SAD.jpg', r)

            l, r = averaging(l, r)
            cv2.imwrite('Results/Averaged/Left to Right Disparity (Region) - SAD.jpg', l)
            cv2.imwrite('Results/Averaged/Right to Left Disparity (Region) - SAD.jpg', r)
        
        elif choice == 'NCC':
            left, right = Region.ncc(levels, template, window)
            cv2.imwrite('Results/Scores/Left to Right Disparity (Region) - NCC.jpg', left)
            cv2.imwrite('Results/Scores/Right to Left Disparity (Region) - NCC.jpg', right)
            
            l, r = validity(left, right)
            cv2.imwrite('Results/Validated/Left to Right Disparity (Region) - NCC.jpg', l)
            cv2.imwrite('Results/Validated/Right to Left Disparity (Region) - NCC.jpg', r)

            l, r = averaging(l, r)
            cv2.imwrite('Results/Averaged/Left to Right Disparity (Region) - NCC.jpg', l)
            cv2.imwrite('Results/Averaged/Right to Left Disparity (Region) - NCC.jpg', r)

    elif type == 2:
        # feature functions
        if choice == 'SSD':
            left, right = feature.ssd(levels, template, window)
            cv2.imwrite('Results/Scores/Left to Right Disparity (Feature) - SSD.jpg', left)
            cv2.imwrite('Results/Scores/Right to Left Disparity (Feature) - SSD.jpg', right)

            l, r = validity(left, right)
            cv2.imwrite('Results/Validated/Left to Right Disparity (Feature) - SSD.jpg', l)
            cv2.imwrite('Results/Validated/Right to Left Disparity (Feature) - SSD.jpg', r)

            l, r = averaging(l, r)
            cv2.imwrite('Results/Averaged/Left to Right Disparity (Feature) - SSD.jpg', l)
            cv2.imwrite('Results/Averaged/Right to Left Disparity (Feature) - SSD.jpg', r)

        elif choice == 'SAD':
            left, right = feature.sad(levels, template, window)
            cv2.imwrite('Results/Scores/Left to Right Disparity (Feature) - SAD.jpg', left)
            cv2.imwrite('Results/Scores/Right to Left Disparity (Feature) - SAD.jpg', right)

            l, r = validity(left, right)
            cv2.imwrite('Results/Validated/Left to Right Disparity (Feature) - SAD.jpg', l)
            cv2.imwrite('Results/Validated/Right to Left Disparity (Feature) - SAD.jpg', r)

            l, r = averaging(l, r)
            cv2.imwrite('Results/Averaged/Left to Right Disparity (Feature) - SAD.jpg', l)
            cv2.imwrite('Results/Averaged/Right to Left Disparity (Feature) - SAD.jpg', r)

        elif choice == 'NCC':
            left, right = feature.ncc(levels, template, window)
            cv2.imwrite('Results/Scores/Left to Right Disparity (Feature) - NCC.jpg', left)
            cv2.imwrite('Results/Scores/Right to Left Disparity (Feature) - NCC.jpg', right)

            l, r = validity(left, right)
            cv2.imwrite('Results/Validated/Left to Right Disparity (Feature) - NCC.jpg', l)
            cv2.imwrite('Results/Validated/Right to Left Disparity (Feature) - NCC.jpg', r)

            l, r = averaging(l, r)
            cv2.imwrite('Results/Averaged/Left to Right Disparity (Feature) - NCC.jpg', l)
            cv2.imwrite('Results/Averaged/Right to Left Disparity (Feature) - NCC.jpg', r)

if __name__ == "__main__":
    main()