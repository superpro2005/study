import cv2
import numpy as np

img = cv2.imread('photo/list2.jpg')
img = cv2.resize(img,(1024,1024) )
orig = img.copy()


gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,thresh = cv2.threshold(gray_img,160,255,cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh ,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnt = max(contours, key = cv2.contourArea)

mask = np.zeros_like(gray_img)
cv2.drawContours(mask,[cnt],-1,255,thickness= -1)
result = cv2.bitwise_and(img,img,mask = mask)

epsilon = 0.02 * cv2.arcLength(cnt,True)
approx = cv2.approxPolyDP(cnt,epsilon,True)
x,y,w,h = cv2.boundingRect(cnt)
cropped = img[y:y+h,x:x+w]

if len(approx) == 4:
    pts = approx.reshape(4,2)

    def points (pts):
        rect = np.zeros((4,2), dtype="float32")
        sum = pts.sum(axis = 1)
        rect[0] = pts[np.argmin(sum)]
        rect[2] = pts[np.argmax(sum)]

        diff = np.diff(pts,axis=1)
        rect[1] =pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    rect  = points(pts)
    tl, tr, bl, br = rect


    widthA = np.linalg.norm(tr-tl)
    widthB = np.linalg.norm(br-bl)
    width = max(int(widthA),int(widthB))

    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    height = max(int(heightA),int(heightB))



    dst = np.array([[0,0],[width-1,0],[width-1,height-1],[0,height-1]],dtype=np.float32)
    M  = cv2.getPerspectiveTransform(rect,dst)
    warped = cv2.warpPerspective(orig,M, (width,height))
    warped = cv2.resize(warped,(512,512))
    cv2.imshow("result.png", cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY))

img_contours = img.copy()
cv2.drawContours(img_contours,contours,-1,(0,0,255), 2)

cv2.imshow("thresh", thresh)
cv2.imshow("result", img_contours)

cv2.waitKey(0)