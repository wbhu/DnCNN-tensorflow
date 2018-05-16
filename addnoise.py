# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:02:33 2018

@author: cm72ein
"""
import os
import numpy as np
import cv2
import glob

global src_dir, save_dir
 
def main():
  GlobalToneCurveY=np.array([0,486,958,1438,1914,2392,2870,3350,3826,4304,4780,5256,\
  5736,6226,6718,7212,7702,8196,8688,9182,9680,10192,10698,11210,11722,12244,12760,\
  13268,13782,14306,14826,15348,15866,16374,16884,17390,17902,18430,18946,19472,\
  19990,20496,21002,21510,22010,22500,22990,23480,23972,24462,24954,25446,25926,\
  26402,26878,27352,27826,28300,28774,29250,29712,30170,30628,31084,31534,31980,\
  32420,32866,33306,33748,34188,34630,35064,35488,35914,36340,36754,37164,37574,\
  37982,38380,38778,39170,39568,39948,40328,40702,41078,41446,41806,42166,42530,\
  42878,43224,43568,43914,44246,44576,44904,45236,45542,45840,46140,46440,46736,\
  47036,47332,47634,47914,48200,48482,48762,49032,49298,49562,49832,50070,50302,\
  50538,50776,51012,51246,51480,51716,51936,52156,52374,52594,52798,53002,53202,\
  53404,53596,53784,53974,54162,54346,54534,54720,54908,55078,55250,55422,55596,\
  55750,55908,56062,56218,56360,56500,56638,56780,56912,57050,57186,57320,57446,\
  57570,57692,57818,57942,58066,58188,58308,58416,58526,58636,58746,58854,58962,\
  59070,59180,59286,59396,59506,59612,59704,59802,59894,59986,60078,60170,60264,\
  60352,60432,60510,60584,60664,60742,60818,60898,60976,61052,61130,61208,61282,\
  61356,61428,61498,61566,61632,61692,61756,61818,61878,61940,62002,62066,62126,\
  62188,62252,62314,62374,62434,62496,62554,62612,62672,62736,62782,62832,62894,\
  62952,63014,63060,63108,63152,63200,63246,63292,63338,63386,63428,63476,63522,\
  63568,63614,63660,63708,63756,63804,63846,63894,63940,63986,64032,64076,64122,\
  64170,64210,64250,64508,64764,65022,65278,65535])
  lut=GlobalToneCurveY/256
  
  src_dir='./data/train'
  save_dir = './data/train_png'
  src_dir_test='./data/test'
  save_dir_test = './data/test_png'
  
  filepaths = glob.glob(src_dir + '/*.jpg')
  filepaths_test = glob.glob(src_dir_test + '/*.jpg')
  def sortKeyFunc(s):
    return int(os.path.basename(s)[:-4])
    
  filepaths_test.sort(key=sortKeyFunc)
  filepaths.sort(key=sortKeyFunc)
  
  
  print("[*] Reading train files...")  
  
  if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        os.mkdir(save_dir_test)
        os.mkdir('./data/train_png/noisy')
        os.mkdir('./data/train_png/original')
        os.mkdir('./data/test_png/noisy')
        os.mkdir('./data/test_png/original')        

  print("[*] Applying noise...")

  sig = np.linspace(0,50,len(filepaths))
  np.random.shuffle(sig)
  sig_test = np.linspace(0,50,len(filepaths_test))
  np.random.shuffle(sig_test)

#  fix = cv2.imread(filepaths[1]).shape[1]
  j = 0
  for i in xrange(len(filepaths)):
        image = cv2.imread(filepaths[i])
        image = cv2.resize(image,(180,180), interpolation = cv2.INTER_CUBIC)
        row,col,ch = image.shape
        mean = 0
        sigma = sig[i]
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype('uint8')
#        noisy = cv2.LUT(noisy, lut)        
#        image = cv2.LUT(image, lut)  
        cv2.imwrite(os.path.join(save_dir, "noisy/%04d.png" %j), noisy)
        cv2.imwrite(os.path.join(save_dir, "original/%04d.png" %j), image)
        j = j+1
#        gauss = np.random.normal(mean,sigma,(row,col,ch))
#        gauss = gauss.reshape(row,col,ch)
#        noisy = image + gauss
#        noisy = np.clip(noisy, 0, 255)
#        noisy = noisy.astype('uint8')
# 
#        cv2.imwrite(os.path.join(save_dir, "noisy/%04d.png" %j), noisy)
#        cv2.imwrite(os.path.join(save_dir, "original/%04d.png" %j), image)
#        j = j+1
#        gauss = np.random.normal(mean,sigma,(row,col,ch))
#        gauss = gauss.reshape(row,col,ch)
#        noisy = image + gauss
#        noisy = np.clip(noisy, 0, 255)
#        noisy = noisy.astype('uint8')
# 
#        cv2.imwrite(os.path.join(save_dir, "noisy/%04d.png" %j), noisy)
#        cv2.imwrite(os.path.join(save_dir, "original/%04d.png" %j), image)
#        j = j+1
        
        
  for i in xrange(len(filepaths_test)):
        image = cv2.imread(filepaths_test[i])
        image = cv2.resize(image,(180,180), interpolation = cv2.INTER_CUBIC)
        row,col,ch = image.shape
        mean = 0
        sigma = sig[i]
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        noisy = noisy.astype('uint8')
#        noisy = cv2.LUT(noisy, lut)
#        image = cv2.LUT(image, lut)  
        
        cv2.imwrite(os.path.join(save_dir_test, "noisy/%d.png" %i), noisy)
        cv2.imwrite(os.path.join(save_dir_test, "original/%d.png" %i), image)

  filepaths = glob.glob('./data/train_png/noisy' + '/*.png')
  filepaths.sort(key=sortKeyFunc)
  im_array = np.array( [cv2.imread(img) for img in filepaths] )  
  np.save(os.path.join('./data/train_png/noisy', "noisy_total_train"), im_array)
  
  print("[*] Noisy and original images saved")

         
if __name__ == "__main__":
 main()
