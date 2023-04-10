#Komparasi 1

import cv2
from matplotlib import pyplot as plt
from natsort import natsorted
import os
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
import datetime
import time as waktu
import pandas as pd

def similarity_index(img1, img2):
    """
    Menghitung skor kesamaan citra (Similarity Index) antara dua citra.
    
    Args:
        img1 (numpy array): Citra pertama dalam format numpy array.
        img2 (numpy array): Citra kedua dalam format numpy array.
        
    Returns:
        skor kesamaan citra antara img1 dan img2 dalam rentang [0, 1].
    """
    # Memastikan ukuran citra sama
    assert img1.shape == img2.shape, "Ukuran citra tidak sama"
    
    # Konversi tipe data citra menjadi float
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    # Menghitung rata-rata dan varians dari masing-masing citra
    mean1 = np.mean(img1)
    mean2 = np.mean(img2)
    var1 = np.var(img1)
    var2 = np.var(img2)
    
    # Menghitung kovarians antara kedua citra
    covar = np.mean((img1 - mean1) * (img2 - mean2))
    
    # Menghitung Similarity Index
    num = 2 * mean1 * mean2
    den1 = mean1**2 + mean2**2
    den2 = var1 + var2
    sim_index = num / (den1 * den2) * (2 * covar)
    
    return sim_index



PATH = 'preprocessed_image/'
RESULTS = 'result/comparation/'
HISTOGRAM_RESULTS = 'result/histogram/'

hasil_evaluasi = []

for filename in natsorted(os.listdir(PATH)):
    time_start = waktu.time()
    img = cv2.imread(PATH + filename, 1)
    print(filename)
    # converting to LAB color space
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    ''' Applying HE '''
    # Konversi gambar ke HSV
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Equalisasi histogram pada channel V
    img_hsv[:,:,2] = cv2.equalizeHist(img_hsv[:,:,2])

    # Konversi kembali ke RGB
    img_equalized = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    ''' Applying CLAHE to L-channel '''
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Stacking the original image with the enhanced image
    result = np.hstack((img, img_equalized, enhanced_img))
    # cv2_imshow(result)
    #   cv2.imwrite(RESULTS+filename, result)



    ''' Showing Histogram '''
    original_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    he_gray = cv2.cvtColor(img_equalized, cv2.COLOR_BGR2GRAY)
    clahe_gray = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    #   plt.hist(original_gray.flat, bins=100, range=(0,255), color='blue', label='original')
    #   plt.hist(he_gray.flat, bins=100, range=(0,255), color='orange', label='HE')
    #   plt.hist(clahe_gray.flat, bins=100, range=(0,255), color='green', label='CLAHE')
    #   plt.xlabel('Value')
    #   plt.ylabel('Frequency')
    #   plt.title('Histogram Comparation')

    #   plt.legend()
    #   plt.savefig(HISTOGRAM_RESULTS+filename)
    #   plt.show()

    # Load gambar asli dan gambar hasil enhancement
    img_asli = original_gray
    img_he = he_gray
    img_clahe = clahe_gray

    # Hitung nilai SSIM (Structural Similarity Index)
    ssim_score = ssim(img_asli, img_he, multichannel=True)

    # Hitung nilai MSE (Mean Squared Error)
    mse_score = mse(img_asli, img_he)

    # Hitung nilai IQI (Image Quality Index)
    iqi_score = similarity_index(img_asli, img_he)

    # Hitung nilai RMSE (Root Mean Squared Error)
    rmse_score = np.sqrt(mse_score)

    # Hitung nilai PSNR (Peak Signal-to-Noise Ratio)
    psnr_score = psnr(img_asli, img_he)


    # Hitung nilai SSIM (Structural Similarity Index)
    ssim_score_clahe = ssim(img_asli, img_clahe, multichannel=True)

    # Hitung nilai MSE (Mean Squared Error)
    mse_score_clahe = mse(img_asli, img_clahe)

    # Hitung nilai IQI (Image Quality Index)
    iqi_score_clahe = similarity_index(img_asli, img_clahe)

    # Hitung nilai RMSE (Root Mean Squared Error)
    rmse_score_clahe = np.sqrt(mse_score_clahe)

    # Hitung nilai PSNR (Peak Signal-to-Noise Ratio)
    psnr_score_clahe = psnr(img_asli, img_clahe)

    time_end = waktu.time()
    time_stop = time_end - time_start

    hasil_evaluasi.append([filename, ssim_score, mse_score, iqi_score, rmse_score, psnr_score, ssim_score_clahe, mse_score_clahe, iqi_score_clahe, rmse_score_clahe, psnr_score_clahe, time_stop])

    read_hasil_evaluasi = [nr[1:11] for nr in hasil_evaluasi]
ssim_score_mean = np.mean(read_hasil_evaluasi[0])
mse_score_mean = np.mean(read_hasil_evaluasi[1])
iqi_score_mean = np.mean(read_hasil_evaluasi[2])
rmse_score_mean = np.mean(read_hasil_evaluasi[3])
psnr_score_mean = np.mean(read_hasil_evaluasi[4])

ssim_score_clahe_mean = np.mean(read_hasil_evaluasi[5])
mse_score_clahe_mean = np.mean(read_hasil_evaluasi[6])
iqi_score_clahe_mean = np.mean(read_hasil_evaluasi[7])
rmse_score_clahe_mean = np.mean(read_hasil_evaluasi[8])
psnr_score_clahe_mean = np.mean(read_hasil_evaluasi[9])
duration_total = np.sum(read_hasil_evaluasi[10])

hasil_evaluasi.append([f"SIMPULAN HASIL", f"SSIM HE: {ssim_score_mean:0.3f}", f" MSE HE: {mse_score_mean:0.3f}", f" IQI HE: {iqi_score_mean:0.3f}", f"RMSE HE: {rmse_score_mean:0.3f}", f" PSNR HE: {psnr_score_mean:0.3f}", f"SSIM CLAHE: {ssim_score_clahe_mean:0.3f}", f" MSE CLAHE: {mse_score_clahe_mean:0.3f}", f" IQI CLAHE: {iqi_score_clahe_mean:0.3f}", f"RMSE CLAHE: {rmse_score_clahe_mean:0.3f}", f" PSNR CLAHE: {psnr_score_clahe_mean:0.3f}", f"Duration Total: {duration_total:0.3f}"])
df = pd.DataFrame(hasil_evaluasi, columns=["Nama Citra", "SSIM HE", "MSE HE", "IQI HE", "RMSE HE", "PRSNR HE", "SSIM CLAHE", "MSE CLAHE", "IQI CLAHE", "RMSE CLAHE", "PRSNR CLAHE", "Duration"])
df.to_csv("result/hasil_perbaikan_iqi.csv")



    # # Cetak hasil penghitungan
    # print("-----------------Nilai HE----------------")
    # print('Nilai SSIM: ', ssim_score)
    # print('Nilai MSE: ', mse_score)
    # print('Nilai IQI: ', iqi_score)
    # print('Nilai RMSE: ', rmse_score)
    # print('Nilai PSNR: ', psnr_score)

    # # Cetak hasil penghitungan
    # print("-----------------Nilai CLAHE----------------")
    # print('Nilai SSIM: ', ssim_score_clahe)
    # print('Nilai MSE: ', mse_score_clahe)
    # print('Nilai IQI: ', iqi_score_clahe)
    # print('Nilai RMSE: ', rmse_score_clahe)
    # print('Nilai PSNR: ', psnr_score_clahe)


