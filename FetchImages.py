__author__ = 'zihaozhu'
import csv
import numpy as np
import sys
import urllib
import ftplib
import pandas as pd
import os
import matplotlib.pyplot as plt
import tarfile
import re
import shutil
import collections
import errno
from PIL import Image
from PIL import ImageOps

sessionBuckets = [range(1, 43), range(43, 81), range(81, 116), range(116, 151), range(151, 192), range(192, 232),
                  range(232, 273), range(273, 310), range(310, 349), range(349, 383), range(383, 420), range(420, 458)]

tarFileName = "oasis_cross-sectional_disc"


# Tracks Ftp download progress
class FtpDownloadTracker:
    sizeWritten = 0
    totalSize = 0
    lastShownPercent = 0

    def __init__(self, totalSize):
        self.totalSize = totalSize

    def handle(self, block):
        self.sizeWritten += 1024
        percentComplete = round((self.sizeWritten / self.totalSize) * 100)

        if (self.lastShownPercent != percentComplete):
            self.lastShownPercent = percentComplete
            print(str(percentComplete) + " percent complete")


# Download GZ compressed image sets through FTP
def downloadImagesGz(filename):
    # path = os.getcwd()+"/"+patientName
    path = "data"
    files = []
    ftp = ftplib.FTP("ftp.nrg.wustl.edu")
    ftp.login("anonymous", "zihaozhu96@gmail.com")
    ftp.cwd(path)
    # ftp.retrlines('LIST')
    try:
        files = ftp.nlst()
    except ftplib.error_perm as resp:
        if str(resp) == "550 No files found":
            print("No files in this directory")
        else:
            raise
    for file in files:
        # print(file)
        try:
            fileSize = ftp.size(file)
        except ftplib.error_perm as resp:
            if str(resp) == "550 Could not get file size.":
                print("File size error!")
                continue
        # print(fileSize)
        print("Fetching file %s of size" % file, fileSize)
        ftp.retrbinary("RETR " + file, open(file, 'wb').write, 1024, FtpDownloadTracker(fileSize))
        print("Finish fetching file %s" % file)
    ftp.quit()


# Decompress tar.gz and tar files
def unzip(tarName):
    tarFiles = set([f for f in os.listdir(os.getcwd()) if os.path.isfile(f) and tarName in f])

    for fname in tarFiles:
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname, "r:gz")
            tar.extractall()
            tar.close()
        elif fname.endswith("tar"):
            tar = tarfile.open(fname, "r:")
            tar.extractall()
            tar.close()
        else:
            print("Error in unzipping compressed files")
            return


# Generate pie chart of men/women without dementia
def genderVisual(df):
    colors = ["#5DA5DA", "#FAA43A"]
    print(len(df))
    print(len(df[df.DementiaScale < 0.5]))
    print(len(df[df.DementiaScale > 0]))

    dementiaRatings = [0, 0.5, 1, 2, 3, sys.maxsize]
    for count, rating in enumerate(dementiaRatings):
        if (count == len(dementiaRatings) - 1):
            print("Finished graphing dementia statistics")
            return
        noDementia = df[(df.DementiaScale >= dementiaRatings[count]) & (df.DementiaScale < dementiaRatings[count + 1])]

        gender = ["Male", "Female"]
        amount = [str(len(noDementia[noDementia['M/F'] == 'M'])) + " Males",
                  str(len(noDementia[noDementia['M/F'] == 'F'])) + " Females"]
        plt.pie([len(noDementia[noDementia['M/F'] == 'M']), len(noDementia[noDementia['M/F'] == 'F'])],
                labels=['M', 'F'], colors=colors, autopct='%1.1f%%')
        plt.axis('equal')

        plt.legend(amount, loc="best")

        if rating == 0:
            plt.title("% Male/Female without dementia")
        elif rating == 0.5:
            plt.title("% Male/Female with very mild dementia")
        elif rating == 1:
            plt.title("% Male/Female with mild dementia")
        elif rating == 2:
            plt.title("% Male/Female with moderate dementia")
        else:
            plt.title("% Male/Female with severe dementia")

        plt.tight_layout()
        plt.show()

        if not os.path.exists(os.getcwd() + '/Graphs'):
            os.makedirs(os.getcwd() + '/Graphs')
        plt.savefig('%s/Graphs/genderSplit.png' % os.getcwd())

#Generate graph comparing dementia rate and age
def AgeVsDementia(df):
    AgeVsDemen = dict()
    for index,row in df.iterrows():
        if row['DementiaScale']>=0.5:
            AgeVsDemen[row['Age']]=AgeVsDemen.get(row['Age'],1)+1


    AgeVsDemen = collections.OrderedDict(sorted(AgeVsDemen.items()))

    AgeVD=list(AgeVsDemen.items())

    Ages = tuple(a[0] for a in AgeVD)
    DementiaCounts = tuple(a[1] for a in AgeVD)

    N = np.arange(len(Ages))
    plt.bar(N,DementiaCounts,align='center', alpha=0.5)
    plt.xticks(N, Ages)
    plt.ylabel("# of those with dementia")
    plt.xlabel("Ages")
    plt.title("# Of patients with dementia vs age")
    plt.show()

    if not os.path.exists(os.getcwd() + '/Graphs'):
        os.makedirs(os.getcwd() + '/Graphs')
    plt.savefig('%s/Graphs/AgeDementia.png' % os.getcwd())

# Clean up folders to save spaces after extracting needed images
def cleanUpFolders():
    for i in range(1, 13):
        path = os.getcwd() + "/disc%d" % i
        if os.path.exists(path):
            print("Removing path %s " % path)
            shutil.rmtree(path)

# Clean up all the GZ files to save room
def cleanUpGz(tarName):
    tarFiles = set([f for f in os.listdir(os.getcwd()) if
                    os.path.isfile(f) and tarName in f and (f.endswith("tar.gz") or f.endswith("tar"))])
    for fname in tarFiles:
        try:
            print("Removing %s " % (os.getcwd() + "/%s" % fname))
            # os.remove(os.getcwd()+"/%s" %fname)
        except:
            raise ("Can't delete GZ files")

#Converting png files to jpg files. Tensorflow prefers Jpg
def pngToJpg():
    path = os.getcwd() + "/tf_files/alzClass/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            # print(image[:-3])
            img = Image.open(path + image)
            img.save(path + img[:-3] + "jpeg")
    path = os.getcwd() + "/tf_files/NoAlzClass/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            # print(image[:-3])
            img = Image.open(path + image)
            img.save(path + img[:-3] + "jpeg")

#Convert png files to Gif files, converting it back to compare learning rate between png and jpeg
def pngToGif():
    path = os.getcwd() + "/tf_files/alzClass/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            fileName = image[:-3] + "gif"
            os.rename(path + image, path + fileName)
    path = os.getcwd() + "/tf_files/NoAlzClass/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            fileName = image[:-3] + "gif"
            os.rename(path + image, path + fileName)

    path = os.getcwd()+"/Verification/Alz/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            fileName = image[:-3] + "gif"
            os.rename(path + image, path + fileName)

    path = os.getcwd()+"/Verification/NoAlz/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            fileName = image[:-3] + "gif"
            os.rename(path + image, path + fileName)



#Extracting the first frame of the gif file and save it as jpeg
def extractGifFrame():
    path = os.getcwd() + "/tf_files/alzClass/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            if not image.endswith("gif"):
                continue
            im = Image.open(path + image)
            bg= ImageOps.invert(im)

            #bg = Image.new("RGB", im.size, (255, 255, 255))
            #bg.paste(im, (0, 0), im)
            newPath = path + image
            bg.save(newPath[:-3] + "jpg")

    path = os.getcwd()+"/tf_files/NoAlzClass/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            if not image.endswith("gif"):
                continue
            im = Image.open(path + image)
            bg= ImageOps.invert(im)

            #bg = Image.new("RGB", im.size, (255, 255, 255))
            #bg.paste(im, (0, 0), im)
            newPath = path + image
            bg.save(newPath[:-3] + "jpg")
"""
    path = os.getcwd()+"/Verification/Alz/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            if not image.endswith("gif"):
                continue
            im = Image.open(path + image)
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, (0, 0), im)
            newPath = path + image
            bg.save(newPath[:-3] + "jpg", quality=99)


    path = os.getcwd()+"/Verification/NoAlz/"
    if os.path.exists(path + ".DS_Store"):
        os.remove(path + ".DS_Store")
    for dir, subdir, images in os.walk(path):
        for image in images:
            if not image.endswith("gif"):
                continue
            im = Image.open(path + image)
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, (0, 0), im)
            newPath = path + image
            bg.save(newPath[:-3] + "jpg", quality=99)
"""
#Use disc12 for inspection of training accuracy
def testAccuracy(df):
    dementiaPatients = df[(df.DementiaScale>=0.5)]
    noDementiaPatients = df[(df.DementiaScale<0.5)]
    alzPatient=[]
    noAlzPatient=[]

    if not os.path.exists(os.getcwd() + '/Verification'):
        os.makedirs(os.getcwd() + '/Verification')
        os.makedirs(os.getcwd()+'/Verification/Alz')
        os.makedirs(os.getcwd()+'/Verification/NoAlz')

    for index, row in dementiaPatients.iterrows():
        patient = int(row['Subject'].split("_")[1])
        path = os.getcwd() + "/%s" % row['Session']
        try:
            if not os.path.exists(os.getcwd() + "/Alz/%s" % row['Session']):
                shutil.copytree(path, os.getcwd() + "/Alz/%s" % row['Session'])
        except OSError as exc:  # python >2.5
            if exc.errno == errno.ENOTDIR:
                shutil.copy(path, os.getcwd() + "/Alz/%s" % row['Session'])
            else:
                raise ("Error in moving files to Alz folder")
        try:
            for dir, subdir, images in os.walk(os.getcwd() + "/Alz/%s" %row['Session']):
                if len(images) == 0:
                    continue
                for image in images:
                    path = dir + "/%s" % image
                    shutil.copyfile(path, os.getcwd() + "/Verification/Alz/%s" % image)
        except:
            raise ("Error in moving files to Verification Alz folder")

    for index,row in noDementiaPatients.iterrows():
        patient = int(row['Subject'].split("_")[1])
        path = os.getcwd() + "/%s" % row['Session']
        try:
            if not os.path.exists(os.getcwd() + "/NoAlz/%s" % row['Session']):
                shutil.copytree(path, os.getcwd() + "/NoAlz/%s" % row['Session'])
        except OSError as exc:  # python >2.5
            if exc.errno == errno.ENOTDIR:
                shutil.copy(path, os.getcwd() + "/NoAlz/%s" % row['Session'])
            else:
                raise ("Error in moving files to NoAlz folder")
        try:
            for dir, subdir, images in os.walk(os.getcwd() + "/NoAlz/%s" %row['Session']):
                if len(images) == 0:
                    continue
                for image in images:
                    path = dir + "/%s" % image
                    shutil.copyfile(path, os.getcwd() + "/Verification/NoAlz/%s" % image)
        except:
            raise ("Error in moving files to Verification No Alz folder")


#Separate images based on whether or not someone has dementia or not
def separatePatients(df):
    dementiaPatients = df[(df.DementiaScale >= 0.5)]
    noDementiaPatients = df[(df.DementiaScale < 0.5)]

    if not os.path.exists(os.getcwd() + '/Alz'):
        os.makedirs(os.getcwd() + '/Alz')
    if not os.path.exists(os.getcwd() + '/NoAlz'):
        os.makedirs(os.getcwd() + '/NoAlz')

    for index, row in dementiaPatients.iterrows():
        path = os.getcwd() + "/%s" % row['Session']
        if int(row['Subject'].split("_")[1])>419: #Ignoring disc 12 for now as a reserve
            continue
        try:
            if not os.path.exists(os.getcwd() + "/Alz/%s" % row['Session']):
                shutil.copytree(path, os.getcwd() + "/Alz/%s" % row['Session'])
        except OSError as exc:  # python >2.5
            if exc.errno == errno.ENOTDIR:
                shutil.copy(path, os.getcwd() + "/Alz/%s" % row['Session'])
            else:
                raise ("Error in moving files to Alz folder")

    for index, row in noDementiaPatients.iterrows():
        path = os.getcwd() + "/%s" % row['Session']
        if int(row['Subject'].split("_")[1])>419: #Ignoring disc 12 for now as a reserve
            continue
        try:
            if not os.path.exists(os.getcwd() + "/NoAlz/%s" % row['Session']):
                shutil.copytree(path, os.getcwd() + "/NoAlz/%s" % row['Session'])
        except OSError as exc:
            if exc.errno == errno.ENOTDIR:
                shutil.copy(path, os.getcwd() + "/NoAlz/%s" % row['Session'])
            else:
                raise ("Error in moving files to NoAlz folder")

#Move all image for alz patients into tf_files for learning
def extractForTfFiles():
    if not os.path.isdir(os.getcwd() + "/tf_files"):
        os.mkdir(os.getcwd() + "/tf_files")
    if not os.path.isdir(os.getcwd() + "/tf_files/alzClass"):
        os.mkdir(os.getcwd() + "/tf_files/alzClass")
    for dir, subdir, images in os.walk(os.getcwd() + "/Alz"):
        if len(images) == 0:
            continue
        for image in images:
            path = dir + "/%s" % image
            shutil.copyfile(path, os.getcwd() + "/tf_files/alzClass/%s" % image)

#Move all images for nonAlz patients into tf_files for learning
def extractNoAlzForTfFiles():
    if not os.path.isdir(os.getcwd() + "/tf_files"):
        os.mkdir(os.getcwd() + "/tf_files")
    if not os.path.isdir(os.getcwd() + "/tf_files/NoAlzClass"):
        os.mkdir(os.getcwd() + "/tf_files/NoAlzClass")
    for dir, subdir, images in os.walk(os.getcwd() + "/NoAlz"):
        if len(images) == 0:
            continue
        for image in images:
            path = dir + "/%s" % image
            # print(path)
            # print(image)
            shutil.copyfile(path, os.getcwd() + "/tf_files/NoAlzClass/%s" % image)

#Removing all the gif files to only inspect jpeg files
def cleanGifFiles():
    if not os.path.isdir(os.getcwd()+"/tf_files/NoAlzClass"):
        raise("NoAlzClass not created")
    if not os.path.isdir(os.getcwd()+"/tf_files/alzClass"):
        raise("alzClass not created")
    path = os.getcwd()+"/tf_files/alzClass/"

    for dir, subdir, images in os.walk(path):
        if len(images)==0:
            continue
        for image in images:
            if image.endswith("gif"):
                os.remove(path+image)
    path = os.getcwd()+"/tf_files/NoAlzClass/"
    for dir, subdir, images in os.walk(path):
        if len(images)==0:
            continue
        for image in images:
            if image.endswith("gif"):
                os.remove(path+image)
    path = os.getcwd()+"/Verification/Alz/"
    for dir, subdir, images in os.walk(path):
        if len(images)==0:
            continue
        for image in images:
            if image.endswith("gif"):
                os.remove(path+image)
    path = os.getcwd()+"/Verification/No/Alz"
    for dir, subdir, images in os.walk(path):
        if len(images)==0:
            continue
        for image in images:
            if image.endswith("gif"):
                os.remove(path+image)

# Extracting particular photo images and organizing them based off of patients
# Discard uncessary data and delete them after to clean up
def extractPhotos():
    # Check if files have been extracted correctly and create folders separating Alz and no Alz
    for i in range(1, 13):
        if not os.path.isdir(os.getcwd() + "/disc%d" % i):
            raise Exception("No directories have been extracted!")

    path = os.getcwd()

    # Walks through each path and extracts needed photos and convert them from gif to png
    # currently using folder 12 for 10% comparison to verify prediction
    for i in range(1, 13):
        for patientFolder, dirs, files in os.walk(path + '/disc%d' % i):

            fsegFiles = [os.path.join(patientFolder, fseg) for fseg in files if
                         "fseg_tra" in fseg and fseg.endswith("gif")]
            T88_111 = [os.path.join(patientFolder, T88_111) for T88_111 in files if
                       "111_t88" in T88_111 and T88_111.endswith("gif") and "fseg" not in T88_111]
            sbj_111 = [os.path.join(patientFolder, sbj_111) for sbj_111 in files if "sbj_111_sag_88.gif" in sbj_111]

            if len(fsegFiles) != 0:
                patientNumber = re.search(r'.*/(OAS.*?)(?=/)', fsegFiles[0], re.M).group(1)

                if not os.path.exists(os.getcwd() + '/%s' % patientNumber):
                    os.makedirs(os.getcwd() + '/%s' % patientNumber)
                if not os.path.exists(os.getcwd() + '/%s/fseg' % patientNumber):
                    os.makedirs(os.getcwd() + '/%s/fseg' % patientNumber)
                for fname in fsegFiles:
                    #fileName = fname.split("/")[-1][:-3] + "png"
                    #print(fileName)
                    fileName = fname.split("/")[-1]
                    #print(fileName)
                    os.rename(fname, os.getcwd() + '/%s' % patientNumber + '/fseg/%s' % fileName)

            if len(T88_111) != 0:
                patientNumber = re.search(r'.*/(OAS.*?)(?=/)', T88_111[0], re.M).group(1)
                if not os.path.exists(os.getcwd() + '/%s' % patientNumber):
                    os.makedirs(os.getcwd() + '/%s' % patientNumber)
                if not os.path.exists(os.getcwd() + '/%s/T88_111' % patientNumber):
                    os.makedirs(os.getcwd() + '/%s/T88_111' % patientNumber)
                for fname in T88_111:
                    #fileName = fname.split("/")[-1][:-3] + "png"
                    fileName = fname.split("/")[-1]
                    os.rename(fname, os.getcwd() + '/%s' % patientNumber + '/T88_111/%s' % fileName)

            if len(sbj_111) != 0:
                patientNumber = re.search(r'.*/(OAS.*?)(?=/)', sbj_111[0], re.M).group(1)
                if not os.path.exists(os.getcwd() + '/%s' % patientNumber):
                    os.makedirs(os.getcwd() + '/%s' % patientNumber)
                if not os.path.exists(os.getcwd() + '/%s/sbj_111' % patientNumber):
                    os.makedirs(os.getcwd() + '/%s/sbj_111' % patientNumber)
                for fname in sbj_111:
                    #fileName = fname.split("/")[-1][:-3] + "png"
                    fileName = fname.split("/")[-1]
                    os.rename(fname, os.getcwd() + '/%s' % patientNumber + '/sbj_111/%s' % fileName)

    return
def removeOASFolders():
    for folder in os.listdir(os.getcwd()):
        if "OAS1_" in folder and ("MR1" in folder or "MR2" in folder):
            shutil.rmtree(folder)

#Parsing the original CSV with patient data
def parsePatientCSV():
    headerLegend = "MR Session,Subject,M/F,Hand,Age,Educ,SES,CDR,MMSE,eTIV,nWBV,ASF,Scans"
    # print(headerLegend.split(","))
    newHeaderLegend = ['Session', 'Subject', 'M/F', 'Hand', 'Age', 'Educ', 'SocioEcon', 'DementiaScale', 'MiniMentalExam',
                       'intracranVol', 'wholeBranVol', 'AtlasScaling', 'Scans']
    # print(newHeaderLegend)
    csvFileName = 'patientList.csv'

    df = pd.read_csv(csvFileName, skiprows=1, na_values=['NaN'],
                     names=['Session', 'Subject', 'M/F', 'Hand', 'Age', 'Educ', 'SocioEcon', 'DementiaScale',
                            'MiniMentalExam', 'intracranVol', 'wholeBranVol', 'AtlasScaling', 'Scans'])
    # Filling empty dementia with default (no dementia)
    df['DementiaScale'].fillna(value=0, inplace=True)
    return df

#Convert everything to jpeg before training
def classifyJpeg(dataframe):
    unzip(tarFileName)
    extractPhotos()

    separatePatients(dataframe)
    extractForTfFiles()
    extractNoAlzForTfFiles()
    testAccuracy(dataframe)
    pngToGif()
    extractGifFrame()
    cleanGifFiles()

#Function that bundels clean up
def cleanUp():
    cleanUpGz(tarFileName)
    cleanUpFolders()
    removeOASFolders()

#Convert everything to png and run tests
def classifyPng(dataframe):
    unzip(tarFileName)
    extractPhotos()

    separatePatients(dataframe)
    extractForTfFiles()
    extractNoAlzForTfFiles()
    testAccuracy(dataframe)

def convertPngToJpeg():
    extractForTfFiles()
    extractNoAlzForTfFiles()
    pngToGif()
    extractGifFrame()
    cleanGifFiles()

dataframe = parsePatientCSV()
#cleanUp()
#classifyPng(dataframe)
convertPngToJpeg()

#AgeVsDementia(dataframe)
# genderVisual(df)
# downloadImagesGz("test")

#print(dataframe)

"""
Old way of dealing with CSV
with open(csvFileName, newline='') as patientList:
    reader = csv.DictReader(patientList)
    try:
        for row in reader:
            print([row[field] for field in headerLegend.rstrip().split(',')])
    except csv.Error as e:
        sys.exit('file {}, line {}: {}'.format(csvFileName, reader.line_num, e))
"""
#pngToJpg()
#pngToGif()
#extractGifFrame()
#cleanGifFiles()
# downloadImages()

#removeOASFolders()
