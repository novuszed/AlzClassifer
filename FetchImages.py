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
        #print(file)
        try:
            fileSize = ftp.size(file)
        except ftplib.error_perm as resp:
            if str(resp) == "550 Could not get file size.":
                print("File size error!")
                continue
        #print(fileSize)
        print("Fetching file %s of size" % file, fileSize)
        ftp.retrbinary("RETR " + file ,open(file, 'wb').write,1024,FtpDownloadTracker(fileSize))
        print("Finish fetching file %s" % file)
    ftp.quit()

#Decompress tar.gz and tar files
def unzip(tarName):

    tarFiles = set([f for f in os.listdir(os.getcwd()) if os.path.isfile(f) and tarName in f])

    for fname in tarFiles:
        if fname.endswith("tar.gz"):
            tar = tarfile.open(fname,"r:gz")
            tar.extractall()
            tar.close()
        elif fname.endswith("tar"):
            tar = tarfile.open(fname,"r:")
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

    dementiaRatings = [0,0.5,1,2,3,sys.maxsize]
    for count,rating in enumerate(dementiaRatings):
        if(count == len(dementiaRatings)-1):
            print("Finished graphing dementia statistics")
            return
        noDementia = df[(df.DementiaScale>=dementiaRatings[count]) & (df.DementiaScale<dementiaRatings[count+1])]

        gender = ["Male","Female"]
        amount = [str(len(noDementia[noDementia['M/F']=='M']))+" Males",str(len(noDementia[noDementia['M/F']=='F']))+" Females"]
        plt.pie([len(noDementia[noDementia['M/F']=='M']),len(noDementia[noDementia['M/F']=='F'])], labels=['M','F'], colors = colors, autopct='%1.1f%%')
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

        if not os.path.exists(os.getcwd()+'/Graphs'):
            os.makedirs(os.getcwd()+'/Graphs')
        plt.savefig('%s/Graphs/genderSplit.png' %os.getcwd())




headerLegend = "MR Session,Subject,M/F,Hand,Age,Educ,SES,CDR,MMSE,eTIV,nWBV,ASF,Scans"
print(headerLegend.split(","))
newHeaderLegend = ['Session', 'Subject', 'M/F', 'Hand', 'Age', 'Educ', 'SocioEcon', 'DementiaScale', 'MiniMentalExam',
                   'intracranVol', 'wholeBranVol', 'AtlasScaling', 'Scans']
print(newHeaderLegend)
csvFileName = 'patientList.csv'

df = pd.read_csv(csvFileName, skiprows=1, na_values=['NaN'],
                 names=['Session', 'Subject', 'M/F', 'Hand', 'Age', 'Educ', 'SocioEcon', 'DementiaScale',
                        'MiniMentalExam', 'intracranVol', 'wholeBranVol', 'AtlasScaling', 'Scans'])
# Filling empty dementia with default (no dementia)
df['DementiaScale'].fillna(value=0, inplace=True)
print(df)

unzip(tarFileName)
# genderVisual(df)
#downloadImagesGz("test")

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

# downloadImages()
