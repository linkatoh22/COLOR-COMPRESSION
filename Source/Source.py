import numpy as np
import PIL
from PIL import Image 
import matplotlib.pyplot as plt
import sys
import math
def img2_1d(filename,height,width,channels):
    img=Image.open(filename)
    img=np.array(img)
    img=img.reshape(height*width,channels)
    return img
def ran_centroid(k_clusters,img_1d,init_centroid):
    centroids = []
    if(init_centroid=='random'):
        for x in range(k_clusters):
            centroids.append(np.random.randint(0,255,3))
    elif init_centroid == 'in_pixels':
        pix = np.unique(img_1d, axis=0)
        ind = np.random.choice(len(pix), size=k_clusters, replace=False)
        centroids = pix[ind].tolist()
    return centroids
def update_pixel(img_1d, height, width, channel, centroids, labels):
  for i in range(len(img_1d)):
    img_1d[i] = centroids[labels[i]]
  img_1d = img_1d.reshape(height, width, channel)
  return img_1d
def K_mean(img_1d,k_clusters,max_iter,init_centroid):
    centroids=ran_centroid(k_clusters,img_1d,init_centroid)
    labels = [None for x in range(len(img_1d))]
    for i in range(max_iter):
        centroidscpy=centroids.copy()
        temp=[]
        for x in range(k_clusters):
            temp.append([])
        for i in range(len(img_1d)):
            distances=np.linalg.norm(centroids-img_1d[i], axis=1) 
            labels[i]=np.argmin(distances)
            temp[labels[i]].append(i)
        for i, cluster in enumerate(temp):
            if len(cluster) > 0:
                mean = np.mean(img_1d[cluster], axis=0)
                centroids[i] = mean
        if np.array_equal(centroids,centroidscpy):
            break
    return centroids,labels
def save_pic(outfile,init_centroid,k_cluster):
    output=input("Enter image's output format (png or pdf):")
    while(output!='pdf' and output!='png'):
        output=input("Invalid type!. PLEASE ENTER AGAIN (png or pdf):")
    print("Image's output format:"+output)
    PIL.Image.fromarray(outfile[0], 'RGB').save('orginal' + '.' + output)
    hinh=1
    for i in init_centroid:
        for j in k_cluster:
            PIL.Image.fromarray(outfile[hinh], 'RGB').save(i+'_'+str(j) + '.' + output)
            hinh+=1
def main():
    outfile=[]
    max_iter=1000
    k_cluster=[3,5,7]
    init_centroid=['random','in_pixels']
    filename=input("Enter image name or path:")

    img=Image.open(filename)
    img=np.array(img)
    outfile.append(img)
    height,width,channels=img.shape
    for i in init_centroid:
        for j in k_cluster:
            img_1d=img2_1d(filename,height,width,channels)
            print("Clustering right now....")
            centroid,labels=K_mean(img_1d,j,max_iter,i)
            print("Done clustering image with",j,"clusters and init_centroid:",i)
            img_1d = update_pixel(img_1d, height, width, channels, centroid, labels)
            outfile.append(img_1d)
    save_pic(outfile,init_centroid, k_cluster)  
main()


