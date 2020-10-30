import numpy as np
import matplotlib.pyplot as plt
_DEBUG = True # show the figure when you are running or not

def myprint(obj):
    if _DEBUG == False:
        return
    else:
        print(obj)

def showOutlier(answer,x=[]):
    if _DEBUG == False:
        return
    if x == []:
        plt.figure()
        plt.plot(answer,"+")
        plt.show()
    else:
        n1 = x[-1]-x[0]
        n2 = np.max(answer)-np.min(answer)
        plt.figure(figsize=[n1,n2])
        plt.plot(x,answer,"+")
        plt.show()

def showMatrix_center(matrix,Y,score_HBOS,truth_list,color="red",ii=None):
    truth_list = truth_list.reshape(-1)
    x1 = np.argwhere(truth_list == 0)
    x2 = np.argwhere(truth_list == 1)
    y1 = Y[truth_list==0]#DBScan
    y2 = Y[truth_list==1]#DBScan
    z1 = score_HBOS[truth_list==0]
    z2 = score_HBOS[truth_list==1]
    w1 = matrix[1, truth_list == 0]  # bin
    w2 = matrix[1, truth_list == 1]  # bin
    v1 = matrix[0, truth_list == 0]  # bin
    v2 = matrix[0, truth_list == 1]

    ax = plt.subplot(223)  # 创建一个三维的绘图工程
    ax.set_title('After sliding windows')  # 设置本图名称
    ax.scatter(x1, y1, alpha=0.1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax.scatter(x2, y2, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax.set_xlabel('position')  # 设置x坐标轴
    ax.set_ylabel('DBScan')  # 设置z坐标轴

    ax1 = plt.subplot(224)  # 创建一个三维的绘图工程
    ax1.set_title('Score')  # 设置本图名称
    ax1.scatter(x1, z1, alpha=0.1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax1.scatter(x2, z2, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax1.set_xlabel('position')  # 设置x坐标轴
    ax1.set_ylabel('types')  # 设置z坐标轴

    ax2 = plt.subplot(222)  # 创建一个三维的绘图工程
    ax2.set_title('position-Type')  # 设置本图名称
    ax2.scatter(x1, w1, alpha=0.1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax2.scatter(x2, w2, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax2.set_xlabel('position')  # 设置x坐标轴
    ax2.set_ylabel('means')  # 设置z坐标轴

    ax2 = plt.subplot(221)  # 创建一个三维的绘图工程
    ax2.set_title('position-RD')  # 设置本图名称
    ax2.scatter(x1, v1, alpha=0.1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax2.scatter(x2, v2, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax2.set_xlabel('position')  # 设置x坐标轴
    ax2.set_ylabel('RD')  # 设置z坐标轴

    if _DEBUG == False:
        plt.savefig("/home/ubunt1604/Desktop/GYwork2/img/" + str(ii) + ".png")
    else:
        plt.show()
    plt.close()

def showMatrix_6Img(matrix, truth_list,color="red",ii=None):
    truth_list = truth_list.reshape(-1)
    x1 = np.argwhere(truth_list == 0)
    x2 = np.argwhere(truth_list == 1)
    w1 = matrix[0,truth_list==0]#
    w2 = matrix[0,truth_list==1]#
    z1 = matrix[1,truth_list==0]#mean
    z2 = matrix[1,truth_list==1]#mean
    y1 = matrix[2,truth_list==0]#DBScan
    y2 = matrix[2,truth_list==1]#DBScan
    h1 = matrix[3,truth_list==0]#type
    h2 = matrix[3,truth_list==1] #type

    ax = plt.subplot(231)  # 创建一个三维的绘图工程
    ax.set_title('3d_image_show')  # 设置本图名称
    ax.scatter(x1, y1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax.scatter(x2, y2, alpha=0.3, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax.set_xlabel('position')  # 设置x坐标轴
    ax.set_ylabel('DBScan')  # 设置z坐标轴

    ax1 = plt.subplot(232)  # 创建一个三维的绘图工程
    ax1.set_title('3d_image_show')  # 设置本图名称
    ax1.scatter(x1, z1, alpha=1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax1.scatter(x2, z2, alpha=0.3, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax1.set_xlabel('position')  # 设置x坐标轴
    ax1.set_ylabel('means')  # 设置z坐标轴

    ax2 = plt.subplot(233)  # 创建一个三维的绘图工程
    ax2.set_title('3d_image_show')  # 设置本图名称
    ax2.scatter(h1, y1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax2.scatter(h2, y2, alpha=0.3, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax2.set_ylabel('means')  # 设置y坐标轴
    ax2.set_xlabel('type')  # 设置z坐标轴

    ax3 = plt.subplot(234)  # 创建一个三维的绘图工程
    ax3.set_title('3d_image_show')  # 设置本图名称
    ax3.scatter(x1, h1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax3.scatter(x2, h2, alpha=0.3, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax3.set_ylabel('type')  # 设置y坐标轴
    ax3.set_xlabel('position')  # 设置z坐标轴

    ax3 = plt.subplot(235)  # 创建一个三维的绘图工程
    ax3.set_title('3d_image_show')  # 设置本图名称
    ax3.scatter(x1, w1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax3.scatter(x2, w2, alpha=0.3, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax3.set_ylabel('RD')  # 设置y坐标轴
    ax3.set_xlabel('type')  # 设置z坐标轴

    ax3 = plt.subplot(236)  # 创建一个三维的绘图工程
    ax3.set_title('3d_image_show')  # 设置本图名称
    ax3.scatter(y1, z1, c='black', s=3)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax3.scatter(y2, z2, alpha=0.3, c=color, s=5)  # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
    ax3.set_ylabel('means')  # 设置y坐标轴
    ax3.set_xlabel('DBScan')  # 设置z坐标轴
    if _DEBUG == False:
        plt.savefig("/home/ubunt1604/Desktop/GYwork2/img/" + str(ii) + ".png")
    else:
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()
    plt.close()

def showMatrix_3D(matrix,truth_list,color="red",ii=None):
    truth_list = truth_list.reshape(-1)
    x1 = np.argwhere(truth_list == 0)
    x2 = np.argwhere(truth_list == 1)
    z1 = matrix[1,truth_list==0]#mean
    z2 = matrix[1,truth_list==1]#mean
    y1 = matrix[3,truth_list==0]#DBScan
    y2 = matrix[3,truth_list==1]#DBScan
    ax = plt.subplot(projection='3d')
    ax.scatter(z1, x1, y1, c='black', s=3)
    ax.scatter(z2, x2, y2, c=color, s=5)
    ax.set_ylabel('position')  # 设置y坐标轴
    ax.set_xlabel('DBScan')  # 设置x坐标轴
    ax.set_zlabel('mean')  # 设置z坐标轴
    if _DEBUG:
        plt.savefig("/home/ubunt1604/Desktop/GYwork2/img/" + str(ii) + ".png")
    else:
        plt.show()

def hist(data):
    if _DEBUG == False:
        return
    plt.figure()
    plt.bar(x=range(len(data)), height=data)
    plt.show()

def Write_Score_File(chr,index,RD,Y,binSize,sorce,filename):
    """
    write cnv score file
    pos start, pos end, ReadDepth, RD after sliding Window, sorce
    """
    output = open(filename, "w")
    output.write(
        "chrom" + '\t' + "start" + '\t' + "end" + '\t' + "ReadDepth" + '\t'+ "slidingWindow" + '\t' + "score" + '\n')
    for i in range(len(index)):
        begin = int(index[i])*binSize
        end = int(index[i])*binSize+binSize
        rd = int(RD[i])
        y = float(Y[i])
        output.write("chr" + str(chr) + '\t' + str(begin) + '\t' + str(end) + '\t' + str(rd) + '\t'+ str(y) + '\t' + str(sorce[i]) + '\n')


def Write_CNV_File(chr, CNVstart, CNVend, CNVtype, CN, filename):
    """
    write cnv result file
    pos start, pos end, type, copy number
    """
    output = open(filename, "w")
    for i in range(len(CNVtype)):
        if CNVtype[i] == 2:
            output.write("chr" + str(chr[i]) + '\t' + str(CNVstart[i]) + '\t' + str(
                CNVend[i]) + '\t' + str("gain") + '\t' + str(CN[i]) + '\n')
        else:
            output.write("chr" + str(chr[i]) + '\t' + str(CNVstart[i]) + '\t' + str(
                CNVend[i]) + '\t' + str("loss") + '\t' + str(CN[i]) + '\n')



