import numpy as np
np.set_printoptions(suppress=True)
import showTool as swT
from sklearn.cluster import KMeans

def del_main(DATA):
    DATA = DATA.reshape(-1)
    length = len(DATA)
    ones = np.ones(length)
    mid = np.mean(DATA)
    index = np.argwhere(DATA>=mid).reshape(-1)
    ones[index] = 0
    id = index[0]

    for i in range(0,id):
        if DATA[id-i-1]-DATA[id-i] < 1:
            ones[id-i] = 0
        else:
            break
    id = index[-1]
    for i in range(id,length-1):
        if DATA[i+1]-DATA[i] < 1:
            ones[i+1] = 0
        else:
            break
    answer = DATA*ones
    #print("+++++++++++++++++++++")
    return answer

def del_after_main(DATA):
    DATA = DATA.reshape(-1)
    length = len(DATA)
    ones = np.ones(length)
    mid = np.mean(DATA)
    index = np.argwhere(DATA > mid).reshape(-1)
    ones[index] = 0
    id = index[0]
    for i in range(0, id):
        if DATA[id - i - 1] - DATA[id - i] < 1:
            ones[id - i] = 0
        else:
            break
    id = index[-1]
    for i in range(id, length - 1):
        if DATA[i + 1] - DATA[i]< 1:
            ones[i + 1] = 0
        else:
            break
    answer = DATA * ones
    return answer

def data_gauss(numbers,windows,sigma):
    numbers = numbers.reshape(1,-1)
    bin_num = numbers.shape[1]
    floor = (windows//2)
    zero = np.zeros([1,windows//2])
    numbers = np.c_[zero,numbers]
    numbers = np.c_[numbers,zero]
    numbers = numbers.reshape(-1)
    answers = np.zeros(bin_num)
    for ii in range(floor,bin_num+floor):
            answers[ii-floor] = np.sum(numbers[ii-floor:ii+floor]*gauss(windows,sigma))
    return answers

def gauss(kernel_size, sigma):
    kernel = np.zeros(kernel_size)
    center = kernel_size // 2
    if sigma <= 0:
        sigma = ((kernel_size - 1) * 0.5 - 1) * 0.3 + 0.8
    s = sigma ** 2
    sum_val = 0
    for i in range(kernel_size):
        x = i - center
        kernel[i] = np.exp(-(x ** 2) / 2 * s)
        sum_val += kernel[i]
    kernel = kernel / sum_val
    #swT.showOutlier(kernel)
    return kernel

def find_od_HBOS(matrix,window):
    matrix = matrix.reshape(2,-1)
    w = matrix[0,:]#RD
    x = matrix[1,:]#Means
    wid = int(np.sqrt(len(w)))//100
    labels = HB_simple(x, wid*100)
    types = set(labels)
    part1_pred = np.zeros(len(x)).reshape(-1)
    mm1 = np.zeros(len(x)).reshape(-1)
    mm2 = np.zeros(len(x)).reshape(-1)
    mm3 = np.zeros(len(x)).reshape(-1)
    for tp in types:
        index = np.argwhere(labels == tp).reshape(-1)
        h_bin = int(np.sqrt(len(index)))
        W = w[index].reshape(1,-1)
        X = x[index].reshape(1, -1) #mean
        Y = moveWindow(W,window,1)#30 ,1
        main_labels = makeHBlist_nor(Y,h_bin)
        score , move_labels = makeHBlist(Y,h_bin)
        if len(types) == 1:
            aaa = delet_error(X,int(np.sqrt(len(Y))))
        else:
            aaa = 1
        if np.std(X) < 0.01:
            main_labels = 0
        all = move_labels*aaa*main_labels
        error_index = index[all>=1]
        mm1[index] = labels[index]
        mm2[index] = Y
        mm3[index] = score
        part1_pred[error_index] = 1
    score = mm3
    Y = mm2
    answer = part1_pred
    print(np.sum(answer))
    return score,Y,answer

def moveWindow(RD,length,step=1):
    RD = RD.reshape(-1)
    if step < 1:
        step = 1
    lens = len(RD)
    answer = np.zeros(lens)
    i = 0
    while i < lens:
        if i < length // 2:
            mid = RD[0:i+1]
            weight = gauss(len(mid),0.3)
            answer[i:i+step] = np.sum(mid*weight)
            #answer[i:i + step] = RD[i]
        elif (lens - i) <= length:
            mid = RD[i:lens]#lens - length
            weight = gauss(len(mid), 0.3)
            answer[i:i+step] = np.sum(mid*weight) # 2*j-len-1:len  gauss(RD[len-KEY_mean:len],100,1)
        else:
            mid = RD[i-(length//2):i+(length//2)]
            weight = gauss(len(mid), 0.3)
            answer[i:i+step] = np.sum(mid*weight)  # gauss(RD[i-(KEY_mean//2):i+(KEY_mean//2)],100,1)
        i = i + step
    return answer



def HB_cluster(data,bin_num):
    bins = np.linspace(0,1,bin_num).reshape(-1)
    answer = np.zeros_like(bins)
    X_mid = data
    for ii in range(len(bins)):
        X_mid = X_mid - 1/bin_num
        index = np.argwhere(X_mid>0).reshape(-1)
        fq = len(X_mid) - len(index)
        X_mid = X_mid[index]
        answer[ii] = fq
    ####swT.hist(answer)
    means = len(data)/(bin_num)
    index = np.argwhere(answer>means).reshape(-1)
    mids = []
    mid = index[0]
    for ii in range(1,len(index)):
        if index[ii]-index[ii-1] == 1:
            if ii == len(index):
                mid = (index[ii - 1] - mid)//2+1
                mids.append(mid)
            continue
        else:
            mid = (index[ii - 1] - mid)//2+1
            mids.append(mid)
            mid = index[ii]
    mid_arr = np.array(mids)
    if len(mid_arr)==0:
        De=[0.03]
    else:
        De = bins[mid_arr]
    min_d = 2
    for ii in range(len(De)):
        min_d = De[ii] if min_d > De[ii] else min_d
    min1 = np.min(answer[np.argwhere(answer>0).reshape(-1)])
    return min1,min_d

def marge_del_1(labels):
    labels = labels.reshape(-1)
    types = set(labels)
    if len(types) <= 1:
        return labels
    nums = np.ones(len(types))
    num_type = np.zeros(len(types))
    i = 0
    for ii in types:
        num_type[i] = ii
        index = np.argwhere(labels==ii).reshape(-1)
        nums[i] = len(index)
        i = i+1
    nums = np.argsort(nums)

    swT.showOutlier(labels)
    change = num_type[nums[1]]
    ii = np.argwhere(labels==num_type[nums[0]]).reshape(-1)[0]
    for jj in range(ii,len(labels)):
        if labels[jj] != num_type[nums[0]]:
            change = labels[jj]
            break
    labels = np.where(labels==num_type[nums[0]],change,labels)
    ####swT.showOutlier(labels)
    types = set(labels)
    for ii in types:
        begin = []
        index = np.argwhere(labels==ii).reshape(-1)
        for j in range(1,len(index)):
            if begin != []:
                if index[j] - index[j - 1] == 1:
                    begin.append(index[j])
                if j == len(index)-1:
                    part = (begin[-1] - begin[0]) / len(index)
                    if part<0.1:
                        labels[begin[0]:begin[-1] + 1] = labels[begin[0] - 1]
                    begin = []
                else:
                    part = (begin[-1] - begin[0]) / len(index)
                    if part < 0.1:
                        labels[begin[0]:begin[-1] + 1] = labels[begin[0] - 1]
                    begin = []
                    begin.append(index[j])
            else:
                if index[j] - index[j - 1] != 1:
                    begin.append(index[j])
    swT.showOutlier(labels)
    return labels

def HB_simple(data,bin_num):
    X = data.reshape(-1)
    fqs = np.zeros(bin_num)
    X_mid = X
    for ii in range(bin_num):
        X_mid = X_mid - 1 / bin_num
        index = np.argwhere(X_mid > 0).reshape(-1)
        fq = len(X_mid) - len(index)
        X_mid = X_mid[index]
        fqs[ii] = fq
    ####swT.hist(fqs)
    means = len(X) / (bin_num*1.4)
    index = np.argwhere(fqs > means).reshape(-1)
    fq_means = (np.max(fqs[index])+np.min(fqs[index]))//2.5
    mmid = np.zeros_like(X)
    for ii in index:
        begin = 1/bin_num * ii
        end = 1/bin_num *(ii+1)
        indexs = np.argwhere((X <= end) & (X > begin)).reshape(-1)
        mmid[indexs] = 1
    xmid = (X * mmid).reshape(-1,1)
    index1 = np.argwhere((fqs>fq_means)).reshape(-1)

    for ii in range(1,len(index1)):
        if index1[ii]-index1[ii-1]==1:
            index1[ii-1] = -1
    K = len(index1[np.argwhere(index1>0)])+1#(0)
    Kmean = KMeans(K).fit(xmid)
    labels = Kmean.labels_
    labels = marge_del_1(labels)
    #swT.showOutlier(labels)
    return labels

def makeHBlist(data,bin_num):
    X = data.reshape(-1)
    X_sorce = np.zeros_like(X)
    X_sort = np.sort(X)
    X_sort_index = np.argsort(X)
    HBlist = np.zeros((3,bin_num+1))
    length_pre = int(len(X)/bin_num)
    for i in range(bin_num+1):
        if (i+1)*length_pre >= len(X):
            if X_sort[-1] == X_sort[i*length_pre] and X_sort[i*length_pre]==0:
                HBlist[0][i] = length_pre*5
            else:
                HBlist[0][i] = X_sort[-1] - X_sort[i*length_pre]+1
            HBlist[1][i] = len(X)-(i-1)*length_pre
            break
        else:
            if X_sort[i*length_pre] == X_sort[(i+1)*length_pre-1] and X_sort[i*length_pre]==0:
                HBlist[0][i] = length_pre*5
            else:
                HBlist[0][i] = X_sort[(i+1)*length_pre-1] - X_sort[i*length_pre]+1
            HBlist[1][i] = length_pre
    sorce = HBlist[1]/HBlist[0]
    sorce = np.log10(1/sorce)
    sorce = normalization(sorce)
    ####swT.hist(sorce)
    for i in range(bin_num+1):
        if (i+1)*length_pre >= len(X):
            index = X_sort_index[i*length_pre:-1]
            X_sorce[index] = sorce[i]
            break
        else:
            index = X_sort_index[i*length_pre : (i+1)*length_pre-1]
            X_sorce[index] = sorce[i]
    #X_sorce = normalization(X_sorce)
    sorce_max = np.max(X_sorce)
    std = np.std(X_sorce)
    threshold_ = sorce_max*0.6
    if std>0.08:
        print("====================")
        print(sorce_max)
        print(threshold_)
        print(std)
        print("====================")
    ####swT.showOutlier(X_sorce)
    label_ = np.where(X_sorce>threshold_,1,0)
    return X_sorce,label_

def normalization(data):
    data = data - np.min(data)
    return data.reshape(-1)


def makeHBlist_nor(data,bin_num):
    X = data.reshape(-1)
    length = np.max(X)-np.min(X)
    step = length/bin_num
    HBlist = np.zeros(bin_num)
    X_mid = X - np.min(X)
    answer = np.zeros_like(X)
    for ii in range(bin_num):
        X_mid = X_mid - step
        index = np.argwhere(X_mid > 0).reshape(-1)
        fq = len(X_mid) - len(index)
        X_mid = X_mid[index]
        HBlist[ii] = fq
    ####swT.hist(HBlist.reshape(-1))
    HBlist = data_gauss(HBlist, bin_num // 10 * 2, 2)
    HBlist = del_main(HBlist)##change

    ####swT.hist(HBlist.reshape(-1))
    index = np.argwhere(HBlist > 0)
    for ii in index:
        eara_start = ii*step + np.min(X)
        eara_end = ii*step+np.min(X)+step
        id = np.argwhere((X >= eara_start) & (X <= eara_end))
        answer[id] = 1
    ####swT.showOutlier(answer)
    return answer

def delet_error(data,bin_num):
    X = data.reshape(-1)
    length = np.max(X)-np.min(X)
    step = length/bin_num
    HBlist = np.zeros(bin_num)
    X_mid = X - np.min(X)
    answer = np.ones_like(X)
    for ii in range(bin_num):
        X_mid = X_mid - step
        index = np.argwhere(X_mid > 0).reshape(-1)
        fq = len(X_mid) - len(index)
        X_mid = X_mid[index]
        HBlist[ii] = fq

    HBlist = data_gauss(HBlist, bin_num // 10 * 2, 2)
    HBlist = del_after_main(HBlist)
    index = np.argwhere(HBlist==np.max(HBlist))
    ####swT.hist(HBlist.reshape(-1))
    for ii in index:
        eara_start = ii*step + np.min(X)
        eara_end = ii*step+np.min(X)+step
        id = np.argwhere((X >= eara_start) & (X <= eara_end))
        answer[id] = 0
    return answer
