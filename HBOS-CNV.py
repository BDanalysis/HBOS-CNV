import warnings
import numpy as np
import sys
from mytime import MyTimer as Time
from sklearn.cluster import DBSCAN
from readTool import read_bam_file,read_ref_file,read_truth_file,Binning
import showTool as swT
from methods import find_od_HBOS,HB_cluster
warnings.simplefilter("ignore")

def calculate_answer(label):
    label = label.reshape(-1)
    index1 = np.argwhere(label==1)
    if len(index1)==0:
        answer1 = [[0,0]]
        return answer1
    begin = index1[0]
    answer1 = []
    for i in range(len(index1)):
        if i == len(index1)-1:
            if index1[i]-index1[i-1]<=1:
                answer1.append([int(begin), int(index1[i])])
            else:
                answer1.append([int(index1[i]), int(index1[i])])
        elif index1[i+1]-index1[i]<=5:
            continue
        else:
            answer1.append([int(begin),int(index1[i]+1)])
            begin = index1[i+1]
    return np.array(answer1)

def calculate_CNV_point(perp,Y,binSize,all_index):
    length = len(perp)
    num_bins = np.shape(Y)[0]
    pery_list = np.zeros(num_bins)
    index = all_index.reshape(-1)
    types = []
    points = []
    prey_mid = []
    for i in range(length):
        start = perp[i][0]
        end = perp[i][1]
        pery_list[start:end] = 1
    normal = Y[pery_list==0]
    balance = np.mean(normal)
    CNV_type = np.ones(num_bins)
    for i in range(num_bins):
        if pery_list[i]==1:
            if balance >= Y[i]:
                CNV_type[i] = 0
            else:
                CNV_type[i] = 2
    sum = 0
    num = 0
    ####swT.showOutlier(CNV_type)
    for i in range(num_bins-1):
        if CNV_type[i] != 1:
            sum = sum + Y[i]
            num = num + 1
        if CNV_type[i] != CNV_type[i+1] and num != 0:
            mean = float(sum/num)
            begin = index[i + 1 - num]
            end = index[i+1]
            points.append([begin * binSize + 1, end * binSize])
            prey_mid.append([i + 1 - num, i+1])
            if mean >= balance:
                types.append(2)
            else:
                types.append(0)
            sum = 0
            num = 0
    prey_mid = np.array(prey_mid)
    points = np.array(points)
    types = np.array(types)
    return points,types,CNV_type,prey_mid


def ensure_bad_pery(perp,RD):
    length = len(perp)
    pery_list = np.zeros(RD.shape[0])
    for i in range(length):
        start = perp[i][0]
        end = perp[i][1]
        pery_list[start:end+1]=1
    return pery_list

def ensure_bad_bin(truth_start,truth_end,RD,binSize):
    length = len(truth_start)
    truth_list = np.zeros(np.shape(RD))
    top = len(RD)
    for i in range(length):
        start = np.ceil(truth_start[i]/binSize).astype(int)
        end = np.ceil(truth_end[i]/binSize).astype(int)
        if start <= top and end <= top:
            truth_list[start:end]=1
    return truth_list

def gcCheck(RD,GC):
    RD = RD/1000
    RD = RD.reshape(-1)
    GC = GC.reshape(-1)
    #RD[RD == 0] = modeRD(RD)
    RD = gc_correct(RD, GC)
    return RD*1000

def modeRD(RD):
    newRD = np.full(len(RD), 0)
    for i in range(len(RD)):
        newRD[i] = int(np.round(RD[i], 3) * 1000)

    count = np.bincount(newRD)
    countList = np.full(len(count) - 49, 0)
    for i in range(len(countList)):
        countList[i] = np.mean(count[i:i + 50])
    modemin = np.argmax(countList)
    modemax = modemin + 50
    mode = (modemax + modemin) / 2
    mode = mode / 1000
    return mode

def gc_correct(RD, GC):
    # correcting gc bias
    bincount = np.bincount(GC)
    global_rd_ave = np.mean(RD)
    for i in range(len(RD)):
        if bincount[GC[i]] < 2:
            continue
        mean = np.mean(RD[GC == GC[i]])
        if RD[i] != 0:
            RD[i] = global_rd_ave * RD[i] / mean
    #print(np.argwhere(np.isnan(RD)))
    return RD

def alignment(RD, GC, truth_list):
    index = np.argwhere(RD>=0)
    RD = RD[index]
    GC = GC[index]
    truth_list = truth_list[index]
    return RD,GC,truth_list,index

def normalization(data):
    domain = np.nanmax(data)-np.nanmin(data)
    data = data - np.nanmin(data)
    data = data/domain
    length = len(data)
    mid = length/domain
    return mid,data.reshape(1,-1)

def add_position(data):
    length = len(data)
    step = 0.02
    end = 1+length*step
    x = np.arange(1, end, step).reshape(-1)
    x = x[0:length]
    answer = np.c_[data, x]
    return step,answer

def makeMatrix(RD,unnormal_RD,truth_list):#GY
    RD = RD.T
    KEY = 30
    lens = RD.shape[0]
    matrix = np.zeros((2, lens))
    for i in range(lens):
        matrix[0][i] = unnormal_RD[i]
    floor = int(lens / KEY)
    for i in range(floor + 1):
        if (lens - i * KEY) <= KEY:
            matrix[1][(i - 1) * KEY:lens] = means(RD[(i - 1) * KEY:lens])
        else:
            matrix[1][i * KEY:(i + 1) * KEY] = means(RD[i * KEY:(i + 1) * KEY])

    data = matrix[1, 0:lens]
    eps, d = HB_cluster(data, 100)
    step, data = add_position(data)
    label,matrix[1][0:lens] = DBScan(data, 2, round(np.sqrt(d ** 2 + step ** 2), 2))  # +step**2
    return truth_list,matrix

def means(RD):
    means = np.mean(RD)
    ones = np.ones_like(RD)
    answer = ones*means
    return answer.reshape(1,-1)

def calculating_CN(prey,RD,CNVRD, CNVtype):
    CNVstart = prey[:, 0]
    CNVend = prey[:, 1]
    CN = np.full(len(CNVtype), 0)
    homoRD = np.mean(RD[CNVRD == 0]) if len(RD[CNVRD == 0])!=0 else 0
    hemiRD = np.mean(RD[CNVRD == 2]) if len(RD[CNVRD == 2])!=0 else 0
    purity = 2 * (homoRD - hemiRD) / (homoRD - 2 * hemiRD)
    for i in range(len(CNVtype)):
        begin = CNVstart[i]
        end = CNVend[i]
        error = np.mean(RD[begin:end])
        CN[i] = int(2*error / (mode * purity) - 2 * (1-purity) / purity)
    return CN

def DBScan(data,KEY,_eps):
    X = data
    db = DBSCAN(eps=_eps, min_samples=KEY).fit(X)
    labels = db.labels_
    K = len(set(labels)) - (1 if -1 in labels else 0)
    answer = labels.astype(float)
    bal = X.T
    for i in range(K):
        balance = np.mean(bal[0][labels==i])
        answer = np.where(labels == i, balance, answer)
    answer = np.where(labels == -1, bal[0], answer)
    return labels,answer

refpath = sys.argv[1]
binSize = int(sys.argv[3])
bam = sys.argv[2]
groundTruth = sys.argv[4]
outpath = "./result"
t = Time()

try:
    truth_start, truth_end = read_truth_file(groundTruth)
except IOError:
    truth_start = []
    truth_end = []

t.start(outpath+"/"+bam)
mid = bam.split("/")
mid = mid[-1].split(".")[0]
outfile = outpath + '/'+mid+".result.txt"
outsorcefile = outpath + '/' + mid +".sorce.txt"
ref = [[] for i in range(23)]
try:
    refList = read_bam_file(bam)
except ValueError:
    print("ValueError:"+bam+"may be the file is error")
    sys.exit(1)
for i in range(len(refList)):
    chr = refList[i]
    chr_num = chr.strip('chr')
    if chr_num.isdigit():
        chr_num = int(chr_num)
        reference = refpath + '/chr' + str(chr_num) + '.fa'
        ref = read_ref_file(reference, chr_num, ref)
chrLen = np.full(23, 0)
for i in range(1, 23):
    chrLen[i] = len(ref[i])
RD,GC,out_chr = Binning(ref, binSize, chrLen, bam)
truth_list = ensure_bad_bin(truth_start,truth_end,RD,binSize)
RD, GC, truth_list,all_index = alignment(RD, GC, truth_list)
unnormal_RD = gcCheck(RD,GC)
middle,normal_RD = normalization(unnormal_RD)
truth_list,matrix = makeMatrix(normal_RD,unnormal_RD,truth_list)
print(np.sum(truth_list))
window = 30
score_HBOS,Y,pre_labels = find_od_HBOS(matrix,window)

pery = calculate_answer(pre_labels)
pery_list = ensure_bad_pery(pery,RD)
swT.showMatrix_center(matrix,Y,score_HBOS,truth_list, "red", str(mid) + "T")  # jj->savefig
swT.showMatrix_center(matrix,Y,score_HBOS,pery_list, "green", str(mid) + "P")  # jj->savefig

pery_finally,types,CNV_type,prey_mid = calculate_CNV_point(pery,Y,binSize,all_index)
mode = np.mean(RD)
CNVRD = CNV_type
try:
    CNVstart = pery_finally[:,0]
    CNVend = pery_finally[:,1]
except IndexError:
    print(out_chr)
    sys.exit(1)

if len(prey_mid) != 0:
    CN = calculating_CN(prey_mid,RD,CNVRD, types)
else:
    CN = np.zeros_like(CNVend)
ones = np.ones_like(CNVend)*out_chr
swT.Write_CNV_File(ones,CNVstart,CNVend,types,CN,outfile)
swT.Write_Score_File(out_chr,all_index,RD,Y,binSize,score_HBOS,outsorcefile)
t.stop()







