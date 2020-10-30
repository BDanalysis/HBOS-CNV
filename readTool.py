import os
import pysam
import numpy as np
import gc
def read_bam_file(filename):
    samfile = pysam.AlignmentFile(filename, "rb")
    chrList = samfile.references
    return chrList

def read_truth_file(filename):
    truth_start = []
    truth_end = []
    with open(filename, 'r') as f:
        line = f.readline()
        for line in f:
            linestr = line.strip('\n')
            linestrlist = linestr.split('\t')
            truth_start.append(int(linestrlist[0]))
            truth_end.append(int(linestrlist[1]))
    return truth_start , truth_end

def read_ref_file(filename, chr_num, ref):
    # read reference file
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            line = f.readline()
            for line in f:
                linestr = line.strip()
                ref[chr_num] += linestr
    else:
        print("Warning: can not open " + str(filename) + '\n')
    return ref


def Binning(ref, binSize, chrLen, filename):#importent
    chrTag = np.full(23, 0)
    chrList = np.arange(23)
    maxNum = int(chrLen.max() / binSize) + 1
    InitRD = np.full((23, maxNum), 0.0)
    print(maxNum)
    # read bam file and get bin rd
    # print("Read bam file: " + str(filename))
    samfile = pysam.AlignmentFile(filename, "rb")
    for line in samfile:
        idx = int(line.pos / binSize)
        if line.reference_name:
            chr = line.reference_name.strip('chr')
            if chr == 'X' or chr == 'x':
                chr = str(23-1)
            if chr.isdigit():
                InitRD[int(chr)][idx] += 1
                chrTag[int(chr)] = 1
    chrList = chrList[chrTag > 0]
    chrNum = len(chrList)
    InitGC = np.full((chrNum, maxNum), 0)
    pos = np.full((chrNum, maxNum), 0)

    # initialize bin_data and bin_head
    count = 0
    out_chr = 0
    for i in range(len(chrList)):
        chr = chrList[i]
        binNum = int(chrLen[chr] / binSize) + 1
        for j in range(binNum):
            pos[i][j] = j
            cur_ref = ref[chr][j * binSize:(j + 1) * binSize]
            N_count = cur_ref.count('N') + cur_ref.count('n')
            if N_count == 0:
                gc_count = cur_ref.count('C') + cur_ref.count('c') + cur_ref.count('G') + cur_ref.count('g')
            else:
                gc_count = 0
                InitRD[chr][j] = -1000000
                count = count + 1
            InitGC[i][j] = int(round(gc_count / binSize, 3) * 1000)
        # delete
        cur_RD = InitRD[chr][:binNum]
        cur_GC = InitGC[i][:binNum]
        out_chr = chr
    del InitRD, InitGC, pos
    gc.collect()# jvm-gc
    return cur_RD, cur_GC,out_chr