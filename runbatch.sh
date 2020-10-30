#!/bin/bash
python HBOS-CNV.py ./reference/ ./chr1.sort.bam 1000 ./GroundTruthCNV
python HBOS-CNV.py ./reference/ ./chr2.sort.bam 1000 ./GroundTruthCNV
python HBOS-CNV.py ./reference/ ./chr3.sort.bam 1000 ./GroundTruthCNV
python HBOS-CNV.py ./reference/ ./chr4.sort.bam 1000 ./GroundTruthCNV
python HBOS-CNV.py ./reference/ ./chr5.sort.bam 1000 ./GroundTruthCNV
python HBOS-CNV.py ./reference/ ./chr6.sort.bam 1000 ./GroundTruthCNV

