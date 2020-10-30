# HBOS-CNV
HBOS-CNV: a new approach to detect copy number variations from next-generation sequencing data

## Installation
The following software must be installed on your machine:

Python : tested with version 3.7

Python >= 3.7

### Python dependencies
* numpy
* sklearn
* pysam
* matplotlib

You can install the above package using the following command：

pip install numpy sklearn  pysam matplotlib


## Running
HBOS-CNV requires two input files, a bam file after sort and a reference folder,
the folder contains the reference sequence for each chromosome of the bam file.

Note: at present, the approach only supports single chromosome samples, like chr5, chrX...
The whole genomes can be detected by batch file
### runnig command
python HBOS-CNV.py [reference] [bamfile] [binSize] [GroundTruth]

[reference]: the reference folder path

[bamfile]: a bam file after sort

[binSize]: the window size ('1000'by default)

[GroundTruth]: Optional, if you are doing a simulation test, you can add the file of correct mutation location.

### run the default example
python HBOS-CNV.py ./reference/ ./test.sort.bam 1000 ./GroundTruthCNV

Finally, two result files will be generated in the folder called result, one is the mutation information of detection and the other is the score of HBOS-CNV

## About Visualization
if you want to visually see the test results,You have two choices:
* In the file showTool.py, change the parameter which named _DEBUG to True. In this way, you can get the figure when you are running.
* In the file showTool.py, change the parameter which named _DEBUG to Talse. In this way, the program will save the figure in folder named img.
