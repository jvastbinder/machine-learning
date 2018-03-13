import csv


def main(infile, outfile):
    dataIn = open(infile, "r")
    dataOut = open(outfile, "w")

    dataLst = []

    dataStart = int(dataIn.readline())
    numFeatures = int(dataIn.readline())
    numYs = int(dataIn.readline())
    documentation = dataIn.readline()

    dataOut.write(str(dataStart) + "\n")
    dataOut.write(str(numFeatures) + "\n")
    dataOut.write(str(numYs) + "\n")
    dataOut.write(str(documentation))

    line = dataIn.readline()
    m = 0
    sniffer = csv.Sniffer()
    position = dataIn.tell()
    delimiter = sniffer.sniff(dataIn.read()).delimiter
    dataIn.seek(position)

    while line:
        line = dataIn.readline()
        m += 1
        if line != "":
            if line[len(line)-1] == '\n':
                line = line[:len(line)-1]
            lineLst = line.split(delimiter)
            for i in range(len(lineLst)):
                lineLst[i] = float(lineLst[i])

            dataLst.append(lineLst)

    minMaxAvg = []
    for i in range(numFeatures):
        minMaxAvg.append([float("inf"), float("-inf"), 0])

    for lst in dataLst:
        for i in range(dataStart, dataStart + numFeatures):
            if float(lst[i]) < minMaxAvg[i - dataStart][0]:
                minMaxAvg[i - dataStart][0] = float(lst[i])
            if float(lst[i]) > minMaxAvg[i - dataStart][1]:
                minMaxAvg[i - dataStart][1] = float(lst[i])
            minMaxAvg[i - dataStart][2] += float(lst[i])

    ranges = []
    for i in range(len(minMaxAvg)):
        minMaxAvg[i][2] /= m
        ranges.append(minMaxAvg[i][1] - minMaxAvg[i][0])

    for lst in dataLst:
        line = ""
        for i in range(len(lst)):
            if (i >= dataStart) and (i < dataStart + numFeatures):
                lst[i] -= minMaxAvg[ i- dataStart][2]
                lst[i] /= ranges[i - dataStart]

            line += str(lst[i]) + delimiter
        line = str(line[:len(line)-1]) + "\n"
        dataOut.write(line)

    dataIn.close()
    dataOut.close()


main("data.txt", "outfile.txt")




