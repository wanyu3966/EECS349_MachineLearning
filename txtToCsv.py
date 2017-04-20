import csv

def readTXT(path):
    fileList = []
    with open(path) as csvfile:
        reader = csv.reader(csvfile, delimiliter = ' ')
        index = 0
        for row in reader:
            fileList.insert(index, row)
            index += 1
            print row
    return fileList

def main():
    readTXT

if __name__ == "__main__":
    sys.exit(main())
