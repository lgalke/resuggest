import fileinput
import numpy as np

def wc(line):
    return len(line.strip().split())

if __name__ == '__main__':
    wc = list(wc(line) for line in fileinput.input())
    wc = np.array(wc)
    print("Mean word count per line", wc.mean())
    print("Std deviation:", wc.std())
    print("Minimum length:", wc.min())
    print("Maximum length:", wc.max())

