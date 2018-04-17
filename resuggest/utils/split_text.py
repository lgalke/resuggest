import re
from argparse import ArgumentParser
from collections import Counter
import sys

def path_split(path):
    parts = re.split(r'/', path)
    prefix = ""
    for path_chunk in parts[:-1]:
        prefix += path_chunk + "/"

    # TODO: basename and further file ending = all till first "." --- prior ending(s) become next attached before by "_"
    postfix = re.split(r'\.',parts[-1])
    further_ending = postfix[0]
    new_core_name = ""
    for name_chunk in postfix[1:]:
        new_core_name += "_" + name_chunk
    return prefix,new_core_name, further_ending

def traing_test_split(filenames, percent,parallel=False):
    # TODO: randomisation for a pair of files for parallel text format
    for filename in filenames:
        prefix,new_core_name,new_ending =path_split(filename)
        print("processing {}".format(filename))
        count = sum(1 for _ in open(filename))
        split = min(int(count/ 100) * percent, count)
        print("{} total sentences".format(count))
        with open(filename) as f:
            train_file_name = '{}train{}.{}'.format(prefix,new_core_name,new_ending)
            with open(train_file_name,"w") as train_file:
                for i in range(split):
                    train_file.write(f.readline())
                print("{} in {}".format(i+1,train_file_name))
            test_file_name = '{}test{}.{}'.format(prefix,new_core_name,new_ending)
            with open(test_file_name,"w") as test_file:
                for j,line in enumerate(f.readlines()):
                    test_file.write(line)
                print("{} in {}".format(j,test_file_name))




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        'files',
        nargs='*',
        type = str,
        help='file(s) to be counted')
    parser.add_argument(
        '-p',
        '--percent',
        type=int,
        help='percent at which infrequent words are cut off')

    args = parser.parse_args()
    #voca_file(filenames=args.files,result_file=args.target_file,percent=args.percent,)
    traing_test_split(filenames=args.files,percent =args.percent)