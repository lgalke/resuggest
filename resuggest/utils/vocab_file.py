import re
from argparse import ArgumentParser
from collections import Counter

def voca_file(filenames, result_file=None, percent=100, counts=False, seq2seq_voc = False):
    # TODO: replace RegEx with better tokenizer
    counter = Counter()
    print(filenames)
    for filename in filenames:
        words = re.findall(r'\w\w+', open(filename).read())
        counter = counter + Counter(words)
    target_voc_size = min(int(len(counter) / 100) * percent, len(words))

    if result_file:
        rf_s = result_file
        output = open(result_file, "w")
        print("created {} file".format(rf_s))
    else:
        rf_s = filename + ".voc"
        output = open(rf_s, "w")
        print("created {} file".format(rf_s))


    # trick <unk>, <s> and </s> at top
    if seq2seq_voc:
        # removes internal end symbols as well
        max_token = counter.most_common(1)[0][1]
        counter.update({"<unk>":max_token+3,"<s>":max_token+2,"</s>":max_token+1})
        del counter['__eot__']
        del counter['__eou__']

    if counts:
        for w, c in counter.most_common(target_voc_size):
            print(w + " " + str(c), file=output)
    else:
        for w, c in counter.most_common(target_voc_size):
            print(w, file=output)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        'files',
        nargs='*',
        type=str,
        help='file(s) to be counted')
    parser.add_argument(
        '-tf',
        '--target_file',
        type=str,
        help='file with wordcounts')
    parser.add_argument(
        '-c',
        '--counts',
        type=bool,
        help='"True" for printing the counts')

    parser.add_argument(
        '-p',
        '--percent',
        type=int,
        default= 100,
        help='percent at which infrequent words are cut off')

    args = parser.parse_args()
    voca_file(filenames=args.files, result_file=args.target_file, percent=args.percent, )
