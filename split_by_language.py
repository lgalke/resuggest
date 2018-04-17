""" Script to split a set of source target by the identified language into
respective files """
import argparse
from collections import defaultdict
import os
import spacy
from spacy_cld import LanguageDetector


def load_pairs(src_file, dst_file):
    """ Load sources and targets """
    sources, targets = [], []
    with open(src_file, 'r') as src_fh:
        for line in src_fh:
            sources.append(line.strip())
    with open(dst_file, 'r') as dst_fh:
        for line in dst_fh:
            targets.append(line.strip())
    return sources, targets


def write_lang_pair(paths, sents):
    """ Write a couple of :code:`sents` to :code:`paths` """
    assert len(paths) == len(sents)
    for fname, sentence in zip(paths, sents):
        with open(fname, 'a') as fpath:
            print(sentence, file=fpath)


def lang_path(lang, fname):
    """ Sneak the language identifier into the filename """
    fname, __ext = os.path.splitext(fname)
    return '.'.join([fname, lang])


def main(sources_path, targets_path):
    """ Identify language of (sources, targets) pairs and write them back in
    new files """
    nlp = spacy.load('en')
    language_detector = LanguageDetector()
    nlp.add_pipe(language_detector)

    sources, targets = load_pairs(sources_path, targets_path)
    assert len(sources) == len(targets)
    print("Processing", len(sources), "pairs...")
    lang_counts = defaultdict(int)

    for src_sent, tgt_sent in zip(sources, targets):
        combined_sent = ' '.join([src_sent, tgt_sent])
        doc = nlp(combined_sent)
        for lang in doc._.languages:
            lang_counts[lang] += 1
            lang_sources_path = lang_path(lang, sources_path)
            lang_targets_path = lang_path(lang, targets_path)
            write_lang_pair((lang_sources_path, lang_targets_path),
                            (src_sent, tgt_sent))

    for lang, count in lang_counts.items():
        print("{}: {}".format(lang, count))

    print("Total resulting pairs:", sum(lang_counts.values()))


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser()

    PARSER.add_argument('-s', '--sources', type=str, default='tmp/sources.txt')
    PARSER.add_argument('-t', '--targets', type=str, default='tmp/targets.txt')

    ARGS = PARSER.parse_args()

    main(ARGS.sources, ARGS.targets)
