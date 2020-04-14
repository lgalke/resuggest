# resuggest

This repository holds the code for the paper: [A Case Study of Closed-Domain Response Suggestion with Limited Training Data](https://link.springer.com/chapter/10.1007/978-3-319-99133-7_18) ([Author Copy](http://lpag.de/assets/pdf/2018-TIR-response-suggestion.pdf))

1. A text processing pipeline that generates paired training data from QuestionPoint XML transcripts.
2. A collection of response suggestion algorithms: retrieval models and representation learning approaches
3. An evaluation framework for response suggestion using BLEU score.

## Usage of Response Suggestion Methods

1. pip install -r requirements
2. Bring data into `sources.txt` and `targets.txt` format, where line i of 'targets.txt' is the response to question in line i of 'sources.txt'
3. Consult `python3 single_eval.py -h` to inspect command line arguments.
4. Run an experiment via `python3 single_eval.py [OPTIONS] path/to/sources.txt path/to/targets.txt`

## Citation

```
@inproceedings{galke_case_2018,
	address = {Cham},
	title = {A {Case} {Study} of {Closed}-{Domain} {Response} {Suggestion} with {Limited} {Training} {Data}},
	copyright = {All rights reserved},
	isbn = {978-3-319-99133-7},
	booktitle = {Database and {Expert} {Systems} {Applications}},
	publisher = {Springer International Publishing},
	author = {Galke, Lukas and Gerstenkorn, Gunnar and Scherp, Ansgar},
	year = {2018},
	pages = {218--229}
}
```
