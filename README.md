# resuggest

This repository holds the code for the paper: [A Case Study of Closed-Domain Response Suggestion with Limited Training Data](https://link.springer.com/chapter/10.1007/978-3-319-99133-7_18) ([Author Copy](http://lpag.de/assets/pdf/2018-TIR-response-suggestion.pdf))

1. A text processing pipeline that generates paired training data from QuestionPoint XML transcripts.
2. A collection of response suggestion algorithms: retrieval models and representation learning approaches
3. An evaluation framework for response suggestion using BLEU score.

## Usage

**available soon**


## Citation

```
@inproceedings{galke_case_2018,
	address = {Cham},
	title = {A {Case} {Study} of {Closed}-{Domain} {Response} {Suggestion} with {Limited} {Training} {Data}},
	copyright = {All rights reserved},
	isbn = {978-3-319-99133-7},
	abstract = {We analyze the problem of response suggestion in a closed domain along a real-world scenario of a digital library. We present a text-processing pipeline to generate question-answer pairs from chat transcripts. On this limited amount of training data, we compare retrieval-based, conditioned-generation, and dedicated representation learning approaches for response suggestion. Our results show that retrieval-based methods that strive to find similar, known contexts are preferable over parametric approaches from the conditioned-generation family, when the training data is limited. We, however, identify a specific representation learning approach that is competitive to the retrieval-based approaches despite the training data limitation.},
	booktitle = {Database and {Expert} {Systems} {Applications}},
	publisher = {Springer International Publishing},
	author = {Galke, Lukas and Gerstenkorn, Gunnar and Scherp, Ansgar},
	year = {2018},
	pages = {218--229}
}
```
