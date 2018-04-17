import argparse
import itertools
import os
import re
import xml.etree.ElementTree as ET
# from experiment_superscript import yield_lines
from resuggest.evaluation import tokenize
from resuggest.utils.tokens import EOT, EOU

DEBUG = False  # print all incidents of a transcript
ADD_SUPPLEMENTARY_TOKENS = False

# How many meta incidents before actual reply
# 0 first: patron screen name
# 1 second: librarian joined chat
# 2 third: default response (Hello and Welcome)
MAIN_ANSWER_INDEX = 3  # index of main answer to PatronQuestion

# Dirt mainly consists of <br> tags

N_DROPPED = 0
N_TRANSCRIPTS = 0
# There are even more html tags
DIRT = re.compile(r'<br>|</?p>')

STATUS = re.compile(r"Librarian '.*' has joined the session.")


def is_meaningful(msg):
    """ Function to filter out status messages """
    if msg.startswith("Patron's screen name:"):
        return False
    if msg.startswith("Patron e-mail changed from:"):
        return False
    if msg.startswith('Set Resolution: '):
        return False
    if msg.startswith('Closed by Librarian # '):
        return False
    if msg.startswith('Referred from: '):
        return False
    if msg.startswith('Claimed by: '):
        return False
    if msg == 'Librarian ended chat session.':
        return False
    if STATUS.match(msg):
        return False

    return True


def clean(dirty):
    raise DeprecationWarning('should be done in preprocessing')
    """ clean the dirt and replace it with blanks """
    cleaned = re.sub(DIRT, ' ', dirty)
    return cleaned


# and newlines
def splitjoin(line):
    """ Get rid of whitespace-like characters (including newlines) """
    return ' '.join(line.split())


def text(xmlnode):
    ''' main method to access the text of any incident'''
    # return clean(x.find('Text').text)

    return xmlnode.find('Text').text


def extract_initial(transcript, response_index=MAIN_ANSWER_INDEX):
    """ Extract only initial question answer pair"""
    global N_DROPPED
    try:
        q = text(transcript.find('PatronQuestion'))
        library = transcript.findall('LibraryIncident')
        a = text(library[response_index])

        return q, a
    except (IndexError, AttributeError):
        N_DROPPED += 1
        return None


def consume_while(generator, keys, key=lambda x: x.tag):
    """
    Consumes :code:`generator` while a :code:`key` of the element is in
    :code:`keys`.
    >>> g = (i for i in range(100))
    >>> keys = list(range(5))
    >>> key = lambda x: x
    >>> elems, rest = consume_while(g, keys, key=key)
    >>> elems
    [0, 1, 2, 3, 4]
    >>> rest = list(rest)
    >>> rest[0]
    5
    >>> rest[1]
    6
    >>> rest[-1]
    99
    """
    elements = []
    peek = next(generator)
    while key(peek) in keys:
        elements.append(peek)
        peek = next(generator)

    return elements, itertools.chain([peek], generator)


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    else:
        return text


def remove_type_prefixes(text):
    text = remove_prefix(text, 'Chat Transcript: ')
    text = remove_prefix(text, 'Qwidget: ')
    return text


def preprocess(xml_elements):
    """ Operates on an interable of xml elements """
    # 1. extract the content of the text tag
    elems = map(text, xml_elements)
    # 2. Filter status messages
    elems = filter(is_meaningful, elems)
    elems = map(remove_type_prefixes, elems)
    if ADD_SUPPLEMENTARY_TOKENS:
        return EOU.join(list(elems) + [EOT])
    else:
        return ' '.join(list(elems))


def harvest_transcript(transcript):
    """ transcript generator of xml subtags """
    g = transcript
    try:
        while True:
            patron, library = False, False
            patron, g = consume_while(g, ['PatronQuestion', 'PatronIncident'])
            library, g = consume_while(g, ['LibraryIncident'])
            patron, library = preprocess(patron), preprocess(library)
            yield (patron, library)
    except StopIteration:
        # Closing words of patron do not matter
        pass


def parse_transcripts(transcripts, full_context=False):
    """ Parse an iterable of transcripts """
    # Idea:
    # Parse transcript iteratively
    # go through child elements of transcript
    # and append to both context and answer
    global N_TRANSCRIPTS

    for transcript in transcripts:
        N_TRANSCRIPTS += 1
        if not full_context:
            # new and better variant
            yield from harvest_transcript(iter(transcript))
        else:
            raise UserWarning("Do not use the old variant")





        # # list better for joining 
        # context = []
        # patron_talking = True
		# # TODO: choice question-answer-pair (answers/question grouped together) vs append whole context
        # for i, child in enumerate(transcript):
        #     if child.tag in ['PatronQuestion', 'PatronIncident']:
        #         if not patron_talking:
        #             context.append(EOT)
        #             # TODO: yield
        #             #if not is_status(utterance):
        #             #yield context_str, utterance
        #             #context.extend([utterance, EOU])

        #             context = []
        #         context.extend([text(child), EOU])
        #     elif child.tag == 'LibraryIncident' and i >= 3:
        #         if patron_talking:
        #             context.append(EOT)
        #         utterance = text(child)
        #         context_str = ' '.join(context)

        #         if DEBUG:
        #             print(context_str)
        #             print(utterance)
        #             input()

        #         if is_meaningful(utterance):
        #             yield context_str, utterance
        #             context.extend([utterance, EOU])
        #         else:
        #             print("DROP", utterance)

    # for transcript in transcripts:
    #     context = ''
    #     utterance = ''
    #     patron_talking = True
    #     for i, child in enumerate(transcript):
    #         if child.tag in ['PatronQuestion', 'PatronIncident']:
    #             if not patron_talking:
    #                 patron_talking = True
    #                 utterance += EOT
    #                 # Yield before extending to context
    #                 if DEBUG:
    #                     print('__context__ ' + splitjoin(context),
    #                             '__utterance__ ' + splitjoin(utterance),
    #                             sep='\n')
    #                     input()
    #                 if CHAINED_UTTERANCES:
    #                     yield context, utterance
    #                     context += utterance
    #                     utterance = ''
    #             context += text(child) + EOU
    #         elif child.tag == 'LibraryIncident' and i >= 3:
    #             if patron_talking:
    #                 # Transition from patron_talking
    #                 patron_talking = False
    #                 context += EOT
    #             utterance += text(child) + EOU
    #             if not CHAINED_UTTERANCES:
    #                 yield context, utterance
    #                 context += utterance
    #                 utterance = ''

    #     if utterance != '':
    #         yield context, utterance + EOT

    # return data.encode('utf-8', 'xmlcharreplace')


def parse_chatlog(xml_file, full=True):
    """ Parse transcripts of an xml file """
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # sum ( ... ) if parse_transcript returns a list

    if full:
        pairs = list(parse_transcripts(root.iter(tag='Transcript')))
        print(xml_file, len(pairs))
    else:
        pairs = [extract_initial(t) for t in root.iter(tag='Transcript')]
        pairs = [e for e in pairs if e is not None]

    return pairs


def store_parallel_text_format(pairs,
                               prefix='tmp',
                               sources='sources.txt',
                               targets='targets.txt',
                               max_tokens=800):
    """ Store iterable of pairs in parallel text format """
    print("mkdir -p", prefix)
    os.makedirs(prefix, exist_ok=True)
    sources = os.path.join(prefix, sources)
    targets = os.path.join(prefix, targets)

    # for pair in pairs:
    #     question, answer = pair
    #     with open(sources, 'a') as src:
    #         print(splitjoin(question), file=src)
    #     with open(targets, 'a') as dest:
    #         print(splitjoin(answer), file=dest)

    src, dst = open(sources, 'w'), open(targets, 'w')
    for question, answer in pairs:
        print(question,answer)
        if not (len(tokenize(question)) > max_tokens or len(tokenize(answer)) > max_tokens):
            print(splitjoin(question), file=src)
            print(splitjoin(answer), file=dst)
        else:
            print("dropped:\n",question,"\n",answer,"\n")
    src.close()
    dst.close()




if __name__ == '__main__':
    import doctest
    doctest.testmod()
    PARSER = argparse.ArgumentParser()
    PARSER.add_argument('files', nargs='+', help='input xml files')
    PARSER.add_argument('-P',
                        '--prefix',
                        default='tmp',
                        help='path prefix to put sources.txt and targets.txt')
    ARGS = PARSER.parse_args()
    QA = list()  # mypy: list(str,str)
    for fpath in ARGS.files:
        QA.extend(parse_chatlog(fpath))
    full_len = len(QA)
    QA = [pair for pair in QA if pair[1]]
    store_parallel_text_format(QA, ARGS.prefix)

    print("Detected", N_TRANSCRIPTS, "transcripts.")
    print(full_len - len(QA), 'pairs dropped.')
    print("Storing", len(list(QA)), "pairs.")
