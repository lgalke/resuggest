""" Helper function for an interactive session """
from utils.tokens import EOU, EOT


def utterance_string(agent, utterance, confidence=None):
    """ Transforms the utterance of an agent into a human readable format,
    optionally with confidence. """
    utt_str = "[{}]".format(agent)
    if confidence is not None:
        print(confidence)
        utt_str += "@{:.2f}".format(confidence)
    utt_str += " {}".format(utterance)
    return utt_str


def input_loop(agent, reset_context=True, confidential=True, use_tokens=False):
    """ Runs an input loop """
    print(agent, "joined the session.")
    context = []
    try:
        while True:
            inp = input("{patron} ")

            if inp == 'ZZ':
                # Force reset the context
                print(agent, "left the session.")
                print(agent, "joined the session.")
                context = []
                continue

            if use_tokens:
                # Extend current context by next user-string
                context.extend([inp, EOU, EOT])
            else:
                context.extend([inp])
            preds = agent.predict(' '.join(context))
            if confidential:
                # Zip the answers to the scores
                preds = zip(*preds)

            for pred in preds:
                print(utterance_string(agent, *pred))

            if reset_context:
                context = []

    except EOFError:
        print('\nBye.')
