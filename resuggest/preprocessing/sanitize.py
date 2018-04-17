import sys
"""
<br> -> ' '
\r -> ''
U+0005 ENQUIRY -> ''
U+000D CARRIAGE RETURN -> ''
"""


def sanitize(text):
    """
    >>> blob = "something<br>else, plus<br> br tag, \u0005 ok \\u000D newline"
    >>> blob
    >>> sanitize(blob)
    'something else, plus  br tag,  ok newline'
    """
    text = text.replace('\u0005', '')
    text = text.replace('<br>', ' ')
    text = text.replace('\u000D', ' ')
    return text


def main():
    with open(sys.argv[1], 'rb') as f:
        # This method works for Umlaute, but retains other errors
        blob = f.read().decode('utf-8', errors='replace')
        # print(blob, end='')
        blob = sanitize(blob)
        sys.stdout.write(blob)
        print("Printing with encoding:", sys.stdout.encoding, file=sys.stderr)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        main()
    else:
        print("Running tests...")
        import doctest
        doctest.testmod()
        print("Done.")
