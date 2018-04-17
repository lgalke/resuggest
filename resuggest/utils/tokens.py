""" Class for tokens with operator overloading to concatenate properly with
strings
>>> "ok" + EOU
'ok __eou__ '
>>> EOT + 'something'
' __eot__ something'
>>> EOU + EOT
' __eou__  __eot__ '
>>> "But what if" + EOU + "ah ok" + EOU + EOT
'But what if __eou__ ah ok __eou__  __eot__ '
"""

# End of unit
EOU = ' __eou__ '

# End of transmission
EOT = ' __eot__ '

if __name__ == '__main__':
    import doctest
    doctest.testmod()
