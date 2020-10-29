import getopt as optparse
import sys
from .termcolor import tc
import math

helps = []
long_opts = ['help']
short_opts = []

def usage ():
    maxlen = max ([len (opt) + 2 for opt, desc in helps]) + 1
    print ("\nMandatory arguments to long options are mandatory for short options too.")
    for opt, desc in helps:
        if '---' in opt:
            color = tc.red
        else:
            color = tc.white
        spaces = maxlen - (len (opt) + 1)
        line = ('  {}{}{}'.format (color (opt), ' ' * spaces, desc))
        print (line)

def add_option (sname = None, lname = None, desc = None):
    global long_opts, short_opts

    val = None
    if lname:
        if lname.startswith ("--"):
            lname = lname [2:]
        try: lname, val = lname.split ("=", 1)
        except ValueError: pass
        assert lname not in long_opts, "option --" + lname + ' is already exist'
        long_opts.append (lname + (val and "=" or ''))

    if sname:
        if sname.startswith ("-"):
            sname = sname [1:]
        try: sanme, val = sname.split ("=", 1)
        except ValueError: pass
        assert sname not in short_opts, "option -" + sname + ' is already exist'
        short_opts.append (sname + (val and ":" or ''))

    if lname and sname:
        opt = "-{}, --{}".format (sname, lname)
    elif sname:
        opt = '-{}'.format (sname)
    elif lname:
        opt = '    --{}'.format (lname)
    if val:
        opt += '=' + val
    helps.append ((opt, desc or ''))

def add_options (*names):
    for name in names:
        assert name and name [0] == "-"
        if name.startswith ("--"):
            add_option (None, name [2:])
        else:
            add_option (name [1:])

class ArgumentOptions:
    def __init__ (self, kopt = {}, argv = []):
        self.__kopt = kopt
        self.argv = argv

    def items (self):
        return self.__kopt.items ()

    def __contains__ (self, k):
        return k in self.__kopt

    def get (self, k, v = None):
        return self.__kopt.get (k, v)

    def set (self, k, v = None):
        self.__kopt [k] = v

    def remove (self, k):
        try:
            del self.__kopt [k]
        except KeyError:
            pass

def options ():
    global long_opts, short_opts

    argopt = optparse.getopt (sys.argv [1:], "".join (short_opts).replace ("=", ":"), long_opts)
    kopts_ = {}
    for k, v in argopt [0]:
        if k in kopts_:
            if not isinstance (kopts_ [k], list):
                kopts [k] = {kopts [k]}
            kopts_ [k].add (v)
        else:
            kopts_ [k] = v
    return ArgumentOptions (kopts_, argopt [1])

