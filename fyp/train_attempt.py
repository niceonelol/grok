"""

CHANGES MADE TO ../grok/training.py:
    - Changed max_epochs to 10,000 from None in add_args and from 1e8 in train function

"""

import grok
from grok.training import *

def make_namespace():
    parser = add_args()
    parser.set_defaults(logdir=os.environ.get("GROK_LOGDIR", "."))
    hparams = parser.parse_args()
    hparams.datadir = os.path.abspath(hparams.datadir)
    hparams.logdir = os.path.abspath(hparams.logdir)
    return hparams


if __name__ == "__main__":
    hparams = make_namespace()
    