from ... import pathtool
import os
from .speech import (
    get_speech,
    get_translate
)

GCP_DIR = os.path.join (os.environ ['HOME'], '.gcp'); pathtool.mkdir (GCP_DIR)
DEFAULT_CREDENTIAL = os.path.join (GCP_DIR, 'credential.json')

def set_credential (path = None):
    global DEFAULT_CREDENTIAL

    if path is None:
        path = DEFAULT_CREDENTIAL
    assert os.path.isfile (path), 'credential file not found: {}'.format (path)
    os.environ ['GOOGLE_APPLICATION_CREDENTIALS'] = path

if os.path.isfile (DEFAULT_CREDENTIAL):
    set_credential ()
