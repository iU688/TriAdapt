"""Verify decoded source data when NPZ serialization differs across environments."""
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def digest_content(path):
    """Preserve dictionary order, array dtype/shape and all numeric bytes."""
    h=hashlib.sha256()
    def token(x):
        b=str(x).encode('utf8');h.update(len(b).to_bytes(8,'big'));h.update(b)
    def visit(x):
        if isinstance(x,np.ndarray):
            token('array');token(x.dtype.str);token(x.shape)
            if x.dtype.hasobject:
                for z in x.flat:visit(z)
            else:h.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x,dict):
            token('dict');token(len(x))
            for k,v in x.items():visit(k);visit(v)
        elif isinstance(x,(list,tuple)):
            token(type(x).__name__);token(len(x))
            for v in x:visit(v)
        elif isinstance(x,np.generic):visit(np.asarray(x))
        elif x is None:token('none')
        elif isinstance(x,(str,int,float,bool)):
            token(type(x).__name__);token(repr(x))
        else:raise TypeError(type(x))
    with np.load(path,allow_pickle=True) as f:
        for key in f.files:
            token(key);value=f[key]
            if key=='metadata' and value.dtype.kind in 'US' and value.shape==():
                m=json.loads(value.item())
                # A later converter added this descriptive field. It is not read by
                # the dataset loader. No scientific metadata field is discarded.
                if 'test_access_policy' in m:
                    if m['test_access_policy']!='not accessed' or m.get('split') not in ('train','validation'):
                        raise ValueError('Unexpected test-access metadata')
                    del m['test_access_policy']
                value=np.asarray(json.dumps(m,sort_keys=True))
            visit(value)
    return h.hexdigest().upper()

def verify_source(path,key,expected_byte_hash):
    actual=hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()
    if actual==expected_byte_hash.upper():return {'sha256':actual,'verification':'archived_bytes'}
    manifest=json.loads((ROOT/'config/source_content_sha256.json').read_text())
    expected=manifest['assets'][key]
    if expected['archived_sha256']!=expected_byte_hash.upper():raise ValueError('Source identity mismatch')
    if digest_content(path)!=expected['content_sha256']:raise ValueError('Decoded source content mismatch: '+key)
    return {'sha256':actual,'verification':'exact_decoded_content','content_sha256':expected['content_sha256']}
