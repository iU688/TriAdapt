"""Dataset-free packaging checks. Does not assert a full training replay."""
import hashlib, json, os, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
paths=[ROOT/'src',ROOT/'runtime',ROOT/'tools',ROOT]
sys.path[:0]=list(map(str,paths))
import numpy as np
from source_pose_geometry import norm_by_bone_length
from common.paper_pose_metrics import bone_normalized_errors
from mupots_multimodule.model import TemporalGraph2DCorrector, KinematicGraphRefiner

def main():
    manifest=json.loads((ROOT/'MANIFEST_SHA256.json').read_text())
    for name,expected in manifest.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest().upper()==expected,name
    rng=np.random.default_rng(42)
    pose=rng.normal(size=(2,17,3)).astype(np.float32)
    np.testing.assert_allclose(norm_by_bone_length(pose,pose),pose,atol=1e-6)
    np.testing.assert_allclose(bone_normalized_errors(pose,pose),0,atol=1e-5)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(map(str,paths)))
    commands=['train.py','tools/train_source.py','runtime/experiments/mechanism_controls/train_parameter_matched_ungated.py']
    for cmd in commands:
        r=subprocess.run([sys.executable,'-B',str(ROOT/cmd),'--help'],env=env,cwd=ROOT,capture_output=True,text=True)
        if r.returncode:raise RuntimeError(cmd+'\n'+r.stderr)
    print(json.dumps({'file_hashes':len(manifest),'model_imports':'passed','source_geometry':'passed','cli_help':len(commands),'full_training_replay':False},indent=2))
if __name__=='__main__':main()
