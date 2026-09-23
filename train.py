"""Run the paper's staged training recipe without changing its scientific rules."""
import argparse,hashlib,json,os,shutil,subprocess,sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'tools'))
from source_content import verify_source
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest().upper()
def invoke(script,args,env):
    cmd=[sys.executable,'-B',str(ROOT/script),*map(str,args)]
    print(' '.join(cmd),flush=True);subprocess.run(cmd,cwd=ROOT,env=env,check=True)
def save_new(p,obj):
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(obj,f,indent=2)
def copy_weight(source,target):
    if target.exists():
        if sha(source)!=sha(target):raise FileExistsError(target)
    else:target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)

def main():
    p=argparse.ArgumentParser(description="Train TriAdapt and its two module-deletion variants")
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--seeds',default='42,123,2026')
    p.add_argument('--plan',action='store_true',help='Check assets without training')
    a=p.parse_args()
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=os.pathsep.join(str(ROOT/x) for x in ['src','runtime','runtime/tools','tools','.']))
    if not a.assets:p.error('train requires --assets')
    env['GRAPHMLP_V8_EXTERNAL_PATHS']=str(a.assets.resolve())
    cfg=json.loads(a.assets.read_text(encoding='utf-8-sig'))
    lockfile=ROOT/'protocol/protocol_freeze_v8_formal_confirmation.json'
    lock=json.loads(lockfile.read_text());seeds=[int(x) for x in a.seeds.split(',')]
    if len(set(seeds))!=len(seeds) or not seeds or any(x not in [42,123,2026] for x in seeds):raise ValueError('Use unique seeds 42,123,2026')
    source=[];source_verifications={}
    for key in ['mpi_train','mpi_val','pw3d_train','pw3d_val']:
        path=Path(cfg['source_assets'][key]).resolve()
        checked=verify_source(path,key,lock['source_assets'][key]['sha256'])
        source_verifications[key]=checked
        lock['source_assets'][key]['sha256']=checked['sha256']
        source+=['--'+key,str(path)]
    public=Path(cfg['checkpoints']['public_graphmlp']).resolve()
    if sha(public)!='76B65A860A0C64DE0137F5E3AE118DB4F2A1F482C549A3211E9F1A26C9C9DEA8':raise ValueError('Public weight mismatch')
    if a.plan:
        print(json.dumps({'seeds':seeds,'stage1_batch':64,'stage2_batch':32,'max_epochs_each_stage':30,'patience':5,'stage2_arms':lock['trained_arms'],'source_verifications':source_verifications,'training_started':False},indent=2));return
    copy_weight(public,ROOT/'checkpoints/public_graphmlp/243frame.pth')
    jobs=[]
    for seed in seeds:
        out=ROOT/f'runtime/outputs/reproduction_fam/seed{seed}'
        if out.exists():raise FileExistsError(out)
        invoke('runtime/experiments/mechanism_controls/train_parameter_matched_ungated.py',source+['--checkpoint',public,'--output_dir',out,'--seed',seed,'--tag',f'ungated_s{seed}','--frames',243,'--dct_keep',27,'--epochs',30,'--early_stop',5,'--samples_per_epoch',100000,'--batch_size',64,'--lr','2e-4'],env)
        fam=ROOT/f'checkpoints/m2/seed{seed}/best_ungated_s{seed}.pth'
        copy_weight(out/f'checkpoints/best_ungated_s{seed}.pth',fam)
        derived=json.loads(json.dumps(lock));derived['m2_checkpoints'][str(seed)]['sha256']=sha(fam)
        derived['replication_provenance']={'original_lock_sha256':sha(lockfile),'source_verifications':source_verifications,'new_source_selected_transfer':str(fam.relative_to(ROOT)),'target_results_used':False}
        runlock=ROOT/f'work/replication_seed{seed}.json';save_new(runlock,derived)
        for arm in lock['trained_arms']:
            dest=ROOT/lock['output_root']/arm/f'seed{seed}'
            if dest.exists():raise FileExistsError(dest)
            invoke('tools/train_source.py',source+['--m2_checkpoint',fam,'--output_dir',dest,'--seed',seed,'--arm',arm,'--stage','formal','--formal_protocol',runlock,'--batch_size',32,'--num_workers',0],env)
            jobs.append({'arm':arm,'seed':seed,'checkpoint_sha256':sha(dest/'best_source_selected.pth')})
    save_new(ROOT/'work/reproduction_training_receipt.json',{'status':'completed_requested_training','jobs':jobs,'target_access':'none'})
    print('Training complete. Checkpoints and source-validation logs are in outputs/formal_v8/.')
if __name__=='__main__':main()
