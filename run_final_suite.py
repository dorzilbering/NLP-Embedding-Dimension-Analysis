"""Resumable final runner for the complete 5-model x 4-task assignment matrix.

Runs one model at a time.  run_experiment's native-embedding cache and persistent
per-model PCA artifacts avoid recomputing expensive GPU work across tasks/reruns.
Each completed task is checkpointed directly in the requested output root.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
from pilot.config import MODEL_SPECS

TASKS=("STSB","Banking77","SciFact","Arxiv-Clustering")
DATA_FILES={
    "Banking77":"banking77_official.json",
    "SciFact":"scifact_official.json",
    "Arxiv-Clustering":"arxiv_official.json",
}

def complete(path,model,task,dimensions):
    result=path/"results.json"
    if not result.exists(): return False
    try:
        b=json.loads(result.read_text(encoding="utf-8")); rows=b.get("scores",[])
        seen={int(r["dimension"]) for r in rows if r.get("model")==model and r.get("task")==task}
        return set(dimensions)<=seen
    except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError):
        return False

def run(cmd):
    print("\n$ "+" ".join(map(str,cmd)),flush=True)
    subprocess.run(cmd,check=True)

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",type=Path,required=True)
    p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--models",nargs="+",choices=MODEL_SPECS,default=list(MODEL_SPECS))
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--max-length",type=int,default=256); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(argv); a.output_root.mkdir(parents=True,exist_ok=True)
    calibration=a.data_dir/"shared_calibration.json"
    if not calibration.exists(): p.error(f"Missing {calibration}")
    for task,name in DATA_FILES.items():
        if not (a.data_dir/name).exists(): p.error(f"Missing {a.data_dir/name}")
    cache=a.output_root/"cache"/"native"; pca_dir=a.output_root/"pca"
    summary={"started":time.time(),"models":a.models,"tasks":list(TASKS),"completed":[]}
    for model in a.models:
        dims=list(MODEL_SPECS[model]["dimensions"])
        print(f"\n===== {model}: dimensions {dims} =====",flush=True)
        for task in TASKS:
            out=a.output_root/"results"/model/task
            if complete(out,model,task,dims):
                print(f"SKIP completed: {model} / {task}",flush=True); summary["completed"].append([model,task,"reused"]); continue
            cmd=[sys.executable,"run_experiment.py","--task",task,"--model",model,"--dimensions",*map(str,dims),"--calibration",str(calibration),"--batch-size",str(a.batch_size),"--max-length",str(a.max_length),"--seed",str(a.seed),"--cache",str(cache),"--pca-dir",str(pca_dir),"--output",str(out)]
            if task!="STSB": cmd += ["--data",str(a.data_dir/DATA_FILES[task])]
            run(cmd); summary["completed"].append([model,task,"ran"])
            (a.output_root/"suite_progress.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
        # A subprocess exits after every task, so GPU model memory is released before the next task.
        # Native embeddings/PCA remain on disk and are reused, making the suite restart-safe.
    summary["finished"]=time.time(); summary["status"]="complete"
    (a.output_root/"suite_progress.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nFINAL SUITE COMPLETE",flush=True)

if __name__=="__main__":
    try: main()
    except subprocess.CalledProcessError as e: raise SystemExit(f"Suite stopped after failed experiment (exit {e.returncode}). Rerun the same command to resume.") from e
