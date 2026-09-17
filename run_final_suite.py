"""Resumable final assignment runner: one model load serves all four tasks."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
from pilot.config import MODEL_SPECS,LOADING_STRATEGY
from pilot.task_config import task_plan
from pilot.task_data import load_bundle
from run_experiment import load_calibration

TASKS=("STSB","Banking77","SciFact","Arxiv-Clustering")
DATA_FILES={"Banking77":"banking77_official.json","SciFact":"scifact_official.json","Arxiv-Clustering":"arxiv_official.json"}

def complete(path,model,task,dimensions):
    f=path/"results.json"
    if not f.exists(): return False
    try:
        rows=json.loads(f.read_text(encoding="utf-8")).get("scores",[])
        return set(dimensions)<={int(r["dimension"]) for r in rows if r.get("model")==model and r.get("task")==task}
    except (OSError,ValueError,TypeError,KeyError,json.JSONDecodeError): return False

def progress(path,state):
    path.mkdir(parents=True,exist_ok=True); (path/"suite_progress.json").write_text(json.dumps(state,indent=2),encoding="utf-8")

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--data-dir",type=Path,required=True); p.add_argument("--output-root",type=Path,required=True)
    p.add_argument("--cache-dir",type=Path,default=Path("/content/nlp_final_native_cache"),help="Temporary native embedding cache; keep this local to avoid large Drive I/O.")
    p.add_argument("--models",nargs="+",choices=MODEL_SPECS,default=list(MODEL_SPECS)); p.add_argument("--batch-size",type=int,default=4); p.add_argument("--max-length",type=int,default=256); p.add_argument("--seed",type=int,default=42)
    a=p.parse_args(argv); calibration_path=a.data_dir/"shared_calibration.json"; calibration=load_calibration(calibration_path)
    bundles={t:load_bundle(a.data_dir/name,t) for t,name in DATA_FILES.items()}
    from huggingface_hub import HfApi
    from datasets import load_dataset
    from pilot.model import FrozenEncoder,check_gpu
    from pilot.core import cached_embeddings
    from pilot.pca_artifact import artifact_identity,get_or_fit_pca
    from pilot.task_runner import evaluate_bundle,evaluate_stsb,save_task_outputs
    import torch
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    hardware=check_gpu(); api=HfApi(); a.output_root.mkdir(parents=True,exist_ok=True); a.cache_dir.mkdir(parents=True,exist_ok=True)
    stsb_revision=api.dataset_info("mteb/stsbenchmark-sts").sha; stsb_rows=list(load_dataset("mteb/stsbenchmark-sts",revision=stsb_revision)["validation"])
    state={"started":time.time(),"models":a.models,"tasks":list(TASKS),"completed":[]}; progress(a.output_root,state)
    for model in a.models:
        dims=list(MODEL_SPECS[model]["dimensions"]); pending=[t for t in TASKS if not complete(a.output_root/"results"/model/t,model,t,dims)]
        if not pending:
            print(f"SKIP completed model: {model}",flush=True); continue
        print(f"\n===== LOAD ONCE: {model} | pending {pending} =====",flush=True)
        base_plan=task_plan("STSB",model,dims,None,a.seed,a.batch_size,a.max_length,None,None,LOADING_STRATEGY,calibration)
        if not base_plan["executable"]: raise ValueError("Execution blocked: "+" ".join(base_plan["blockers"]))
        revision=api.model_info(base_plan["model_id"]).sha; encoder=FrozenEncoder(model,revision,a.batch_size,a.max_length)
        identity={"model":base_plan["model_id"],"revision":revision,"width":base_plan["native_dimension"],"max_length":a.max_length,"quantization":base_plan["quantization"]}
        cache=a.cache_dir/model
        def encode(texts): return cached_embeddings(cache,identity,texts,encoder.encode,native_dimension=base_plan["native_dimension"])[0]
        pca_path=a.output_root/"pca"/f"{model}.npz"; pca_identity=artifact_identity(base_plan,revision,calibration,a.max_length); pca,reused=get_or_fit_pca(pca_path,pca_identity,calibration,encode)
        print(f"PCA {'reused' if reused else 'fitted'}: {pca_path}",flush=True)
        for task in pending:
            out=a.output_root/"results"/model/task; started=time.perf_counter()
            plan=task_plan(task,model,dims,None,a.seed,a.batch_size,a.max_length,None,bundles.get(task),LOADING_STRATEGY,calibration)
            if not plan["executable"]: raise ValueError(f"{model}/{task} blocked: "+" ".join(plan["blockers"]))
            print(f"RUN {model} / {task}",flush=True)
            if task=="STSB": rows,preds,meta=evaluate_stsb(stsb_rows,stsb_revision,plan,encode,pca,calibration)
            else: rows,preds,meta,_=evaluate_bundle(bundles[task],plan,encode,calibration,pca=pca)
            meta.update(model_revision=revision,hardware=hardware,seconds=time.perf_counter()-started,pca_artifact=str(pca_path),pca_artifact_reused=bool(reused))
            save_task_outputs(out,rows,preds,meta,None); state["completed"].append([model,task]); progress(a.output_root,state)
            print(f"DONE {model} / {task} -> {out}",flush=True)
        del encoder
        import gc; gc.collect(); torch.cuda.empty_cache()
    state.update(finished=time.time(),status="complete"); progress(a.output_root,state); print("\nFINAL SUITE COMPLETE",flush=True)

if __name__=="__main__":
    try: main()
    except (ValueError,RuntimeError,ImportError,OSError) as e: raise SystemExit(f"Suite stopped: {e}. Rerun the same command to resume from completed task checkpoints.") from e
