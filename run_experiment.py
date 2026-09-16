"""Run any assignment model on any assignment task with a shared per-model PCA basis."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
from pilot.config import MODEL_SPECS,TASK_SPECS,LOADING_STRATEGY
from pilot.task_config import task_plan
from pilot.task_data import load_bundle,fingerprint

def load_calibration(path):
    b=json.loads(Path(path).read_text(encoding="utf-8"))
    if b.get("role")!="shared_pca_calibration" or b.get("manifest_hash")!=fingerprint(b.get("texts",[])): raise ValueError("Invalid calibration bundle.")
    return b

def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--task",choices=TASK_SPECS,required=True); p.add_argument("--model",choices=MODEL_SPECS,required=True)
    p.add_argument("--dimensions",nargs="+",type=int); p.add_argument("--data",type=Path); p.add_argument("--calibration",type=Path,required=True)
    p.add_argument("--batch-size",type=int,default=4); p.add_argument("--max-length",type=int,default=256); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output",type=Path,required=True); p.add_argument("--cache",type=Path,default=Path("cache/native")); p.add_argument("--dry-run",action="store_true")
    a=p.parse_args(argv); calibration=load_calibration(a.calibration); bundle=load_bundle(a.data,a.task) if a.task!="STSB" and a.data else None
    plan=task_plan(a.task,a.model,a.dimensions,None,a.seed,a.batch_size,a.max_length,None,bundle,LOADING_STRATEGY,calibration)
    if a.dry_run: print(json.dumps(plan,indent=2)); return
    if not plan["executable"]: p.error("Execution blocked: "+" ".join(plan["blockers"]))
    if a.task!="STSB" and bundle is None: p.error("--data is required for this task")
    from huggingface_hub import HfApi
    from pilot.model import FrozenEncoder,check_gpu
    from pilot.core import cached_embeddings,checked,normalize,project
    from pilot.task_runner import evaluate_bundle,fit_shared_pca,save_task_outputs
    import torch
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    hardware=check_gpu(); api=HfApi(); revision=api.model_info(plan["model_id"]).sha
    encoder=FrozenEncoder(a.model,revision,a.batch_size,a.max_length); identity={"model":plan["model_id"],"revision":revision,"width":plan["native_dimension"],"max_length":a.max_length,"quantization":plan["quantization"]}
    def encode(texts): return cached_embeddings(a.cache,identity,texts,encoder.encode)[0]
    started=time.perf_counter()
    if a.task=="STSB":
        from datasets import load_dataset
        from scipy.stats import spearmanr
        dsrev=api.dataset_info("mteb/stsbenchmark-sts").sha; ds=load_dataset("mteb/stsbenchmark-sts",revision=dsrev); rows0=list(ds["validation"])
        texts=list(dict.fromkeys([r[k] for r in rows0 for k in ("sentence1","sentence2")])); lookup={t:i for i,t in enumerate(texts)}; x=checked(encode(texts),len(texts),plan["native_dimension"]); pca=fit_shared_pca(calibration,plan,encode)
        rows=[]; preds={}
        for dim in plan["dimensions"]:
            z=normalize(x) if dim==plan["native_dimension"] else project(x,pca,dim); sims=np.array([float(z[lookup[r["sentence1"]]]@z[lookup[r["sentence2"]]]) for r in rows0]); score=float(spearmanr(sims,[r["score"] for r in rows0]).statistic)
            rows.append({"model":a.model,"task":"STSB","dataset":"mteb/stsbenchmark-sts","dataset_revision":dsrev,"split":"validation","dimension":dim,"reduction":"native" if dim==plan["native_dimension"] else "pca","metric":"cosine_spearman","score":score,"seed":a.seed,"protocol":"sts-full-validation-v2","n_eval":len(rows0)}); preds[str(dim)]=sims.tolist()
        meta={"schema_version":2,"plan":plan,"model_revision":revision,"dataset_revision":dsrev,"calibration_hash":calibration["manifest_hash"]}; save_task_outputs(a.output,rows,{"by_dimension":preds},meta,pca)
    else:
        rows,preds,meta,pca=evaluate_bundle(bundle,plan,encode,calibration); meta.update(model_revision=revision,hardware=hardware,seconds_before_export=time.perf_counter()-started); save_task_outputs(a.output,rows,preds,meta,pca)
    print(f"Saved results to {a.output}")
if __name__=="__main__":
    try: main()
    except (ValueError,RuntimeError,ImportError,OSError) as e: raise SystemExit(f"Experiment stopped: {e}") from e
