"""Shared task evaluation; caller supplies native encoder and shared PCA calibration."""
import csv,json
from pathlib import Path
import numpy as np
from pilot.core import checked,fit_pca,normalize,project
from pilot.evaluation import classification,retrieval
from pilot.task_data import fingerprint,validate_bundle

def fit_shared_pca(calibration,plan,encode):
    if not plan["pca_components"]: return None
    texts=calibration["texts"]; x=checked(encode(texts),len(texts),plan["native_dimension"])
    return fit_pca(x,plan["pca_components"],plan["seed"])

def evaluate_bundle(bundle,plan,encode,calibration):
    task=plan["task"]; validate_bundle(bundle,task); pca=fit_shared_pca(calibration,plan,encode)
    if task=="Arxiv-Clustering":
        from pilot.arxiv import evaluate_arxiv
        rows,pred,meta=evaluate_arxiv(bundle,plan,encode,pca); meta["calibration_hash"]=calibration["manifest_hash"]; return rows,pred,meta,pca
    width=plan["native_dimension"]
    if task=="SciFact": groups={"queries":bundle["queries"],"corpus":bundle["corpus"]}
    else: groups={"fit":bundle["train"],"evaluation":bundle["evaluation"]}
    vectors={n:checked(encode([r["text"] for r in rs]),len(rs),width) for n,rs in groups.items()}
    rows=[]; predictions={}
    for dim in plan["dimensions"]:
        transformed={n:(normalize(x) if dim==width else project(x,pca,dim)) for n,x in vectors.items()}
        if task=="SciFact":
            scores,pred=retrieval(transformed["queries"],[r["id"] for r in bundle["queries"]],transformed["corpus"],[r["id"] for r in bundle["corpus"]],bundle["qrels"]); n_eval=len(bundle["queries"])
        elif task=="Banking77":
            scores,pred=classification(transformed["fit"],[r["label"] for r in bundle["train"]],transformed["evaluation"],[r["label"] for r in bundle["evaluation"]],plan["seed"]); n_eval=len(bundle["evaluation"])
        else: raise ValueError("Unknown task")
        predictions[str(dim)]=pred
        for metric,score in scores.items(): rows.append({"model":plan["model"],"task":task,"dataset":bundle["source"]["dataset_id"],"dataset_revision":bundle["source"]["revision"],"split":bundle["source"]["evaluation_split"],"dimension":dim,"reduction":"native" if dim==width else "pca","metric":metric,"score":float(score),"seed":plan["seed"],"protocol":plan["protocol"]["id"],"n_eval":n_eval})
    meta={"schema_version":2,"plan":plan,"bundle_hash":fingerprint(bundle),"source":bundle["source"],"calibration_hash":calibration["manifest_hash"],"calibration_count":len(calibration["texts"]),"pca":{"components":plan["pca_components"],"pre_normalize":True,"whiten":False,"solver":"randomized"}}
    return rows,{"by_dimension":predictions},meta,pca

def write_metrics(path,rows):
    with Path(path).open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def save_task_outputs(output,rows,predictions,metadata,pca):
    output=Path(output); output.mkdir(parents=True,exist_ok=True); write_metrics(output/"metrics.csv",rows)
    (output/"results.json").write_text(json.dumps({"metadata":metadata,"scores":rows},indent=2,allow_nan=False),encoding="utf-8")
    (output/"predictions.json").write_text(json.dumps(predictions,indent=2,allow_nan=False),encoding="utf-8")
    if pca is not None: np.savez(output/"pca.npz",mean=pca.mean_,components=pca.components_,explained_variance_ratio=pca.explained_variance_ratio_)
def sts_metrics(result):
    return [{"model":"Phi4-mini","task":"STSB","dataset":r["dataset"],"dataset_revision":result["metadata"]["dataset_revision"],"split":r["split"],"dimension":r["dimension"],"reduction":r["reduction"],"metric":r["metric"],"score":r["cosine_spearman"],"seed":r["seed"],"protocol":"sts-validation-pilot-v1","n_eval":r["pairs"]} for r in result["scores"]]
