"""Task plans for the complete assignment."""
from pilot.config import MODEL_SPECS,TASK_SPECS,LOADING_STRATEGY,validate_dimensions
from pilot.task_data import fingerprint,validate_bundle
PROTOCOLS={
 "STSB":{"id":"sts-validation-pilot-v1","metrics":["cosine_spearman"]},
 "SciFact":{"id":"exact-cosine-linear-ndcg-v2","metrics":["ndcg_at_10","recall_at_10","recall_at_100","mrr_at_10"]},
 "Banking77":{"id":"banking77-official-train-test-logistic-v2","metrics":["accuracy","macro_f1"]},
 "Arxiv-Clustering":{"id":"arxiv-s2s-per-set-minibatch-kmeans-v2","metrics":["v_measure","adjusted_rand","nmi"]}}

def task_plan(task,model="Phi4-mini",dimensions=None,train_count=None,seed=42,batch_size=4,max_length=256,clusters=None,bundle=None,loading_strategy=LOADING_STRATEGY,calibration=None):
    if task not in TASK_SPECS or model not in MODEL_SPECS: raise ValueError("Unknown assignment task/model.")
    spec=MODEL_SPECS[model]; dims=validate_dimensions(spec["dimensions"] if dimensions is None else dimensions,spec["native_dimension"])
    if set(dims)-set(spec["dimensions"]): raise ValueError("Only assignment-required dimensions are supported.")
    components=max((d for d in dims if d<spec["native_dimension"]),default=0)
    if batch_size<1 or not 8<=max_length<=512 or loading_strategy!=LOADING_STRATEGY: raise ValueError("Invalid execution settings.")
    if clusters is not None: raise ValueError("Arxiv K is determined per official set.")
    blockers=[]; summary=None
    if task!="STSB":
        if bundle is None: blockers.append("Provide the prepared official task bundle with --data.")
        else:
            validate_bundle(bundle,task); summary={"source":bundle["source"],"bundle_hash":fingerprint(bundle)}
    if components:
        if calibration is None: blockers.append("Provide the shared training-only PCA calibration bundle with --calibration.")
        else:
            texts=calibration.get("texts",[]) if isinstance(calibration,dict) else []
            if calibration.get("role")!="shared_pca_calibration" or len(texts)<=components: blockers.append(f"Calibration must contain >{components} eligible texts.")
    return {"model":model,"model_id":spec["model_id"],"task":task,"native_dimension":spec["native_dimension"],"dimensions":list(dims),
            "pca_components":components,"seed":seed,"batch_size":batch_size,"max_length":max_length,"loading_strategy":loading_strategy,
            "quantization":spec.get("t4_quantization"),"protocol":PROTOCOLS[task],"data":summary,"implemented":True,"executable":not blockers,
            "blockers":blockers,"reduction_policy":"one PCA basis per model fitted on shared training-only calibration and reused across tasks"}
