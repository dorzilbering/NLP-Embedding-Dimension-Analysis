"""ArxivClusteringS2S official-set evaluation."""
from pilot.task_data import fingerprint
DATASET_ID="mteb/arxiv-clustering-s2s"; TASK_NAME="ArxivClusteringS2S"

def validate_arxiv_bundle(bundle):
    if not isinstance(bundle,dict) or bundle.get("schema_version")!=1 or bundle.get("task")!="Arxiv-Clustering": raise ValueError("Unexpected Arxiv bundle.")
    if "train" in bundle or "evaluation" in bundle: raise ValueError("Legacy Arxiv train/evaluation bundle fields are not supported.")
    source=bundle.get("source",{}); sets=bundle.get("sets")
    if source.get("dataset_id")!=DATASET_ID or source.get("task_name")!=TASK_NAME or source.get("evaluation_split")!="test": raise ValueError("Expected official Arxiv test provenance.")
    if not isinstance(sets,list) or not sets: raise ValueError("Missing clustering sets.")
    for i,g in enumerate(sets):
        sentences=g.get("sentences",[]); labels=g.get("labels",[])
        if g.get("id")!=f"test:{i}" or not sentences or len(sentences)!=len(labels):
            raise ValueError(f"Invalid clustering set test:{i} (sentences={len(sentences)}, labels={len(labels)}).")
    if source.get("sets_hash")!=fingerprint(sets): raise ValueError("Arxiv set hash mismatch.")
    return bundle

def build_bundle(rows,revision,configuration):
    sets=[{"id":f"test:{i}","sentences":r["sentences"],"labels":r["labels"]} for i,r in enumerate(rows)]
    degenerate=[g["id"] for g in sets if len(set(g["labels"]))<2]
    bundle={"schema_version":1,"task":"Arxiv-Clustering","sets":sets,"source":{"dataset_id":DATASET_ID,"task_name":TASK_NAME,"configuration":configuration,"revision":revision,"evaluation_split":"test","selection":"full","set_count":len(sets),"sets_hash":fingerprint(sets),"degenerate_sets":degenerate}}
    return validate_arxiv_bundle(bundle)

def evaluate_arxiv(bundle,plan,encode,pca):
    from pilot.core import checked,normalize,project
    from pilot.evaluation import clustering
    validate_arxiv_bundle(bundle); width=plan["native_dimension"]; per_dim={}; predictions={}; evaluable=[g for g in bundle["sets"] if len(set(g["labels"]))>=2]
    if not evaluable: raise ValueError("No evaluable Arxiv clustering sets with at least two classes.")
    for dim in plan["dimensions"]:
        per_set=[]; predictions[str(dim)]={}
        for g in evaluable:
            x=checked(encode(g["sentences"]),len(g["sentences"]),width); x=normalize(x) if dim==width else project(x,pca,dim)
            metrics,pred=clustering(x,g["labels"],plan["seed"]); per_set.append(metrics); predictions[str(dim)][g["id"]]=pred
        per_dim[dim]=per_set
    total=sum(len(g["sentences"]) for g in evaluable); rows=[]
    for dim,setscores in per_dim.items():
        for metric in plan["protocol"]["metrics"]: rows.append({"model":plan["model"],"task":"Arxiv-Clustering","dataset":DATASET_ID,"dataset_revision":bundle["source"]["revision"],"split":"test","dimension":dim,"reduction":"native" if dim==width else "pca","metric":metric,"score":sum(s[metric] for s in setscores)/len(setscores),"seed":plan["seed"],"protocol":plan["protocol"]["id"],"n_eval":total})
    meta={"schema_version":2,"plan":plan,"source":bundle["source"],"bundle_hash":fingerprint(bundle),"aggregation":"unweighted arithmetic mean over non-degenerate official sets","group_order":[g["id"] for g in evaluable],"excluded_degenerate_sets":[g["id"] for g in bundle["sets"] if len(set(g["labels"]))<2]}
    return rows,{"by_dimension":predictions},meta
