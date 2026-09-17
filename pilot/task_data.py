"""Local, provenance-labelled task data. Validation is offline and read-only."""
import hashlib, json, math, unicodedata
from pathlib import Path
NEW_TASKS = ("SciFact", "Banking77", "Arxiv-Clustering")

def text_key(text): return " ".join(unicodedata.normalize("NFKC", text).casefold().split())
def fingerprint(value): return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
def _required_strings(obj,names,description):
    if not isinstance(obj,dict) or any(not isinstance(obj.get(n),str) or not obj[n].strip() for n in names): raise ValueError(f"{description} requires nonempty strings: {', '.join(names)}")
def _records(rows,role,labelled=False):
    if not isinstance(rows,list) or not rows: raise ValueError(f"{role} must be a nonempty list.")
    ids=set()
    for row in rows:
        _required_strings(row,["id","text"]+(["label"] if labelled else []),role)
        if not text_key(row["text"]) or row["id"] in ids: raise ValueError(f"Empty text or duplicate ID in {role}.")
        ids.add(row["id"])
    return rows

def _no_overlap(fit,evaluation,allow_training_duplicates=False,allow_cross_split_overlap=False):
    fit_keys=[text_key(r["text"]) for r in fit]
    if not allow_training_duplicates and len(set(fit_keys))!=len(fit_keys): raise ValueError("Fit/reference texts must be deduplicated before export.")
    overlap=set(fit_keys)&{text_key(r["text"]) for r in evaluation}
    if overlap and not allow_cross_split_overlap: raise ValueError("Leakage: fit/reference text overlaps evaluation data.")
    return overlap

def validate_bundle(bundle,task):
    if task=="Arxiv-Clustering":
        from pilot.arxiv import validate_arxiv_bundle; return validate_arxiv_bundle(bundle)
    if not isinstance(bundle,dict) or task not in NEW_TASKS or bundle.get("schema_version")!=1 or bundle.get("task")!=task: raise ValueError("Task data bundle has an unexpected schema/task.")
    source,fit_source=bundle.get("source"),bundle.get("fit_source")
    _required_strings(source,["dataset_id","revision","configuration","evaluation_split","selection","text_format"],"source")
    _required_strings(fit_source,["dataset_id","revision","configuration","split","role","split_method"],"fit_source")
    for provenance in (source,fit_source):
        if provenance["revision"].strip().casefold() in ("main","master","latest","todo","unknown","unverified"): raise ValueError("Record an immutable revision/checksum.")
    if task=="SciFact":
        reference=_records(bundle.get("reference"),"reference"); queries=_records(bundle.get("queries"),"queries"); corpus=_records(bundle.get("corpus"),"corpus")
        _no_overlap(reference,queries+corpus)
        qids,dids={r["id"] for r in queries},{r["id"] for r in corpus}; qrels=bundle.get("qrels")
        if not isinstance(qrels,dict) or set(qrels)!=qids: raise ValueError("Qrels must cover exactly the selected evaluation queries.")
        for qid,j in qrels.items():
            if not isinstance(j,dict) or not j or set(j)-dids: raise ValueError(f"Missing/unknown corpus IDs in qrels for {qid}.")
            if any(type(g) not in (int,float) or not math.isfinite(g) or g<0 for g in j.values()) or not any(g>0 for g in j.values()): raise ValueError("Invalid qrels.")
    else:
        fit=_records(bundle.get("train"),"train/reference",labelled=task=="Banking77"); evaluation=_records(bundle.get("evaluation"),"evaluation",labelled=True)
        overlap=_no_overlap(fit,evaluation,allow_training_duplicates=task=="Banking77",allow_cross_split_overlap=task=="Banking77")
        if {r["id"] for r in fit}&{r["id"] for r in evaluation}: raise ValueError("Training and evaluation IDs must be distinct.")
        if task=="Banking77":
            by_text={}
            for r in fit:
                key=text_key(r["text"]); previous=by_text.get(key)
                if previous is not None and previous!=r["label"]: raise ValueError("Conflicting labels for duplicate Banking77 training text.")
                by_text[key]=r["label"]
            labels={r["label"] for r in fit}
            if len(labels)!=77 or {r["label"] for r in evaluation}-labels: raise ValueError("Banking77 requires all 77 train classes and no unseen test class.")
            bundle.setdefault("validation_notes",{})["normalized_train_test_text_overlap_count"]=len(overlap)
    return bundle

def load_bundle(path,task):
    def unique_keys(pairs):
        out={}
        for k,v in pairs:
            if k in out: raise ValueError(f"Duplicate JSON key: {k}")
            out[k]=v
        return out
    return validate_bundle(json.loads(Path(path).read_text(encoding="utf-8-sig"),object_pairs_hook=unique_keys),task)
