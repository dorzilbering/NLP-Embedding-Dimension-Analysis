"""Official-shape synthetic Banking77 exports; no real dataset/model downloads."""
import json
from pathlib import Path
import socket,subprocess,sys
from types import SimpleNamespace
import pytest
from pilot.banking77 import build_bundle,validate_official_bundle,SPLIT_COUNTS
from pilot.task_config import task_plan
from pilot.task_data import fingerprint

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket,"connect",lambda *a,**k:(_ for _ in ()).throw(AssertionError("Network forbidden")))
    for key in ("HF_HUB_OFFLINE","HF_DATASETS_OFFLINE","TRANSFORMERS_OFFLINE"): monkeypatch.setenv(key,"1")
class Partition(list):
    column_names=["text","label"]; features={"label":SimpleNamespace(names=[f"intent_{i}" for i in range(77)])}
@pytest.fixture
def dataset(): return {s:Partition({"text":f"synthetic {s} text {i}","label":i%77} for i in range(n)) for s,n in SPLIT_COUNTS.items()}
def export(dataset): return build_bundle(dataset,"a"*40,"fixture-config")
def calibration(n=3000):
    texts=[{"id":f"c:{i}","text":f"cal {i}"} for i in range(n)]; return {"schema_version":1,"role":"shared_pca_calibration","texts":texts,"manifest_hash":fingerprint(texts)}

def test_full_export_and_plan(dataset):
    data=export(dataset); assert len(data["train"])==10003 and len(data["evaluation"])==3080; assert data["evaluation"][-1]["id"]=="test:3079"
    plan=task_plan("Banking77",bundle=data,calibration=calibration()); assert plan["executable"] and plan["pca_components"]==1536
    assert plan["protocol"]["classifier"]==dict(C=1.,solver="lbfgs",max_iter=1000,tol=1e-4,class_weight=None)

@pytest.mark.parametrize("fault",["count","schema","label","classes","splits","conflict"])
def test_bad_source_rejected(dataset,fault):
    if fault=="count": dataset["train"].pop()
    elif fault=="schema": dataset["test"].column_names=["text","intent"]
    elif fault=="label": dataset["test"][0]["label"]=77
    elif fault=="classes": dataset["test"].features={"label":SimpleNamespace(names=["bad"]*77)}
    elif fault=="splits": dataset["validation"]=dataset.pop("test")
    else: dataset["train"][1]["text"]=dataset["train"][0]["text"]
    with pytest.raises(ValueError): export(dataset)

def test_official_cross_split_text_overlap_is_preserved_and_recorded(dataset):
    dataset["test"][0]["text"]=dataset["train"][0]["text"].upper(); data=export(dataset); validate_official_bundle(data)
    assert data.get("validation_notes",{}).get("normalized_train_test_text_overlap_count",0)==1

@pytest.mark.parametrize("fault",["revision","source","split","subset","hash","row_order"])
def test_invalid_export_rejected(dataset,fault):
    data=export(dataset)
    if fault=="revision": data["source"]["revision"]=data["fit_source"]["revision"]="not-a-sha"
    elif fault=="source": data["source"]["dataset_id"]=data["fit_source"]["dataset_id"]="wrong/source"
    elif fault=="split": data["source"]["evaluation_split"]="validation"
    elif fault=="subset": data["source"]["selection"]="subset"
    elif fault=="hash": data["evaluation"][0]["text"]="altered text"
    else: data["train"][0],data["train"][1]=data["train"][1],data["train"][0]
    with pytest.raises(ValueError): validate_official_bundle(data)

def test_duplicate_training_rows_preserved(dataset):
    dataset["train"][77]=dict(dataset["train"][0]); data=export(dataset); assert len(data["train"])==10003 and data["source"]["train_unique_texts"]==10002
    assert task_plan("Banking77",bundle=data,calibration=calibration())["executable"]

def test_shared_pca_is_separate_from_classifier_training(monkeypatch):
    import numpy as np, pilot.task_runner as runner
    from pilot.task_config import PROTOCOLS
    rng=np.random.default_rng(3); train=[dict(id=f"t{i}",text=f"t{i}",label=str(i%2)) for i in range(12)]; train.append(dict(id="duplicate",text="t0",label="0")); evaluation=[dict(id=f"e{i}",text=f"e{i}",label=str(i)) for i in range(2)]
    source=dict(dataset_id="synthetic",revision="fixture-v1",configuration="fixture",evaluation_split="test",selection="subset",text_format="plain")
    data=dict(schema_version=1,task="Banking77",source=source,train=train,evaluation=evaluation,fit_source=dict(dataset_id="synthetic",revision="fixture-v1",configuration="fixture",split="train",role="train",split_method="synthetic"))
    cal=calibration(12); mapping={r["text"]:rng.normal(size=8) for r in train+evaluation+cal["texts"]}; classifier_rows=[]
    monkeypatch.setattr(runner,"validate_bundle",lambda b,t:b); monkeypatch.setattr(runner,"classification",lambda x,l,ex,el,s:(classifier_rows.append(len(x)) or {"accuracy":0.,"macro_f1":0.},["0","0"]))
    plan=dict(task="Banking77",model="synthetic",native_dimension=8,seed=42,dimensions=[8,4,2],pca_components=4,protocol=PROTOCOLS["Banking77"])
    runner.evaluate_bundle(data,plan,lambda texts:np.asarray([mapping[t] for t in texts]),cal); assert classifier_rows==[13,13,13]

def test_preparation_dry_run_stdlib_only(tmp_path):
    script=Path(__file__).resolve().parents[1]/"prepare_banking77.py"; output=tmp_path/"unused.json"; proc=subprocess.run([sys.executable,"-B","-S",str(script),"--dry-run","--output",str(output)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr; assert json.loads(proc.stdout)["official_split_counts"]==SPLIT_COUNTS and not output.exists()

def test_preparation_mocked_download_and_immutable_revision(dataset,tmp_path,monkeypatch,capsys):
    import prepare_banking77
    calls=[]
    def info(identifier,revision): calls.append(("resolve",identifier,revision)); return SimpleNamespace(sha="a"*40)
    def load(identifier,**kwargs): calls.append(("load",identifier,kwargs)); return dataset
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(HfApi=lambda:SimpleNamespace(dataset_info=info)))
    monkeypatch.setitem(sys.modules,"datasets",SimpleNamespace(load_dataset=load))
    output=tmp_path/"bundle.json"; prepare_banking77.main(["--output",str(output)]); data=json.loads(output.read_text()); validate_official_bundle(data); assert task_plan("Banking77",bundle=data,calibration=calibration())["executable"]
    assert calls[0][0]=="resolve" and calls[1][0:2]==("load","parquet") and set(calls[1][2]["data_files"])=={"train","test"}

def test_full_local_bundle_dry_run(dataset,tmp_path,capsys):
    import run_experiment
    path=tmp_path/"bank.json"; path.write_text(json.dumps(export(dataset)),encoding="utf-8")
    run_experiment.main(["--task","Banking77","--model","Phi4-mini","--data",str(path),"--dimensions","3072","--dry-run"])
    plan=json.loads(capsys.readouterr().out); assert plan["executable"]
    assert plan["data"]["source"]["official_counts"]==SPLIT_COUNTS
