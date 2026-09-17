"""Offline tests for ArxivClusteringS2S under the assignment-wide shared PCA protocol."""
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace
import numpy as np
import pytest
from pilot.arxiv import build_bundle, validate_arxiv_bundle
from pilot.evaluation import clustering
from pilot.task_config import task_plan
from pilot.task_runner import evaluate_bundle, save_task_outputs

@pytest.fixture(autouse=True)
def offline(monkeypatch):
    monkeypatch.setattr(socket.socket, "connect", lambda *a, **k: (_ for _ in ()).throw(AssertionError("Network forbidden in arXiv tests")))

@pytest.fixture
def bundle():
    return build_bundle([{"sentences":["a","b","c","d"],"labels":["x","x","y","y"]},{"sentences":["a","e"],"labels":["p","q"]}], "a"*40, "default")

def calibration(n=2000):
    texts=[{"id":f"c:{i}","text":f"cal {i}"} for i in range(n)]
    from pilot.task_data import fingerprint
    return {"schema_version":1,"role":"shared_pca_calibration","texts":texts,"manifest_hash":fingerprint(texts)}

def test_standard_clustering_deterministic_and_label_names_do_not_transform(monkeypatch):
    from sklearn.cluster import MiniBatchKMeans
    original,calls=MiniBatchKMeans.fit_predict,[]
    def spy(self,x,*args,**kwargs): assert not args and not kwargs; calls.append((x.copy(),self.n_clusters,self.random_state)); return original(self,x)
    monkeypatch.setattr(MiniBatchKMeans,"fit_predict",spy)
    x=np.array([[1,.01],[1,-.01],[-1,.01],[-1,-.01]])
    first=clustering(x,["a","a","b","b"],42); second=clustering(x,["renamed-a","renamed-a","renamed-b","renamed-b"],42)
    assert first==second and first[0]=={"v_measure":1.,"adjusted_rand":1.,"nmi":1.}
    assert all(len(v)==4 and k==2 and seed==42 for v,k,seed in calls)

def test_per_set_macro_aggregation_and_export(bundle,monkeypatch,tmp_path):
    import pilot.evaluation as evaluator
    seen=[]
    def score(x,labels,seed): seen.append((len(x),list(labels),seed)); value=1. if len(x)==4 else 0.; return dict(v_measure=value,adjusted_rand=value,nmi=value),[0]*len(x)
    monkeypatch.setattr(evaluator,"clustering",score)
    plan=task_plan("Arxiv-Clustering",dimensions=[3072],bundle=bundle)
    encoded=[]
    def encode(texts): encoded.append(list(texts)); return np.ones((len(texts),3072),dtype=np.float32)
    rows,predictions,metadata,pca=evaluate_bundle(bundle,plan,encode)
    assert [r["score"] for r in rows]==[.5,.5,.5]
    assert [len(t) for t in encoded]==[4,2] and len(seen)==2
    assert predictions["by_dimension"]["3072"]["test:0"]==[0]*4
    assert metadata["aggregation"]=="unweighted arithmetic mean over official sets" and pca is None
    save_task_outputs(tmp_path/"out",rows,predictions,metadata,pca); assert not (tmp_path/"out"/"pca.npz").exists()

def test_reduced_dimensions_require_shared_calibration(bundle):
    for dimension in (1536,768,384):
        blocked=task_plan("Arxiv-Clustering",dimensions=[3072,dimension],bundle=bundle)
        assert not blocked["executable"] and any("calibration" in b.lower() for b in blocked["blockers"])
        allowed=task_plan("Arxiv-Clustering",dimensions=[3072,dimension],bundle=bundle,calibration=calibration())
        assert allowed["executable"]
    with pytest.raises(ValueError,match="no longer supported"): task_plan("Arxiv-Clustering",dimensions=[3072],clusters=2,bundle=bundle)

@pytest.mark.parametrize("fault",["alignment","single_class","legacy","hash","split","id"])
def test_invalid_structure_rejected(bundle,fault):
    if fault=="alignment": bundle["sets"][0]["labels"].pop()
    elif fault=="single_class": bundle["sets"][0]["labels"]=["a"]*4
    elif fault=="legacy": bundle["train"]=[]
    elif fault=="hash": bundle["source"]["sets_hash"]="wrong"
    elif fault=="split": bundle["source"]["evaluation_split"]="train"
    else: bundle["sets"][1]["id"]="test:0"
    with pytest.raises(ValueError): validate_arxiv_bundle(bundle)

def test_mocked_preparation_and_config_without_models(bundle,monkeypatch,tmp_path,capsys):
    import prepare_arxiv,run_experiment
    class Rows(list): column_names=["sentences","labels"]
    rows=Rows({"sentences":g["sentences"],"labels":g["labels"]} for g in bundle["sets"])
    monkeypatch.setitem(sys.modules,"huggingface_hub",SimpleNamespace(HfApi=lambda:SimpleNamespace(dataset_info=lambda *a,**k:SimpleNamespace(sha="a"*40))))
    monkeypatch.setitem(sys.modules,"datasets",SimpleNamespace(get_dataset_config_names=lambda *a,**k:["default"],load_dataset=lambda identifier,name,revision:{"test":rows}))
    path=tmp_path/"arxiv.json"; prepare_arxiv.main(["--output",str(path)]); assert json.loads(path.read_text())==bundle; capsys.readouterr()
    run_experiment.main(["--task","Arxiv-Clustering","--model","Phi4-mini","--data",str(path),"--dry-run"])
    assert not json.loads(capsys.readouterr().out)["executable"]
    run_experiment.main(["--task","Arxiv-Clustering","--model","Phi4-mini","--data",str(path),"--dimensions","3072","--dry-run"])
    assert json.loads(capsys.readouterr().out)["executable"]
    before=path.read_bytes()
    with pytest.raises(SystemExit): prepare_arxiv.main(["--output",str(path)])
    assert path.read_bytes()==before

def test_preparation_stdlib_dry_run(tmp_path):
    script=Path(__file__).resolve().parents[1]/"prepare_arxiv.py"; proc=subprocess.run([sys.executable,"-B","-S",str(script),"--dry-run"],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr; assert json.loads(proc.stdout)["task"]=="ArxivClusteringS2S"
