"""Synthetic tests for the assignment-wide task/calibration architecture."""
import json
import numpy as np
import pytest

from pilot.config import MODEL_SPECS
from pilot.task_config import PROTOCOLS, task_plan
from pilot.task_data import validate_bundle
from pilot.task_runner import evaluate_bundle


def banking_bundle():
    source={"dataset_id":"PolyAI/banking77","revision":"fixture-sha","configuration":"default","evaluation_split":"test","selection":"full","text_format":"plain"}
    train=[{"id":f"train:{i}","text":f"train {i}","label":str(i%77)} for i in range(154)]
    evaluation=[{"id":f"test:{i}","text":f"test {i}","label":str(i%77)} for i in range(77)]
    return {"schema_version":1,"task":"Banking77","source":source,"fit_source":{"dataset_id":source["dataset_id"],"revision":source["revision"],"configuration":"default","split":"train","role":"train","split_method":"official"},"train":train,"evaluation":evaluation}


def calibration(n=5000):
    return {"schema_version":1,"role":"shared_pca_calibration","texts":[{"id":f"c:{i}","text":f"calibration text {i}"} for i in range(n)]}


def test_all_five_models_are_implemented():
    assert set(MODEL_SPECS)=={"Qwen3-8B","Gemma3-4B","Llama3.2-3B","Phi4-mini","Mistral-7B"}
    for model,spec in MODEL_SPECS.items():
        assert spec["model_id"] and spec["dimensions"][0]==spec["native_dimension"]


def test_shared_calibration_unblocks_reduced_dimensions():
    data=banking_bundle(); cal=calibration()
    for model,spec in MODEL_SPECS.items():
        plan=task_plan("Banking77",model=model,bundle=data,calibration=cal)
        assert plan["implemented"] and plan["executable"]
        assert plan["dimensions"]==list(spec["dimensions"])
        assert plan["pca_components"]==max(spec["dimensions"][1:])


def test_reduced_dimensions_require_calibration():
    plan=task_plan("Banking77",model="Phi4-mini",bundle=banking_bundle())
    assert not plan["executable"]
    assert any("calibration" in b.lower() for b in plan["blockers"])


def test_native_only_does_not_require_calibration():
    data=banking_bundle()
    plan=task_plan("Banking77",model="Phi4-mini",dimensions=[3072],bundle=data)
    assert plan["executable"] and plan["pca_components"]==0


def test_banking_official_text_overlap_is_recorded_not_rejected():
    data=banking_bundle(); data["evaluation"][0]["text"]=data["train"][0]["text"]
    assert validate_bundle(data,"Banking77") is data


def test_conflicting_duplicate_training_labels_rejected():
    data=banking_bundle(); data["train"][1]["text"]=data["train"][0]["text"]; data["train"][1]["label"]="different"
    with pytest.raises(ValueError,match="Conflicting"):
        validate_bundle(data,"Banking77")


def test_engine_fits_pca_only_on_calibration(monkeypatch):
    import pilot.task_runner as runner
    data=banking_bundle(); cal=calibration(20)
    plan={"task":"Banking77","model":"synthetic","native_dimension":12,"dimensions":[12,4,2],"pca_components":4,"seed":42,"protocol":PROTOCOLS["Banking77"]}
    rng=np.random.default_rng(1); mapping={r["text"]:rng.normal(size=12).astype(np.float32) for r in cal["texts"]+data["train"]+data["evaluation"]}
    fitted=[]; original=runner.fit_pca
    def fit(x,*args): fitted.append(x.copy()); return original(x,*args)
    monkeypatch.setattr(runner,"fit_pca",fit)
    rows,_,meta,pca=evaluate_bundle(data,plan,lambda texts:np.asarray([mapping[t] for t in texts]),cal)
    assert len(fitted)==1 and fitted[0].shape==(20,12) and pca.components_.shape==(4,12)
    assert len(rows)==3*len(PROTOCOLS["Banking77"]["metrics"])
    assert meta["pca"]["policy"].startswith("shared training-only")


def test_arxiv_reduced_plan_is_no_longer_blocked_by_task_design():
    # Data absence remains a blocker; PCA policy itself is assignment-wide rather than Arxiv-test-fitted.
    plan=task_plan("Arxiv-Clustering",model="Phi4-mini",calibration=calibration())
    assert any("--data" in b for b in plan["blockers"])
    assert not any("Arxiv" in b and "PCA" in b for b in plan["blockers"])
