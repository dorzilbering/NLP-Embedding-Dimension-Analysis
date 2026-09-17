"""Resolve all assignment checkpoints before spending GPU time."""
import json
from pilot.config import MODEL_SPECS

def main():
    from huggingface_hub import HfApi
    api=HfApi(); rows=[]
    for name,spec in MODEL_SPECS.items():
        try:
            info=api.model_info(spec["model_id"]); rows.append({"model":name,"model_id":spec["model_id"],"accessible":True,"revision":info.sha})
        except Exception as e:
            rows.append({"model":name,"model_id":spec["model_id"],"accessible":False,"error":f"{type(e).__name__}: {e}"})
    print(json.dumps(rows,indent=2))
    if not all(r["accessible"] for r in rows): raise SystemExit(2)
if __name__=="__main__": main()
