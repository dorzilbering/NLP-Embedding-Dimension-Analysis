"""Prepare official Banking77 data only; never imports/loads a language model."""
import argparse, importlib.metadata, json, re
from pathlib import Path
from pilot.banking77 import DATASET_ID, NUM_CLASSES, SPLIT_COUNTS, build_bundle

# First official repository revision containing script-free Parquet exports.
PARQUET_REVISION = "0b62e79500b84f0fff5a36a4404028f48fbc8621"

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--revision",default=PARQUET_REVISION); parser.add_argument("--configuration",default="default"); parser.add_argument("--output",type=Path,default=Path("data/banking77_official.json")); parser.add_argument("--dry-run",action="store_true"); args=parser.parse_args(argv)
    if args.dry_run:
        print(json.dumps({"dataset":DATASET_ID,"requested_revision":args.revision,"configuration":args.configuration,"official_split_counts":SPLIT_COUNTS,"classes":NUM_CLASSES,"fields":["text","label"],"output":str(args.output),"network_access":False,"dataset_contents_verified":False},indent=2)); return
    if args.output.exists(): parser.error("Output already exists; choose a new path. Existing bundles are never overwritten.")
    from huggingface_hub import HfApi
    from datasets import load_dataset
    revision=HfApi().dataset_info(DATASET_ID,revision=args.revision).sha
    if not isinstance(revision,str) or not re.fullmatch(r"[0-9a-fA-F]{40}",revision): raise ValueError("Could not resolve an immutable dataset commit SHA.")
    # Explicit Parquet paths avoid the legacy banking77.py dataset script, unsupported by current datasets releases.
    base=f"hf://datasets/{DATASET_ID}@{revision}/data"
    dataset=load_dataset("parquet",data_files={"train":f"{base}/train-00000-of-00001.parquet","test":f"{base}/test-00000-of-00001.parquet"})
    bundle=build_bundle(dataset,revision,args.configuration); bundle["preparation"]={"packages":{name:importlib.metadata.version(name) for name in ("datasets","huggingface-hub")},"transport":"official repository Parquet"}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(bundle,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"saved":str(args.output),"revision":revision,"configuration":args.configuration,"counts":SPLIT_COUNTS,"train_unique_texts":bundle["source"]["train_unique_texts"]},indent=2))

if __name__=="__main__":
    try: main()
    except (ValueError,RuntimeError,ImportError,OSError) as error: raise SystemExit(f"Banking77 preparation stopped: {error}") from error
