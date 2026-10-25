
from __future__ import annotations
import csv, hashlib, importlib.metadata, json, platform, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_FILES = [
    ROOT / "data/raw/BIORED.zip",
    ROOT / "data/raw/bc5cdr/CDR_TrainingSet.PubTator.txt",
    ROOT / "data/raw/bc5cdr/CDR_DevelopmentSet.PubTator.txt",
    ROOT / "data/raw/bc5cdr/CDR_TestSet.PubTator.txt",
]
HF_MODELS = [
    "alvaroalon2/biobert_diseases_ner",
    "alvaroalon2/biobert_chemical_ner",
    "sarahmiller137/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext-ft-ncbi-disease",
    "OpenMed/OpenMed-NER-ChemicalDetect-PubMed-335M",
    "samrawal/bert-base-uncased_clinical-ner",
    "d4data/biomedical-ner-all",
    "dmis-lab/biobert-v1.1",
    "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    "bert-base-uncased",
    "distilbert-base-uncased",
]
PACKAGES = ["numpy","matplotlib","scikit-learn","spacy","scispacy","transformers","tokenizers","sentencepiece","tiktoken","psutil","torch","torchvision","torchaudio"]

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def generate() -> Path:
    out=ROOT/'results'/'reproducibility'
    out.mkdir(parents=True, exist_ok=True)
    rows=[]
    for p in RAW_FILES:
        rows.append({"file":str(p.relative_to(ROOT)),"exists":p.exists(),"size_bytes":p.stat().st_size if p.exists() else None,"sha256":sha256(p) if p.exists() else None})
    with (out/'raw_data_sha256.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    versions={}
    for name in PACKAGES:
        try: versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name]=None
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        cuda = torch.version.cuda
    except Exception:
        gpu=cuda=None
    model_revisions=[]
    try:
        from huggingface_hub import model_info
        for model in HF_MODELS:
            try:
                info=model_info(model)
                model_revisions.append({"model":model,"resolved_sha":info.sha})
            except Exception as exc:
                model_revisions.append({"model":model,"resolved_sha":None,"error":str(exc)})
    except Exception as exc:
        model_revisions=[{"model":m,"resolved_sha":None,"error":f"huggingface_hub unavailable: {exc}"} for m in HF_MODELS]
    payload={
        "python":sys.version,
        "platform":platform.platform(),
        "packages":versions,
        "cuda_runtime":cuda,
        "gpu":gpu,
        "models":model_revisions,
        "raw_data":rows,
        "notes":[
            "The repository records the exact resolved Hugging Face revision used by each run.",
            "Seed 42 and DEV-only model selection are implemented in the experiment code.",
            "CUDA/FP16 training can show small numerical variation across driver/kernel versions even with a fixed seed."
        ],
    }
    (out/'provenance.json').write_text(json.dumps(payload,indent=2)+"\n",encoding='utf-8')
    try:
        freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True,errors='replace')
        (out/'pip_freeze.txt').write_text(freeze,encoding='utf-8')
    except Exception as exc:
        (out/'pip_freeze.txt').write_text(f"pip freeze failed: {exc}\n",encoding='utf-8')
    print(f"Reproducibility snapshot: {out}")
    return out

if __name__=='__main__': generate()
