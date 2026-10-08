"""Builds notebooks/training.ipynb and notebooks/inference.ipynb from the train.py / inference.py sections.
Run after editing either script: python make_notebooks.py"""
import json, re, sys, os

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
BANNER = re.compile(r'\n(?=# =+\n# [^\n]+\n# =+\n)')


def sections(path):
    src = open(os.path.join(ROOT, path), encoding='utf-8').read()
    src = src.split("\nif __name__ == '__main__':")[0]
    parts = BANNER.split(src)
    out = {'header': parts[0].strip()}
    for p in parts[1:]:
        out[p.split('\n')[1][2:].strip()] = p.strip()
    return out


def cell(kind, text):
    lines = text.strip('\n').split('\n')
    c = {'cell_type': kind, 'metadata': {}, 'source': [l + '\n' for l in lines[:-1]] + [lines[-1]]}
    if kind == 'code':
        c.update(execution_count=None, outputs=[])
    return c


def write(name, cells):
    nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                       'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
    with open(os.path.join(ROOT, 'notebooks', name), 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)


TIMM_CHECK = """# DINOv3 needs timm >= 1.0.20. Online: pip upgrade. Offline: attach a timm wheel as a dataset.
import glob, subprocess, sys
from importlib.metadata import version
if tuple(map(int, version('timm').split('.')[:3])) < (1, 0, 20):
    wheels = glob.glob('/kaggle/input/**/timm*.whl', recursive=True)
    target = wheels[0] if wheels else 'timm>=1.0.20'
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', target])
print('timm', version('timm'))"""

# ---------------- training notebook ----------------
t = sections('train.py')
train_cells = [cell('markdown', """# CSIRO Image2Biomass - Training (Dual-Stream DINOv3)
Generated from `train.py` - edit the script, not this notebook.

- Labels from `wide.csv` (attach as a dataset); 5-fold CV grouped by `Sampling_Date`, stratified by `State`, real images only; metric = official global weighted R2.
- Fixed epochs, SWA of the last epochs; metric-weighted MSE + interval classification.
- Set `--full_train` in the last cell to train one all-data model for submission.
- Outputs: `/kaggle/working/models/*.pt` (attach these to the inference notebook) and `oof_predictions.csv`."""),
               cell('code', TIMM_CHECK)]
train_cells += [cell('code', s) for k, s in t.items() if k != 'header']
train_cells.insert(2, cell('code', t['header']))
train_cells.append(cell('code', """# Kaggle run configuration: images from the competition data, labels from wide.csv (attach it as a dataset)
import glob
images = glob.glob('/kaggle/input/**/train.csv', recursive=True)
wide = glob.glob('/kaggle/input/**/wide.csv', recursive=True)
args = parse_args([
    '--data_path', wide[0] if wide else 'wide.csv',
    '--img_root', os.path.dirname(images[0]) if images else '.',
    '--output_dir', '/kaggle/working/models',
    '--log_dir', '/kaggle/working/logs',
    # '--full_train',  # one all-data model for submission instead of 5-fold CV
])
run_training(args)"""))
write('training.ipynb', train_cells)

# ---------------- inference notebook ----------------
i = sections('inference.py')
inf_header = re.sub(r'\nfrom train import \([^)]*\)\n', '\n', i['header'])
predict_only = t['Train / Predict Loops'].split('\n\ndef train_one_epoch')[0]
inf_cells = [cell('markdown', """# CSIRO Image2Biomass - Inference (Dual-Stream DINOv3)
Generated from `train.py` + `inference.py` - edit the scripts, not this notebook.

Attach the trained `.pt` checkpoints as a dataset and select them with `--models` (paths or glob patterns);
matching checkpoints are averaged, and each stores its own backbone and image size. Runs offline (`pretrained=False`)."""),
             cell('code', TIMM_CHECK),
             cell('code', t['header']),
             cell('code', t['Constants']),
             cell('code', t['Augmentations (applied independently to each sub-image view)']),
             cell('code', t['Model: Dual-Stream DINOv3 with Cross-View Attention']),
             cell('code', t['Metric & Post-Processing']),
             cell('code', predict_only),
             cell('code', inf_header + '\n\n' + '\n\n'.join(v for k, v in i.items() if k != 'header')),
             cell('code', """# Kaggle run configuration
matches = glob.glob('/kaggle/input/**/test.csv', recursive=True)
DATA_DIR = os.path.dirname(matches[0]) if matches else '.'
args = parse_args([
    # Checkpoints to average: one or more paths / glob patterns (** searches all attached datasets)
    '--models', '/kaggle/input/**/*.pt',
    '--test_csv', os.path.join(DATA_DIR, 'test.csv'),
    '--img_root', DATA_DIR,
    '--output_csv', '/kaggle/working/submission.csv',
    # '--postprocess', 'none',  # default 'first_place': lowers OOF but adds ~+0.01 on the private LB
])
run_inference(args)""")]
write('inference.ipynb', inf_cells)
print('train cells', len(train_cells), '| inference cells', len(inf_cells), '| train sections', list(t))
