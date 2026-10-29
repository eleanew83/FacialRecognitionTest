from pathlib import Path
import hashlib
import json
import random
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
from torchvision.transforms import functional as TF

ROOT=Path('/rds/user/ylj20/hpc-work/FacialRecognitionTest/animal-face-id')
sys.path.insert(0,str(ROOT))
from src.datasets.transforms import build_transforms
from src.config.base import load_config

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
OUT=ROOT/'reports/figures/augmentation_comparison_2026-10-28/meeting_examples'
OUT.mkdir(parents=True,exist_ok=True)
OLD=ROOT/'artifacts/augmentation_sweeps/20261025T023752Z_jyY1BT'
NEW=ROOT/'artifacts/augmentation_sweeps/20261028T164412Z_JmVXbV'
configs={a:((OLD if a in (30,100,150) else NEW)/f'configs/train_macaque_arcface_aggressive_rot{a}.yaml') for a in (30,45,60,100,150)}
data={a:load_config(str(p)).data for a,p in configs.items()}
config=data[45]
raw=Path(config['raw_root'])
manifest=Path(config['splits_path'])
train=json.loads(manifest.read_text())['train']
chosen=[[r for r in train if r['id']==name][index] for name,index in (('Abby',0),('Harry',3),('Warty',0))]
mean=torch.tensor(config['mean']).view(3,1,1)
std=torch.tensor(config['std']).view(3,1,1)
def display(tensor):
    return TF.to_pil_image((tensor.detach().cpu()*std+mean).clamp(0,1))
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
plt.rcParams.update({'font.family':'Carlito','font.size':11})
fig,axes=plt.subplots(3,6,figsize=(12,7.2),layout='constrained')
metadata={'created':'2026-10-29','purpose':'Meeting illustrations, regenerated from actual training transforms; not cached training batches or new evaluation results.',
          'display':'Originals are existing detection crops, resized to 224×224 without augmentation. Augmented tensors are de-normalized and clipped to [0,1] for display, including random erasing.',
          'selection':'Training-manifest entries 0 for Abby, 3 for Harry, and 0 for Warty (zero-based per identity). Harry was chosen from a visual contact sheet to show a visible face rather than an ear-only crop; no model-performance selection.',
          'torch_version':torch.__version__,'torchvision_version':torchvision.__version__,
          'split_manifest_sha256':sha(manifest),'transform_source_sha256':sha(ROOT/'src/datasets/transforms.py'),
          'configs':{str(a):{'path':str(p.relative_to(ROOT)),'sha256':sha(p),'data':data[a]} for a,p in configs.items()},'examples':[]}
originals=[]
for i,row in enumerate(chosen):
    source=raw/row['path']
    assert sha(source)==row['sha256'],source
    with Image.open(source) as im:
        image=im.convert('RGB')
    originals.append(image)
    image.save(OUT/f'{row["id"]}_original_crop.png')
    clean=T.Resize((224,224))(image)
    clean.save(OUT/f'{row["id"]}_original_display.png')
    axes[i,0].imshow(clean)
    axes[i,0].set_title(f'{row["id"]}\nOriginal crop',fontsize=11)
    record={'identity':row['id'],'split':'train','path':str(source),'sha256':sha(source),'original_size':list(image.size),'seed':42+i,'augmentations':[]}
    for j,a in enumerate((30,45,60,100,150),1):
        pipeline=build_transforms('train',data[a])
        seed=42+i
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        # Record the parameters sampled by the actual affine transform.
        affine=next(op for op in pipeline.transforms if isinstance(op,T.RandomAffine))
        original_get_params=affine.get_params
        captured=[]
        def capture(*args,**kwargs):
            params=original_get_params(*args,**kwargs)
            captured.append(params)
            return params
        affine.get_params=capture
        augmented=display(pipeline(image.copy()))
        angle,translate,scale,shear=captured[0]
        augmented.save(OUT/f'{row["id"]}_training_rot{a}.png')
        axes[i,j].imshow(augmented)
        axes[i,j].set_title(f'±{a}° training\nSample: {angle:+.1f}°',fontsize=11)
        record['augmentations'].append({'policy_degrees':a,'sampled_angle':angle,'translation_pixels':list(translate),'scale':scale,'shear':list(shear),'output':f'{row["id"]}_training_rot{a}.png'})
    for ax in axes[i]:
        ax.set_xticks([]);ax.set_yticks([])
        for spine in ax.spines.values():spine.set_visible(False)
    metadata['examples'].append(record)
fig.suptitle('Original training crops and examples from the actual augmentation pipelines',fontsize=15)
for ext in ('png','pdf'):
    fig.savefig(OUT/f'original_and_training_augmentations.{ext}',dpi=220)
plt.close(fig)

# Show the precise deterministic inference views, separately from random training augmentation.
angles=[-30,-20,-10,0,10,20,30]
tta_config={**config,'tta_angles':angles}
views=build_transforms('test',tta_config)(originals[0])
fig,axes=plt.subplots(2,4,figsize=(9,5.4),layout='constrained')
panels=[T.Resize((224,224))(originals[0])]+[display(t) for t in views]
titles=['Original crop\n(reference only)']+[f'TTA view: {a:+d}°' for a in angles]
for ax,im,title in zip(axes.flat,panels,titles):
    ax.imshow(im);ax.set_title(title,fontsize=11);ax.set_xticks([]);ax.set_yticks([])
    for spine in ax.spines.values():spine.set_visible(False)
for a,im in zip(angles,panels[1:]):
    im.save(OUT/f'Abby_tta_{a:+d}.png')
fig.suptitle('Seven-angle probability averaging: inference views of the same crop',fontsize=14)
for ext in ('png','pdf'):
    fig.savefig(OUT/f'seven_angle_inference_views.{ext}',dpi=220)
plt.close(fig)
metadata['tta']={'source_identity':'Abby','source_split':'train, used only to illustrate the inference transform','angles':angles,'view_shape':list(views.shape),'reference_is_extra_view':False,'explanation':'Original crop is a display reference. The seven inference views include 0° exactly once. Class probabilities, not image pixels or model weights, are averaged.'}
metadata['output_sha256']={p.name:sha(p) for p in sorted(OUT.glob('*.png'))}
(OUT/'examples_manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
print('Saved training and TTA example sheets, PDFs, individual images and provenance to',OUT,flush=True)
print('Identities:',[r['identity'] for r in metadata['examples']],flush=True)
