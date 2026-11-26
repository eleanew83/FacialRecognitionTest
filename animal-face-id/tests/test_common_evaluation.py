"""Shared-cohort comparison must preserve pixels, labels, folds and pairing."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tools'))
import compare_reviewed_results as C
from src.datasets.evaluation_subset import select_evaluation_subset
from src.datasets.macaque_faces import MacaqueFacesDataset
from src.datasets.split_integrity import file_fingerprint


def row(identity, digest, path):
    return {'id': identity, 'sha256': digest, 'pixels_sha256': digest+'pixels', 'path': path,
            'source_path': path}


class CommonCohortTests(unittest.TestCase):
    def test_frozen_evaluation_gets_distinct_output_tag_and_records_cohort(self):
        import numpy as np
        from src.training.evaluate import run_final_evaluation
        cohort={'name':'shared_test','sha256':'frozen-hash','path':'benchmark.json'}
        loader=SimpleNamespace(dataset=SimpleNamespace(classes=['A','B'],evaluation_benchmark=cohort))
        head=SimpleNamespace(output_class_names=['A','B'])
        output={'labels':np.array([0,1]),'logits':np.array([[.9,.1],[.1,.9]]),
                'embeddings':np.zeros((2,2)),'paths':['a','b'],'view_probabilities':None}
        with tempfile.TemporaryDirectory() as folder, \
             patch('src.training.evaluate.build_dataloader',return_value=loader), \
             patch('src.training.evaluate._load_checkpoint',return_value=(None,head)), \
             patch('src.training.evaluate._collect_outputs',return_value=output):
            result=run_final_evaluation({'data':{'dataset_name':'animal_faces'},'model':{'head':'linear'}},
                ckpt='previous.pt',device='cpu',tag='reference',out_dir=folder,
                write_report=False,make_plots=False)
            self.assertEqual(result['evaluation_benchmark'],cohort)
            self.assertEqual(result['evaluation_tag'],'reference_shared_test')
            self.assertTrue(list(Path(folder).glob('*shared_test_metrics_summary.json')))

    def test_moved_changed_and_training_camera_photos_are_excluded(self):
        a = row('A','a','test/A/frame1_crop0.jpg')
        b = row('A','b','test/A/frame2_crop0.jpg')
        c = row('B','c','test/B/shared_crop0.jpg')
        moved = row('B','moved','test/B/moved.jpg')
        before = {'train':[row('A','train','train/A/shared_crop0.jpg')], 'val':[], 'test':[a,b,c,moved]}
        after = {'train':before['train']+[moved], 'val':[], 'test':[a,{**b,'sha256':'recropped'},c]}
        common, excluded = C.common_cohort(before, after, 'test')
        self.assertEqual([x['sha256'] for x,_ in common], ['a'])
        self.assertEqual(len(excluded), 2)

    def test_predictions_pair_by_exact_sample_even_when_csv_order_changes(self):
        samples = [{'id': 'A', 'sha256':'a', 'source_group':'g1',
                    'current_path':'new/a.jpg', 'historical_path':'old/a.jpg'},
                   {'id': 'B', 'sha256':'b', 'source_group':'g2',
                    'current_path':'new/b.jpg', 'historical_path':'old/b.jpg'}]
        rows = [{'image_path':'new/b.jpg','true_id':'B','pred_id':'B'},
                {'image_path':'new/a.jpg','true_id':'A','pred_id':'B'}]
        aligned = C.align_predictions(rows, samples, ['A','B'], verify_files=False)
        self.assertEqual([r['sha256'] for r in aligned], ['a','b'])
        with self.assertRaisesRegex(ValueError, 'complete frozen'):
            C.align_predictions(rows[:1], samples, ['A','B'], verify_files=False)
        with self.assertRaisesRegex(ValueError, 'labels'):
            C.align_predictions([{**r,'true_id':'A'} for r in rows], samples, ['A','B'], verify_files=False)

    def test_modified_image_bytes_block_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'image.jpg';path.write_bytes(b'changed')
            sample={'id':'A','sha256':hashlib.sha256(b'original').hexdigest(), 'source_group':'g',
                    'current_path':str(path),'historical_path':'old.jpg'}
            with self.assertRaisesRegex(ValueError, 'image changed'):
                C.align_predictions([{'image_path':str(path),'true_id':'A','pred_id':'A'}], [sample], ['A'])

    def test_paired_bootstrap_uses_camera_groups_and_reports_known_delta(self):
        old = [{'sha256':'a','source_group':'g','true_id':'A','pred_id':'B'},
               {'sha256':'b','source_group':'g','true_id':'B','pred_id':'B'}]
        new = [{**old[0],'pred_id':'A'},old[1]]
        result=C.paired_accuracy(old,new,repeats=50)
        self.assertEqual(result['accuracy_delta'],.5)
        self.assertEqual(result['accuracy_delta_ci95'],[.5,.5])


class FrozenDatasetTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup)
        self.root=Path(temp.name);self.payload={}
        number=0
        for split in ['train','val','test']:
            rows=[]
            for identity in ['A','B']:
                for frame in range(2):
                    number+=1
                    p=self.root/split/identity/(str(frame)+'.jpg');p.parent.mkdir(parents=True,exist_ok=True)
                    Image.new('RGB',(8,8),(number*10,20,30)).save(p)
                    rows.append({'id':identity,'path':str(p.relative_to(self.root)),**file_fingerprint(p)})
            self.payload[split]=rows
        self.manifest=self.root/'splits.json';self.manifest.write_text(json.dumps(self.payload))
        self.spec={'schema_version':1,'name':'frozen','class_names':['A','B'],
                   'splits':{s:[self.payload[s][0],self.payload[s][2]] for s in ['val','test']}}
        self.benchmark=self.root/'benchmark.json';self.benchmark.write_text(json.dumps(self.spec))
        self.digest=hashlib.sha256(self.benchmark.read_bytes()).hexdigest()

    def dataset(self, split):
        return MacaqueFacesDataset(self.root,self.manifest,split,
            evaluation_subset=self.benchmark,evaluation_subset_sha256=self.digest)

    def test_filter_changes_only_eval_samples_and_preserves_training_class_indices(self):
        train,val,test=self.dataset('train'),self.dataset('val'),self.dataset('test')
        self.assertEqual([len(train),len(val),len(test)],[4,2,2])
        self.assertEqual(train.class_to_idx,val.class_to_idx)
        self.assertEqual(train.class_to_idx,test.class_to_idx)
        self.assertIsNone(train.evaluation_benchmark)
        self.assertEqual(test.evaluation_benchmark['sha256'],self.digest)

    def test_mutating_frozen_benchmark_is_rejected(self):
        self.benchmark.write_text(self.benchmark.read_text()+' ')
        with self.assertRaisesRegex(ValueError,'benchmark changed'):
            self.dataset('test')

    def test_changed_labels_pixels_missing_classes_and_duplicates_are_rejected(self):
        for bad in [self.payload['train'][:2],
                    [self.payload['test'][0],self.payload['test'][0]],
                    [{**self.payload['test'][0],'pixels_sha256':'changed'},self.payload['test'][2]]]:
            spec={**self.spec,'splits':{'test':bad}}
            self.benchmark.write_text(json.dumps(spec))
            with self.assertRaises(ValueError):
                select_evaluation_subset(self.payload['test'],'test',self.benchmark,['A','B'])


if __name__=='__main__':
    unittest.main()
