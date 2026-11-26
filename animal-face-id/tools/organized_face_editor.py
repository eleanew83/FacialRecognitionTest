"""Stage 2: derive face crops from completed, versioned original-photo batches."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

spec = importlib.util.spec_from_file_location('organized_face_transactions', Path(__file__).with_name('crop_corrections.py'))
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)

ACTIONS = {'face_crop': 'Draw / adjust face crop',
           'exclude': 'No usable face — exclude face output', 'defer': 'Defer face / identity check'}


class OrganizedFaceStore(C.CorrectionStore):
    def __init__(self, review, originals):
        self.base_review = review
        self.originals = originals
        stage = originals.root / 'face_stage'
        stage.mkdir(exist_ok=True)
        proxy = SimpleNamespace(output=stage, inventory=review.inventory, records={}, state={'annotations': {}})
        self.source_members = {}
        super().__init__(proxy)
        self.refresh_sources()

    def refresh_sources(self):
        """Read live source-stage state; never amend completed face/review work."""
        source_state = self.originals.read()
        self.originals.refresh_catalog()
        records, members, annotations = {}, {}, {}
        by_path = {}
        for r in self.base_review.records.values():
            by_path.setdefault(r['source_photo'], []).append(r)
        for group, original in self.originals.catalog['records'].items():
            history = source_state['entries'].get(group, {}).get('history', [])
            if not history:
                continue
            txn = json.loads((self.originals.root / 'transactions' / (history[-1] + '.json')).read_text())
            effect = txn.get('effect', {})
            if effect.get('decision') != 'edit':
                continue
            before = {v['path']: v for v in original['active_files']}
            final = {}
            for output in effect.get('outputs', []):
                old = before.get(output['path'])
                if old and old['identity'] == output['identity'] and old['sha256'] == output['sha256']:
                    continue
                final[output['path']] = {**output, 'retained': False}
            splits = {r['split'] for v in original['active_files'] for r in by_path.get(v['path'], [])}
            split = next(iter(splits)) if len(splits) == 1 else 'unassigned'
            for path, member in sorted(final.items()):
                if member['identity'] not in self.identities:
                    continue
                C.confined(self.source_root, path)
                key = C.sha(json.dumps([group, history[-1], path, member['sha256']]).encode())[:24]
                member = {**member, 'parent_group': group, 'parent_transaction': history[-1],
                          'source_decision': effect['decision'], 'parent_filename': original['filename'],
                          'split': split, 'key': key}
                members[key] = member
                records[key] = {'key': key, 'identity': member['identity'], 'split': split, 'crop_path': '',
                                'crop_absolute': None, 'source_photo': path, 'source_path': Path(path).name,
                                'record_type': 'organised_original'}
                annotations[key] = {'status': 'organised original', 'suggested_identity': member['identity'],
                                    'notes': 'Assigned identity: ' + member['identity'] + '\nOrganised photograph: ' + original['filename'] +
                                    '\nDraw only this individual’s face. The whole-monkey original remains intact.'}
        self.source_members = members
        self.review.records = records
        self.review.state = {'annotations': annotations}

    def require_source(self, key):
        self.record(key)
        member = self.source_members[key]
        if C.file_sha(member['path']) != member['sha256']:
            raise C.Conflict('The organised original changed after Stage 1. Recheck its original-photo batch before saving a face.')
        return member

    def variants(self, key, state=None):
        return {'master': ('Organised original — ' + self.record(key)['identity'],
                           C.confined(self.source_root, self.record(key)['source_photo']))}

    def validate_draft(self, key, draft):
        cleaned = super().validate_draft(key, draft)
        if cleaned['action'] not in ACTIONS or cleaned['target_identity'] != self.record(key)['identity'] or cleaned['keep_existing_face']:
            raise ValueError('This stage saves one face under its organised identity. Fix the original-photo assignment in Stage 1 if necessary.')
        if cleaned['source_sha256'] != self.source_members[key]['sha256']:
            raise C.Conflict('Face rectangle belongs to an older original. Recheck Stage 1.')
        return cleaned

    def context(self):
        with self.originals.guard():
            self.refresh_sources()
            data = super().context()
            state = self.read()
            for row in data['records']:
                effect = self.effects(row['key'], state)
                row['decision'] = effect['decision']
                row['applied'] = bool(state['entries'].get(row['key'], {}).get('history')) and effect['decision'] != 'defer'
                row['source_group'] = self.source_members[row['key']]['parent_group']
            data['actions'] = ACTIONS
            data['stage'] = 'organised_faces'
            data['ready_originals'] = len(self.source_members)
            data['inactive_saved_entries'] = len(set(state['entries']) - set(self.source_members))
            return data

    def metadata(self, key):
        with self.originals.guard():
            self.refresh_sources()
            self.require_source(key)
            meta = super().metadata(key)
            meta['organised_source'] = self.source_members[key]
            return meta

    def save_draft(self, payload):
        with self.originals.guard():
            self.refresh_sources()
            self.require_source(payload.get('key'))
            return super().save_draft(payload)

    def preview(self, payload):
        with self.originals.guard():
            self.refresh_sources()
            member = self.require_source(payload.get('key'))
            plan = super().preview(payload)
            for name in ('image_url', 'face_image_url', 'source_image_url'):
                if plan.get(name):
                    plan[name] = plan[name].replace('/api/corrections/', '/api/organized-faces/')
            if plan['effect'].get('face'):
                plan['effect']['face'].update(source_group=member['parent_group'], parent_transaction=member['parent_transaction'],
                                              organised_original=member['path'], organised_original_sha256=member['sha256'])
            plan['organised_source'] = member
            plan['action_label'] = ACTIONS[plan['draft']['action']]
            C.atomic_json(self.root / 'previews' / plan['id'] / 'plan.json', plan)
            return plan

    def apply(self, payload):
        with self.originals.guard():
            self.refresh_sources()
            token = payload.get('token', '')
            if not isinstance(token, str) or len(token) != 32 or any(c not in '0123456789abcdef' for c in token):
                raise ValueError('Invalid face preview.')
            plan = json.loads((self.root / 'previews' / token / 'plan.json').read_text())
            member = self.require_source(plan['key'])
            if member != plan.get('organised_source'):
                raise C.Conflict('Original-photo organisation changed after face preview. Reload before saving.')
            return super().apply(payload)

    def undo(self, payload):
        with self.originals.guard():
            self.refresh_sources()
            return super().undo(payload)

    def image(self, key, variant):
        with self.originals.guard():
            self.refresh_sources()
            self.require_source(key)
            return super().image(key, variant)

    def export(self):
        with self.originals.guard():
            self.refresh_sources()
            state = self.read()
            approved, decisions = [], []
            for key, member in self.source_members.items():
                effect = self.effects(key, state)
                source_ok = C.file_sha(member['path']) == member['sha256']
                decision = effect['decision'] or 'pending'
                if not source_ok:
                    decision = 'source_changed'
                face = effect.get('current_face')
                if source_ok and decision == 'corrected' and face:
                    if C.file_sha(face['path']) == face['sha256']:
                        approved.append({**face, 'identity': member['identity'], 'split_hint': member['split'],
                                         'parent_group': member['parent_group'], 'parent_transaction': member['parent_transaction']})
                    else:
                        decision = 'face_output_changed'
                decisions.append({'key': key, 'identity': member['identity'], 'original_path': member['path'], 'decision': decision})
            return {'schema_version': 1, 'dataset_id': state['dataset_id'], 'created_at': C.now(),
                    'approved_faces': approved, 'decisions': decisions,
                    'inactive_saved_keys': sorted(set(state['entries']) - set(self.source_members)),
                    'note': 'Face outputs for completed original-photo batches. Merge with the earlier approved corrections and exclusions during dataset assembly; split hints are not a final training split.'}

    def handle_get(self, handler, route, query):
        if not C.own_user_connection(handler):
            return handler.send_json({'error': 'Use forwarding from your own CSD3 account.'}, 403)
        if route == '/organized-faces':
            return handler.send(editor_html().encode(), 'text/html; charset=utf-8')
        if route == '/api/organized-faces/export':
            return handler.send(json.dumps(self.export(), indent=2).encode(), 'application/json', filename='organised_face_manifest.json')
        return super().handle_get(handler, route.replace('/api/organized-faces/', '/api/corrections/'), query)

    def handle_post(self, handler, route, payload):
        return super().handle_post(handler, route.replace('/api/organized-faces/', '/api/corrections/'), payload)


def editor_html():
    html = Path(__file__).with_name('crop_corrections.html').read_text()
    html = html.replace('/api/corrections/', '/api/organized-faces/')
    html = html.replace('Macaque correction editor', 'Stage 2 · Face crops from organised originals')
    html = html.replace('Draw a face crop, correct an identity, or optionally edit/move/copy an original-folder photo. Drafts preserve your work. Preview lists the exact file changes before Apply. Your completed review is stored separately.',
                        'Only new or replaced images from completed original-photo edits appear here. No-changes-needed photos and untouched copies keep their earlier face crops. Check the assigned identity, draw its face rectangle, Preview, then Save. The organised original stays intact. Use Stage 1 to correct a folder assignment. Reload to collect newly finished originals.')
    html = html.replace('<a href="/">Back to review grid</a>', '<a href="/originals">Stage 1: organise originals</a><a href="/corrections" target="_blank">Earlier face corrections</a>')
    html = html.replace('<option value="attention">Flagged in completed review</option>', '<option value="unfinished">Unfinished face crops</option><option value="attention">All organised originals</option><option value="deferred">Deferred faces</option>')
    html = html.replace('<option value="uncertain">Uncertain filename candidates</option>', '')
    html = html.replace('scope===\'all\'||scope===\'unfinished\'', "scope==='deferred'&&r.decision==='defer'||scope==='all'||scope==='unfinished'")
    html = html.replace("row.applied=!!meta.history.length||!!meta.baseline_applied;", "row.applied=!!meta.history.length&&meta.effects.decision!=='defer';row.decision=meta.effects.decision;")
    html = html.replace("context.records.find(r=>r.key===key).applied=meta.history.length>0||!!meta.baseline_applied;", "context.records.find(r=>r.key===key).applied=meta.history.length>0&&meta.effects.decision!=='defer';")
    html = html.replace("context=fresh;await selectRecord(key||rows[0]?.key);", "context=fresh;filter();await selectRecord(context.records.some(r=>r.key===key)?key:rows[0]?.key);")
    html = html.replace("$('target').value=saved?.target_identity||", "$('target').value=meta.record.identity||saved?.target_identity||")
    html = html.replace("'macaque-correction-draft:'", "'macaque-organised-face-draft:'")
    html = html.replace('<style>', '<style>#target{pointer-events:none;background:#e9f1f6}#keepExistingLabel{display:none!important}')
    html = html.replace('Confirmed target identity', 'Assigned identity from Stage 1')
    html = html.replace("function updateActionHelp(){", "function updateActionHelp(){$('target').disabled=true;")
    html = html.replace("$('apply').disabled=!previewToken;", "$('target').disabled=true;$('apply').disabled=!previewToken;")
    html = html.replace('Apply this correction', 'Save face crop / decision')
    html = html.replace('Correction applied with backup and undo record. Completed review decisions are unchanged.', 'Face crop / decision saved. The organised original and earlier face corrections are preserved.')
    html = html.replace('Ready. Draw or choose an operation; drafts autosave. File changes require Preview and Apply.', 'Ready for Stage 2. Draw this individual’s face, Preview, then Save. If the queue is empty, finish a Stage 1 batch first. Drafts autosave.')
    html = html.replace('Preview ready. Check the output images and file paths, then Apply if correct.', 'Preview ready. Check the face and its assigned identity, then Save face crop / decision.')
    html = html.replace('Download correction backup', 'Export approved faces + decisions')
    html = html.replace('No face crop exists yet. Draw one from the source photo.', 'Draw a face box on this organised original. Its identity is shown above.')
    html = html.replace('Reload saved corrections', 'Reload newly organised originals')
    html = html.replace('Math.min(1,($(\'canvasScroll\').clientWidth-18)/canvas.width)', "Math.min(1,($('canvasScroll').clientWidth-18)/canvas.width,Math.max(180,window.innerHeight*.62)/canvas.height)")
    return html
