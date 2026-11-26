/* Embedded into the portable HTML. No fetch, server, CDN or installation. */
(() => {
  'use strict';
  const pack = JSON.parse(document.getElementById('offlineData').textContent);
  const clone = value => JSON.parse(JSON.stringify(value));
  const uuid = () => Array.from(crypto.getRandomValues(new Uint8Array(16)), b => b.toString(16).padStart(2, '0')).join('');
  const storageKey = 'macaque-offline:' + pack.manifest.package_id;
  const fresh = () => ({schema: pack.manifest.schema, dataset_id: pack.manifest.dataset_id,
    package_id: pack.manifest.package_id, session_id: uuid(), created_at: new Date().toISOString(), items: {}});
  let state = fresh(), revision = 0, preview = null, persistent = true;
  let previousExport = '', changedSinceExport = false;
  try { const saved = localStorage.getItem(storageKey); if (saved) state = validate(JSON.parse(saved)); }
  catch (error) { persistent = false; }
  window.CROP_OFFLINE_IMAGES = {};
  const assets = new Map();
  function validate(value) {
    if (value.schema !== pack.manifest.schema || value.package_id !== pack.manifest.package_id ||
        value.dataset_id !== pack.manifest.dataset_id || !/^[a-f0-9]{32}$/.test(value.session_id) ||
        !value.items || typeof value.items !== 'object' || Array.isArray(value.items))
      throw new Error('Choose a results JSON from this exact offline package.');
    const ids = new Set();
    for (const [key, item] of Object.entries(value.items)) {
      if (!pack.metadata[key] || !item || !Array.isArray(item.events) || item.events.length > 1000)
        throw new Error('Invalid offline image/history.');
      if (item.draft) clean(key, item.draft);
      for (const event of item.events) {
        if (!/^[a-f0-9]{32}$/.test(event.id) || ids.has(event.id)) throw new Error('Invalid offline operation ID.');
        ids.add(event.id); clean(key, event.draft);
      }
    }
    return value;
  }
  function save() {
    state.updated_at = new Date().toISOString(); revision++; changedSinceExport = true;
    try { localStorage.setItem(storageKey, JSON.stringify(state)); persistent = true; }
    catch (error) { persistent = false; }
    updateCounts();
  }
  function item(key) { return state.items[key] || (state.items[key] = {draft: null, events: []}); }
  function clean(key, draft) {
    if (!draft || !Object.hasOwn(pack.context.actions, draft.action) || !pack.context.identities.includes(draft.target_identity))
      throw new Error('Choose a valid operation and target identity.');
    const variant = pack.metadata[key].variants.find(v => v.id === draft.variant && v.available);
    if (!variant || variant.sha256 !== draft.source_sha256) throw new Error('Source image differs from the offline package.');
    if (typeof draft.notes !== 'string' || draft.notes.length > 5000) throw new Error('Correction note is too long.');
    const b = draft.box;
    if (b !== null && (!Array.isArray(b) || b.length !== 4 || !b.every(Number.isFinite) ||
        b[0] < 0 || b[1] < 0 || b[2] <= 0 || b[3] <= 0 || b[0] + b[2] > 1.000001 || b[1] + b[3] > 1.000001))
      throw new Error('Draw a rectangle inside the photograph.');
    return clone(draft);
  }
  async function asset(digest) {
    if (!digest) throw new Error('No existing face crop. Create a face crop first.');
    if (assets.has(digest)) return assets.get(digest);
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'assets/' + digest + '.js';
      script.onload = () => {
        const data = window.CROP_OFFLINE_IMAGES[digest];
        delete window.CROP_OFFLINE_IMAGES[digest]; script.remove();
        if (!data) { reject(new Error('Image asset is missing. Extract the whole ZIP first.')); return; }
        const bytes = Uint8Array.from(atob(data.split(',')[1]), c => c.charCodeAt(0));
        const url = URL.createObjectURL(new Blob([bytes], {type: 'image/jpeg'}));
        resolve(url);
      };
      script.onerror = () => { script.remove(); reject(new Error('Cannot open image. Extract the entire ZIP and keep assets beside index.html.')); };
      document.head.append(script);
    });
    assets.set(digest, promise);
    promise.catch(() => assets.delete(digest));
    // Bound image data in memory while keeping the current photos cached.
    if (assets.size > 10) {
      const oldest = assets.keys().next().value, old = assets.get(oldest);
      assets.delete(oldest); old.then(url => URL.revokeObjectURL(url)).catch(() => {});
    }
    return promise;
  }
  function currentFace(key) {
    const meta = pack.metadata[key];
    let face = meta.face_asset ? {asset: meta.face_asset, box: null,
      identity: meta.effects.current_face?.identity || meta.record.identity} : null;
    for (const event of state.items[key]?.events || []) {
      const d = event.draft;
      if (d.action === 'face_crop') face = {asset: meta.variants.find(v => v.id === d.variant).asset, box: d.box, identity: d.target_identity};
      else if (d.action.startsWith('relabel_') && face) face.identity = d.target_identity;
    }
    return face;
  }
  async function region(digest, box) {
    const url = await asset(digest);
    if (!box) return url;
    const image = new Image(); image.src = url; await image.decode();
    const x = Math.floor(box[0] * image.naturalWidth), y = Math.floor(box[1] * image.naturalHeight);
    const right = Math.min(image.naturalWidth, Math.ceil((box[0] + box[2]) * image.naturalWidth));
    const bottom = Math.min(image.naturalHeight, Math.ceil((box[1] + box[3]) * image.naturalHeight));
    if (Math.min(right - x, bottom - y) < 2) throw new Error('The crop is too small.');
    const canvas = document.createElement('canvas'); canvas.width = right - x; canvas.height = bottom - y;
    canvas.getContext('2d').drawImage(image, x, y, canvas.width, canvas.height, 0, 0, canvas.width, canvas.height);
    // Browser preview only; final pixels are generated from the original on CSD3.
    return canvas.toDataURL('image/jpeg', 0.94);
  }
  function updateCounts() {
    const entries = Object.values(state.items), confirmed = entries.reduce((n, i) => n + i.events.length, 0);
    document.getElementById('offlineCount').textContent = `${confirmed} confirmed operation(s) on ${entries.filter(i => i.events.length).length} image(s). ` +
      (changedSinceExport ? 'Download results to back up this session.' : previousExport ? 'Results downloaded at ' + previousExport + '.' : 'Download results before closing.') +
      (persistent ? '' : ' Browser storage unavailable — download frequently.');
  }
  async function api(path, payload) {
    const route = path.split('/').pop();
    if (path === '/api/corrections/context') {
      const context = clone(pack.context); context.revision = revision;
      for (const row of context.records) {
        const i = state.items[row.key]; row.offline_saved = !!i?.events.length;
        row.applied = row.offline_saved || !!pack.metadata[row.key].history.length;
        row.has_draft = !!i?.draft;
      }
      return context;
    }
    if (path.startsWith('/api/corrections/record/')) {
      const key = route, meta = clone(pack.metadata[key]), i = state.items[key];
      if (i?.draft) meta.draft = clone(i.draft);
      // Baseline online drafts need not reference a source included in this package.
      if (meta.draft && !meta.variants.some(v => v.id === meta.draft.variant)) meta.draft = null;
      meta.baseline_applied = !!meta.history.length;
      meta.history = (i?.events || []).map(e => e.id); meta.draft_applied = false;
      meta.revision = revision; meta.effects.current_face = currentFace(key);
      if (meta.effects.current_face) meta.effects.faces = {[meta.effects.current_face.identity]: meta.effects.current_face};
      meta.effects.face_pending = false;
      for (const event of i?.events || []) {
        if (['edit_source','move_source','copy_source'].includes(event.draft.action)) meta.effects.face_pending = true;
        if (event.draft.action === 'face_crop' || event.draft.action.startsWith('relabel_')) meta.effects.face_pending = false;
      }
      return meta;
    }
    if (!payload || payload.dataset_id !== pack.manifest.dataset_id) throw new Error('Dataset mismatch.');
    if (route === 'draft') {
      item(payload.key).draft = clean(payload.key, payload.draft); save(); return {revision};
    }
    if (route === 'preview') {
      const key = payload.key, d = clean(key, payload.draft), source = pack.metadata[key].variants.find(v => v.id === d.variant);
      if (['face_crop','edit_source'].includes(d.action) && !d.box) throw new Error('Draw a crop rectangle first.');
      const face = currentFace(key), relabel = d.action.startsWith('relabel_');
      if (relabel && !face) throw new Error('Create a face crop before relabelling this original-only photo.');
      const sourceAction = ['edit_source','move_source','copy_source','relabel_move','relabel_copy'].includes(d.action);
      const warnings = ['This saves instructions locally. Actual file changes and backups happen only after import on CSD3.'];
      if (sourceAction) warnings.push('The photograph shown in this package stays unchanged. Later boxes still use this export-time original.');
      if (['edit_source','move_source','copy_source'].includes(d.action)) warnings.push('Also create or verify a face crop before building the training dataset.');
      const faceUrl = d.action === 'face_crop' ? await region(source.asset, d.box) : relabel ? await region(face.asset, face.box) : null;
      const sourceUrl = sourceAction ? await region(source.asset, d.box) : null;
      preview = {id: uuid(), key, draft: d};
      return {id: preview.id, face_image_url: faceUrl, source_image_url: sourceUrl,
        action_label: pack.context.actions[d.action], target_identity: d.target_identity,
        operations: faceUrl || sourceUrl ? [{description: 'Queue ' + pack.context.actions[d.action] + ' for ' + d.target_identity,
          path: 'Export-time source: ' + source.path}] : [], warnings,
        backup_directory: 'Created on CSD3 when the downloaded results are imported with --apply'};
    }
    if (route === 'apply') {
      if (!preview || preview.id !== payload.token) throw new Error('Preview this correction again first.');
      const i = item(preview.key);
      i.events.push({id: uuid(), draft: clone(preview.draft), saved_at: new Date().toISOString()});
      i.draft = clone(preview.draft); preview = null; save(); return {revision};
    }
    if (route === 'undo') {
      const i = item(payload.key);
      if (!i.events.length) throw new Error('No offline correction to undo.');
      i.events.pop(); i.draft = i.events.length ? clone(i.events.at(-1).draft) : null;
      save(); return {revision};
    }
    throw new Error('Unavailable offline operation.');
  }
  async function resolveImage(path) {
    const url = new URL(path, 'https://offline.invalid');
    const key = url.pathname.split('/').pop(), variant = url.searchParams.get('variant');
    if (variant === 'face') { const face = currentFace(key); if (!face) throw new Error('No face crop yet.'); return region(face.asset, face.box); }
    return asset(pack.metadata[key].variants.find(v => v.id === variant).asset);
  }
  function download() {
    validate(state);
    const url = URL.createObjectURL(new Blob([JSON.stringify(state, null, 2) + '\n'], {type: 'application/json'}));
    const link = document.createElement('a'); link.href = url;
    link.download = 'macaque_results_' + state.package_id.slice(0, 8) + '_' + new Date().toISOString().replace(/[:.]/g, '-') + '.json';
    document.body.append(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url), 5000);
    previousExport = new Date().toLocaleTimeString(); changedSinceExport = false; updateCounts();
  }
  function setup() {
    document.querySelector('h1').textContent = 'Macaque offline correction editor';
    document.querySelector('h1 + p').textContent = 'Work without CSD3 or internet. Draw on the full-size original, Preview, then Save offline correction. Download results after each session; load that JSON to resume. Photos in this package stay unchanged until you import the instructions on CSD3.';
    document.querySelector('a[href="/"]').hidden = true;
    document.getElementById('export').textContent = 'Download results';
    document.getElementById('apply').textContent = 'Save offline correction';
    document.getElementById('undoApplied').textContent = 'Undo last offline correction';
    document.getElementById('save').textContent = 'Save draft locally';
    const option = document.createElement('option'); option.value = 'unfinished'; option.textContent = 'Unfinished flagged images';
    document.getElementById('scope').prepend(option); document.getElementById('scope').value = 'unfinished';
    document.querySelector('#scope option[value="applied"]').textContent = 'Saved corrections';
    const restore = document.createElement('button'); restore.textContent = 'Load results JSON';
    const input = document.createElement('input'); input.type = 'file'; input.accept = '.json,application/json'; input.hidden = true;
    document.getElementById('export').after(restore, input);
    restore.onclick = async () => {
      try { await window.CROP_OFFLINE.flush?.(); input.click(); }
      catch (error) { alert(error.message); }
    };
    input.onchange = async () => {
      try {
        const restored = validate(JSON.parse(await input.files[0].text()));
        if (Object.keys(state.items).length && !confirm('Replace browser progress with this downloaded results file? Download current results first if you need to keep them.')) return;
        state = restored; save();
        for (const key of Object.keys(pack.metadata)) {
          try { localStorage.removeItem('macaque-correction-draft:' + pack.manifest.package_id + ':' + key); } catch (error) {}
        }
        await window.CROP_OFFLINE.refresh?.();
      } catch (error) { alert(error.message); }
    };
    const count = document.createElement('p'); count.id = 'offlineCount'; count.className = 'status warn';
    document.getElementById('status').before(count); updateCounts();
    // Expose serializable instructions for the shared UI and browser testing.
    window.CROP_OFFLINE = {api, resolveImage, download, package_id: pack.manifest.package_id, results: () => clone(state),
      ready: 'Ready offline. Preview and Save offline correction, then Download results before closing.',
      saved: 'Saved on this laptop. Download results to back it up; CSD3 photos have not changed.'};
  }
  setup();
})();
