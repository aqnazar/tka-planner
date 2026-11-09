# Online landmark study

Surgeons place the study landmarks in their own browser. They work on the 10 study knees
and one group of 20 knees from the VSD dataset, in two rounds: *No markers* (place every
landmark) and then *Automatic markers* (move the planner's wrong ones). The planner does not
run online. Each knee is computed on the study lead's machine by the planner's own
`pick_targets`, encrypted under the study passphrase, and decrypted in the browser. The
page reuses `web/viewer.js` and `web/study.js` unchanged.

Saves go to Netlify Blobs exactly as the browser sent them. `online/pull.py` writes them
home through `tka_planner.study.save_picks`, so online and local picks are the same files.

Research use only. Not a medical device.

## Steps

```
python -m online.vsd_cases "<Datasets>/VSD_Fischer/repo" "<Surgeon Study>/vsd_cases"
python -m online.build_site                  # all knees; --only P009L:left VSD002:left for a trial
cd "<Surgeon Study>/online_site/dist"
npm install                                  # once: @netlify/blobs for the functions
npx netlify-cli login                        # once per machine
npx netlify-cli sites:create --name knee-study-<random> --account-slug <team>   # once; links dist/
npx netlify-cli deploy --prod --dir public --functions netlify/functions
python -m online.pull --site https://<site>.netlify.app
```

`<Surgeon Study>/online_site/` holds:

- `secrets.json`: the passphrase, the admin key and the salt.
- `groups.json`: which cadavers are in groups A, B and C.
- `site.json`: the site's address and Netlify project.
- `dist/`: the deployable site.

None of it belongs in the repository.

A change to the page alone (`site/public/*.js|html|css`) needs no rebuild. Copy the file
into `dist/public/` and deploy again; the knee files are unchanged.

To check the live site, sign in under a test name, then delete those saves with
`npx netlify-cli blobs:delete landmark-study <key> --force` (run inside `dist/`). The key
prefix is `picks/<first 16 hex of sha256(the normalised name)>/`.

## What is where

- `vsd_cases.py`: VSD whole limbs → knee-only cases.
  - The femur keeps 175 mm and the tibia 175 mm; the fibular head is joined to the tibia.
  - Edges are subdivided to 0.7 mm so the plateau search has enough points.
  - Cadavers lying turned more than 30° are turned square about scanner Z, with the patella in front. The turn is recorded in `orientation.json`.
- `build_site.py`: the encrypted knee files and the manifest (AES-256-GCM, PBKDF2-SHA256 310 000). Web bones are simplified within 0.05 mm.
- `site/public/`: the page (`online.js` connects the planner's viewer and Pick code to the knee files).
- `site/netlify/functions/`: `save` (passphrase token), `progress` (a surgeon's own saves) and `list` (admin key, for the pull).
- `pull.py`: saves → `save_picks` → `<Surgeon Study>/output/picks/`. The ledger is `online_pulled.json`.

VSD credit: Fischer M.C.M., Sci. Data 10, 763 (2023); CC BY-NC-SA 4.0.
