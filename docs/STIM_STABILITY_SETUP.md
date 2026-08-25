# Weekly stimulus-stability report — setup

A Sunday digest that, per stimulated channel on the rig, renders average-stim
waveform PNGs (per file / day / week), the access-resistance (Rₐ) drift figure,
and a stim-vs-evoked (1–50 ms) correlation scatter; uploads them to Google Drive;
keeps a running **"Stim Stability"** Sheet (one row per week per channel with
`=IMAGE` thumbnails + a folder link); and emails the headline figures. Every
figure is also attached to the email, so it works even if Drive is off.

Ships **disabled**. Order: (0) dry-run → (1) Drive auth → (2) config → (3) enable.

---

## 0. Dry-run first (no Drive / Sheet / email — safe now)

```
python -m src.notifications.stim_stability_weekly --dry-run
```

PNGs land in `data/stim_stability_preview/<animal>/<channel>/`. Eyeball them.

---

## 1. Drive auth — pick ONE

Figures upload to Google Drive. A **service account can't own files on a personal
(non-Workspace) Drive** (no storage quota), so choose based on what you have.

### Option A — Personal Gmail Drive (`auth_mode: oauth`)  ← default in config
Uploads happen **as you**, into your own 15 GB Drive.

1. **Create an OAuth client** in the GCP project `sheet-sync-473717`:
   - https://console.cloud.google.com/apis/credentials → **Create credentials →
     OAuth client ID → Application type: Desktop app**.
   - If prompted, configure the **OAuth consent screen** (User type: External),
     add yourself as a test user, and — important — **Publish it to "In
     production"**, or Google expires the refresh token after 7 days.
   - Enable the **Drive API**:
     https://console.cloud.google.com/apis/library/drive.googleapis.com
   - Download the client JSON → save as `secrets/drive_oauth_client.json`.
2. **One-time consent** (on a machine with a browser):
   ```
   python -m src.utils.drive_upload --authorize
   ```
   A browser opens; approve. A refresh token is written to
   `secrets/drive_oauth_token.json`. The daemon only *loads + refreshes* this
   token afterward — it never opens a browser.
3. **Make a folder** in your Drive (e.g. "Stim Stability"), open it, and copy the
   id from the URL `…/folders/<ID>` → that's `google_drive.folder_id`.

### Option B — Workspace Shared Drive (`auth_mode: service_account`)
Only if you have a Google Workspace Shared Drive (creation may be blocked by your
org admin). Enable the Drive API in `sheet-sync-473717`, create a Shared Drive +
folder, share it with the service account
`qc-monitor@sheet-sync-473717.iam.gserviceaccount.com` as **Content manager**,
set `auth_mode: service_account` and `shared_drive_folder_id: <folder id>`.

### Option C — No Drive
Leave `folder_id` blank. Figures still arrive as email attachments; the sheet
keeps the numeric columns (no hosted thumbnails).

---

## 2. `config/config.yaml`

```yaml
google_drive:
  auth_mode: oauth
  oauth_client_file: secrets/drive_oauth_client.json
  token_file: secrets/drive_oauth_token.json
  folder_id: "PASTE_YOUR_DRIVE_FOLDER_ID"
```
```yaml
  stim_stability_weekly:
    enabled: true
    weekday: 6            # Sunday (0=Mon .. 6=Sun)
    hour: 8
    recipients: ["you@umn.edu"]     # or [] to use alerting.smtp.recipients
    # sheet_id: ""        # blank = the existing BHZ workbook
    perfile_span: last_day          # last_day (~24 PNGs/ch) | week (all)
```

---

## 3. Enable + verify

The digest fires from the running `main.py` daemon's poll loop on Sunday ≥ 08:00.
To test now, temporarily set `weekday` to today's weekday and `hour` to the
current hour, restart the daemon, and within ~30 s confirm:

- the Drive folder gets a `YYYY-Www` subfolder full of PNGs,
- a **"Stim Stability"** tab appears with one row per channel (thumbnails + folder
  link),
- the email arrives with the figures.

Then set `weekday: 6`, `hour: 8` back. Re-running the same week updates the row in
place (idempotent).

## Notes
- `=IMAGE` thumbnails need the file readable by "anyone with the link"; the job
  sets that per file. On personal Drive this works; a locked-down Workspace may
  block it (the folder link still works).
- **kaleido** (optional) gives the impedance figure the exact dashboard look via
  Plotly; without it (or without Chrome) it falls back to matplotlib.
- Token stopped working after ~a week? The OAuth consent screen is still in
  "Testing" — set it to "In production" and re-run `--authorize`.
