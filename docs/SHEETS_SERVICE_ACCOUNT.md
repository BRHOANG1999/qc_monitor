# Google Sheets via a service account

Use this when you want the QC Monitor's Surgeries tab to read private
sheets without "Publish to web". A Google Cloud service account is a
robot identity that authenticates with a JSON key — no browser flow, no
refresh tokens.

You'll do this once. Per-sheet you'll do a one-line Share.

## 1. Create a Google Cloud project (free)

Open <https://console.cloud.google.com/projectcreate>. Name it something
like `qc-monitor`. Pick "No organization" if it asks. Wait ~30 s for
provisioning.

## 2. Enable the Sheets API

In the project, go to
<https://console.cloud.google.com/apis/library/sheets.googleapis.com>
and click **Enable**.

## 3. Create the service account

1. <https://console.cloud.google.com/iam-admin/serviceaccounts> → **Create service account**.
2. Name: `qc-monitor-sheets`. Click **Create and continue**.
3. Skip "Grant this service account access to project" — not needed.
   Click **Done**.

## 4. Generate a JSON key

1. Click the new service account in the list.
2. **Keys** tab → **Add key** → **Create new key** → **JSON** → **Create**.
3. A JSON file downloads. Save it to a stable location on the QC PC,
   e.g. `D:\code\qc_monitor\secrets\sheets-sa.json`.
4. **Important**: do not commit this file. Add `secrets/` to `.gitignore`
   if it isn't there.

The JSON contains a `client_email` field — copy that address (looks
like `qc-monitor-sheets@qc-monitor-xxxxx.iam.gserviceaccount.com`).

## 5. Share each sheet with the service account

For each Google Sheet the dashboard should read:

1. Open the sheet in your browser.
2. Click **Share** (top right).
3. Paste the service account's email. Role: **Viewer** is enough.
4. **Uncheck** "Notify people" (the SA can't read email).
5. Click **Share**.

The sheet stays private to everyone else — you've only added a single
robot reader.

## 6. Wire into `config.yaml`

```yaml
surgeries:
  enabled: true
  refresh_minutes: 10
  service_account_file: secrets/sheets-sa.json   # or absolute path
  sheets:
    - label: "Surgery log"
      sheet_id: "17fm0UnfT2xd2C3FKbB5U_rlh1qt3gMaVBktyJgD6P1E"
      tab_name: "Surgeries"     # exact tab name -- shown at the bottom of the sheet
    - label: "Animal roster"
      sheet_id: "1Y6oCBTq44XleFaJkwiXubnUzEtM0wn-lPz2x0ydlr14"
      tab_name: "Roster"
```

The `sheet_id` is the long alphanumeric chunk in the sheet URL between
`/d/` and `/edit`. The `tab_name` is the exact label of the worksheet
tab (case-sensitive). Different tabs in the same workbook = different
entries with the same `sheet_id` but different `tab_name`.

Restart `python main.py`. Navigate to **Lab → Surgeries**. The footer
should now read `… (API) -- fetched <timestamp>` for each sheet.

## Adding more sheets later

Just append another entry under `sheets:` and share the new sheet with
the service account email. No GCP changes needed.

## Rotating the key

If the JSON ever leaks (committed by mistake, sent over email, etc.):

1. <https://console.cloud.google.com/iam-admin/serviceaccounts> → click
   the SA → **Keys** → delete the old key. The dashboard's next read
   will 401 until you replace it.
2. Generate a new key (Step 4) and overwrite the JSON file. No code
   change needed — the dashboard re-reads the file at next refresh.

## Troubleshooting

- **`The caller does not have permission`** in the footer: the sheet
  hasn't been shared with the service account's email. Re-do Step 5.
- **`Requested entity was not found`**: wrong `sheet_id` (check the
  URL between `/d/` and `/edit`) or wrong `tab_name` (case-sensitive).
- **`google.auth.exceptions.RefreshError`**: the JSON key was revoked
  or never matched. Regenerate (Step 4).
- **First load is slow.** First Sheets API call per process pays a
  handshake cost (~1 s). Subsequent reads are fast and cached for
  `refresh_minutes`.
