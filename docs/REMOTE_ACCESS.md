# Remote access setup (Cloudflare Tunnel + Cloudflare Access)

This runbook explains how to expose the QC Monitor dashboard to your
undergraduate mentees over the public internet, with SSO so each
person logs in with their UMN/Google email and every edit is
attributable.

## Prerequisites

- A domain on Cloudflare (free tier is fine). For the rest of this doc
  the example hostname is `qc.example.com` — substitute your own.
- The QC Monitor PC stays on whenever undergrads need access.
- `python main.py` already runs the dashboard locally on
  `http://localhost:8050`.

## Step 1 — install `cloudflared`

Download the Windows MSI from
<https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/>
and install it.

## Step 2 — create the tunnel

```powershell
cloudflared tunnel login
cloudflared tunnel create qc-monitor
```

Login opens a browser tab where you authorize against your Cloudflare
account. `tunnel create` prints a UUID and writes a credentials file
to `%USERPROFILE%\.cloudflared\<UUID>.json`.

## Step 3 — write the tunnel config

Save the file below as `%USERPROFILE%\.cloudflared\config.yml`
(replace `<UUID>` and the hostname):

```yaml
tunnel: <UUID>
credentials-file: C:\Users\<you>\.cloudflared\<UUID>.json

ingress:
  - hostname: qc.example.com
    service: http://localhost:8050
  - service: http_status:404
```

Then create the DNS record:

```powershell
cloudflared tunnel route dns qc-monitor qc.example.com
```

## Step 4 — install as a Windows service

```powershell
cloudflared service install
```

The tunnel now starts automatically on boot. Verify with:

```powershell
Get-Service cloudflared
cloudflared tunnel info qc-monitor
```

## Step 5 — configure Cloudflare Access (SSO)

In the Cloudflare Zero Trust dashboard
(<https://one.dash.cloudflare.com/>):

1. **Settings → Authentication** — add a Google identity provider (or
   keep the default one-time-PIN provider that emails a code to
   approved addresses).
2. **Access → Applications → Add an application → Self-hosted**:
   - Application domain: `qc.example.com`
   - Session duration: 24 hours
   - Identity providers: Google (and/or one-time PIN)
3. **Add a policy**:
   - Action: **Allow**
   - Selector: **Emails** — list each allowed user
     (you + each undergrad). Save.
4. After saving the application, click into it and copy the
   **Application Audience (AUD) Tag**. You'll need this for
   `config/config.yaml`.

Your team domain looks like `yourname.cloudflareaccess.com` — find it
under Settings → Custom Pages.

## Step 6 — wire the app into Access mode

Edit `D:\code\qc_monitor\config\config.yaml`:

```yaml
auth:
  enabled: true
  cf_team_domain: yourname.cloudflareaccess.com
  cf_audience: <AUD-TAG-FROM-STEP-5>
  dev_bypass: false
  dev_bypass_email: hoang392@umn.edu
```

Restart `python main.py`. From any device, browse to
`https://qc.example.com`. You should see the Cloudflare Access login
page; log in with an allowed email; then the dashboard loads with
"signed in as <your-email>" in the header.

## Step 7 — verify the gate works defense-in-depth

From a machine on the same LAN as the QC Monitor PC, try

```powershell
curl https://qc.example.com/media/video/1
```

without a valid Access cookie. You should get **403** from the Flask
hook even though Cloudflare Access isn't directly in the loop —
because the app verifies the JWT itself.

## Day-to-day: adding/removing users

Edit the Access policy in the Cloudflare dashboard (Step 5.3) — no
code change required, no app restart. Removed users lose access on
their next request.

## Local development

Set `auth.dev_bypass: true` in `config/config.yaml` to skip auth on
localhost. The dashboard will treat every request as
`auth.dev_bypass_email`. **Never** enable `dev_bypass` on a host that
is reachable from the public internet.

## Troubleshooting

- *403 on every page after enabling auth*: the AUD tag or team domain
  in `config.yaml` doesn't match the Access app. Check the Cloudflare
  dashboard.
- *Login loop*: cookies blocked by browser, or the Access session
  expired and the user is being re-prompted. Clear the
  `CF_Authorization` cookie for `qc.example.com` and retry.
- *Tunnel not connecting*: `cloudflared tunnel info qc-monitor` and
  `Get-Service cloudflared` show status; `cloudflared.log` lives in
  `C:\Windows\System32\config\systemprofile\.cloudflared\`.
