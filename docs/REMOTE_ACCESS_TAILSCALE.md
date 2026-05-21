# Remote access via Tailscale

Quick path for opening the QC Monitor dashboard to other machines you
own (laptop, surgery-room tablet, home PC). For broader access with
SSO — invited undergrads, collaborators outside your tailnet — use
`REMOTE_ACCESS.md` (Cloudflare Tunnel) instead.

## Prerequisites

- Tailscale is already installed and running on the QC Monitor PC. The
  SMB share at `100.106.104.22` is a tailnet IP, so this is true.
- `python main.py` already serves the dashboard locally on
  `http://localhost:8050`.
- `config.yaml` already has `dashboard.host: 0.0.0.0` so the server
  binds to every interface (including Tailscale's virtual NIC). No
  config changes needed.

## Step 1 — Install Tailscale on the remote PC

Download from <https://tailscale.com/download> and sign in with the
*same identity* that owns this PC's tailnet membership. The new PC
appears in your admin console at <https://login.tailscale.com/admin/machines>.

## Step 2 — Find the QC PC's tailnet IP

From the QC PC:

```powershell
tailscale ip -4
```

Should print `100.106.104.22` (matches the SMB share you already
use). Use that IP in the URL below.

## Step 3 — Open Windows Firewall for port 8050 (QC PC, one-time)

The firewall rule is scoped to the Tailscale virtual interface so
port 8050 stays closed to your regular LAN and the public internet.

```powershell
New-NetFirewallRule -DisplayName "QC Monitor dashboard (Tailscale)" `
  -Direction Inbound -LocalPort 8050 -Protocol TCP -Action Allow `
  -InterfaceAlias "Tailscale"
```

If `InterfaceAlias` errors out, list adapters with `Get-NetAdapter` and
substitute the actual Tailscale NIC name (`Tailscale`, `tailscale0`, or
similar depending on version).

## Step 4 — Verify

From the remote PC, browse to <http://100.106.104.22:8050>. You should
see the QC Monitor dashboard. The header reads "signed in as
`<dev_bypass_email>`" because `config.yaml` has `auth.dev_bypass: true`
— acceptable on a tailnet because only authorized tailnet members can
reach that IP at all.

## Security boundary

Tailscale ACLs default to "any member device can reach any other
member device." If you ever add a less-trusted device to your tailnet
(e.g. a collaborator's laptop), tighten the ACL in the Tailscale admin
console before they connect, or move that user to the Cloudflare Tunnel
path in `REMOTE_ACCESS.md`. Never set `dashboard.host: 0.0.0.0` *and*
`auth.dev_bypass: true` on a machine that's reachable from the public
internet.

## Troubleshooting

- **Times out / connection refused.** The firewall rule above didn't
  apply. Re-run `New-NetFirewallRule` and confirm with
  `Get-NetFirewallRule -DisplayName "QC Monitor*"`.
- **Page loads but shows 'unauthenticated'.** `auth.dev_bypass` is
  `false`. Either set it to `true` (tailnet) or follow `REMOTE_ACCESS.md`
  (Cloudflare SSO).
- **`tailscale ip -4` prints nothing.** Tailscale isn't running.
  `Start-Service Tailscale` then retry.
- **IP changed after a reinstall.** Tailnet IPs are sticky per device
  but reissued if you re-add a device. Re-run Step 2.
