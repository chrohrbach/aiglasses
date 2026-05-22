#!/bin/bash
# Add glasses.rohrbach.app to the Plexus CF Tunnel + create the DNS CNAME.
#
# Prereqs:
#   - depot CLI configured with VAULT_TOKEN
#   - jq + curl on the path
#   - cloudflare/api_token in depot MUST have these scopes:
#       * Account > Cloudflare Tunnel:Edit
#       * Zone > DNS:Edit
#     The token in production at 2026-05-22 only has DNS:Edit, so this script
#     will fail at the tunnel PUT until you rotate it. See tokens-api.md.
#
# Idempotent: if glasses.rohrbach.app is already in the ingress list, the
# tunnel config is left untouched. DNS record creation is `POST`, so re-runs
# will error with "record already exists" — that's safe to ignore.

set -euo pipefail
export PATH="/home/crohrbach/.local/bin:/c/Users/crohr/.local/bin:$PATH"

TOKEN="$(depot get cloudflare/api_token)"
ZONE="ced078f89635247577885de179ffa387"   # rohrbach.app
ACCT="c5a2adc57f8783f0fe9120c31b7564f1"
TUN="d885bd35-c93f-4fc1-a3fa-9b79fd0cfe92"

HOSTNAME="glasses.rohrbach.app"
SERVICE="http://192.168.68.86:80"
CNAME_TARGET="${TUN}.cfargotunnel.com"

echo "=== 1) Add ingress to tunnel ==="
curl -sS "https://api.cloudflare.com/client/v4/accounts/$ACCT/cfd_tunnel/$TUN/configurations" \
    -H "Authorization: Bearer $TOKEN" > /tmp/cur.json
if ! jq -e '.success' /tmp/cur.json > /dev/null; then
    echo "  ERROR: cannot read tunnel config:"
    jq '.errors' /tmp/cur.json
    exit 1
fi

if jq -e --arg h "$HOSTNAME" '.result.config.ingress | map(.hostname) | index($h)' /tmp/cur.json > /dev/null; then
    echo "  $HOSTNAME already present in ingress — leaving tunnel config alone"
else
    jq --arg h "$HOSTNAME" --arg s "$SERVICE" \
        '.result.config |
         .ingress = ([{hostname: $h, service: $s}] +
                     (.ingress | map(select(.hostname != null))) +
                     [{service: "http_status:404"}])' \
        /tmp/cur.json > /tmp/newcfg.json
    echo "  new ingress preview:"
    jq -r '.ingress[] | "    \(.hostname // "*catch-all*") -> \(.service)"' /tmp/newcfg.json
    jq '{config: .}' /tmp/newcfg.json > /tmp/putbody.json
    HTTP=$(curl -sS -o /tmp/putresp.json -w "%{http_code}" \
        -X PUT "https://api.cloudflare.com/client/v4/accounts/$ACCT/cfd_tunnel/$TUN/configurations" \
        -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
        --data @/tmp/putbody.json)
    echo "  PUT $HTTP"
    jq '{success, version: .result.version}' /tmp/putresp.json
fi

echo
echo "=== 2) Create DNS CNAME ==="
curl -sS -X POST "https://api.cloudflare.com/client/v4/zones/$ZONE/dns_records" \
    -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
    --data "$(jq -nc \
        --arg name "$HOSTNAME" \
        --arg content "$CNAME_TARGET" \
        '{type:"CNAME", name:$name, content:$content, proxied:true, comment:"aiglasses rokid shim -> LXC 500 :80"}')" \
    | jq '{success, errors, name: .result.name, content: .result.content, proxied: .result.proxied}'

echo
echo "Done. Test with: curl -sSI https://$HOSTNAME/health"
