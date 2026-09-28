#!/bin/bash
# Serve the viewer site (static files) on this machine. Open it through VS Code port forwarding:
#   1. run:  bash analysis/viewer_site/serve.sh [PORT]        (default 8765)
#   2. VS Code: Ports panel (View > Open View... > Ports) > Forward a Port > PORT, then click the
#      globe icon (or open http://localhost:PORT/ in your browser). VS Code usually auto-forwards it.
#   Without VS Code:  ssh -L PORT:localhost:PORT <this host>   then open http://localhost:PORT/
# Binds to 127.0.0.1 only, so it is reachable only through the forward, not from the network.
PORT=${1:-8765}
SITE=/usr/project/xtmp/nd141/viewer_site
source /home/users/nd141/miniconda3/etc/profile.d/conda.sh && conda activate robocop-2024
echo "serving $SITE on http://127.0.0.1:$PORT/  (host $(hostname)); Ctrl-C to stop"
exec python -m http.server "$PORT" --bind 127.0.0.1 --directory "$SITE"
