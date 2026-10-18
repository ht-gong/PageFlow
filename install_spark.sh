#!/usr/bin/env bash

sudo apt-get update -y
sudo apt-get install -y openjdk-17-jdk

# Detect JAVA_HOME automatically
JAVA_PATH=$(readlink -f "$(which javac)")
JAVA_HOME=$(dirname "$(dirname "$JAVA_PATH")")

echo "Detected JAVA_HOME = $JAVA_HOME"

# Append to ~/.bashrc if not already present
if ! grep -q "export JAVA_HOME=" "$HOME/.bashrc"; then
  cat >> "$HOME/.bashrc" <<EOF

# --- Java environment ---
export JAVA_HOME=$JAVA_HOME
export PATH=\$JAVA_HOME/bin:\$PATH
EOF
  echo "Added JAVA_HOME to ~/.bashrc"
else
  echo "JAVA_HOME already configured in ~/.bashrc"
fi

# Apply immediately for current session
# shellcheck disable=SC1090
source "$HOME/.bashrc"

# Verify
echo "JAVA_HOME is now: $JAVA_HOME"