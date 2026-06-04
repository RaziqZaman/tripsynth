#!/usr/bin/env bash
set -euo pipefail

SCENARIO_DIR="${SCENARIO_DIR:-06x_stage1_md_internal}"
CONFIG="$SCENARIO_DIR/config_synthetic.xml"
LOG="$SCENARIO_DIR/matsim_synthetic.log"

echo "[$(date --iso-8601=seconds)] Building MATSim runner"
/usr/bin/time -f "build elapsed: %E" mvn -f 06_matsim/pom.xml -DskipTests package

echo "[$(date --iso-8601=seconds)] Starting MATSim: $CONFIG"
echo "MATSim writes detailed iteration/event logs below; this runner prints elapsed time when it exits."
MATSIM_HEAP="${MATSIM_HEAP:-96g}"
if [[ -z "${MAVEN_OPTS:-}" ]]; then
  export MAVEN_OPTS="-Xmx${MATSIM_HEAP} -XX:+UseG1GC"
fi
echo "MAVEN_OPTS=$MAVEN_OPTS"
/usr/bin/time -f "matsim elapsed: %E" mvn -f 06_matsim/pom.xml exec:java \
  -Dexec.args="$CONFIG" \
  2>&1 | tee "$LOG"

echo "[$(date --iso-8601=seconds)] Finished MATSim: $CONFIG"
