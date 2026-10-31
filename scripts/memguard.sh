#!/usr/bin/env bash
# Last line of defence on unified memory: if MemAvailable drops below a floor, kill the
# largest genaro process before the kernel OOM killer picks the desktop instead.
#   setsid nohup scripts/memguard.sh > logs/memguard.log 2>&1 &
FLOOR_GB=${FLOOR_GB:-6}
while true; do
  avail_kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  if (( avail_kb < FLOOR_GB * 1024 * 1024 )); then
    pid=$(ps -eo pid,rss,args --sort=-rss | awk '/[p]ython -m genaro/ {print $1; exit}')
    if [[ -n "$pid" ]]; then
      echo "$(date +%T) MemAvailable $((avail_kb / 1024)) MiB < ${FLOOR_GB} GiB: killing $pid ($(ps -o args= -p "$pid" | cut -c1-120))"
      kill "$pid"; sleep 5; kill -9 "$pid" 2>/dev/null
    fi
  fi
  sleep 2
done
