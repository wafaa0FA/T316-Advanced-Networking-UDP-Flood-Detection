#!/bin/bash
# pcap_to_csv.sh - batch convert all .pcap in ~/tma_project/pcaps to CSV using tshark
PCAP_DIR=~/tma_project/pcaps
OUT_DIR=~/tma_project/pcaps_csv
mkdir -p "$OUT_DIR"

for f in "$PCAP_DIR"/*.pcap; do
  [ -f "$f" ] || continue
  base=$(basename "$f" .pcap)
  echo "Converting $f -> $OUT_DIR/${base}.csv"
  tshark -r "$f" -T fields \
    -e frame.number -e frame.time_epoch -e ip.src -e ip.dst \
    -e tcp.srcport -e tcp.dstport -e udp.srcport -e udp.dstport \
    -e ip.proto -e frame.len -e tcp.analysis.retransmission \
    -E header=y -E separator=, -E quote=d > "$OUT_DIR/${base}.csv"
done
echo "Done."
