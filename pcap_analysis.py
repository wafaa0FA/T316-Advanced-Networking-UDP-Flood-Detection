#!/usr/bin/env python3
"""
pcap_analysis.py - Automated PCAP Analysis Tool

Analyzes all PCAP files in a directory using tshark and generates
comprehensive statistics in CSV and JSON formats.

Requirements:
    - tshark installed (apt-get install -y tshark)
    - Python 3.8+
    - pandas, tqdm

Usage:
    python3 pcap_analysis.py --pcap-dir pcaps --out-dir results --workers 4
"""

import os
import sys
import json
import argparse
import logging
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil

# Configuration
DEFAULT_PCAP_DIR = 'pcaps'
DEFAULT_OUT_DIR = 'results'
DEFAULT_WORKERS = 4
DEFAULT_MIN_SIZE = 1024  # 1KB minimum

class PcapAnalyzer:
    """Main PCAP analysis class."""
    
    def __init__(self, pcap_dir, out_dir, workers=4, min_size=1024, recursive=False):
        self.pcap_dir = Path(pcap_dir)
        self.out_dir = Path(out_dir)
        self.workers = workers
        self.min_size = min_size
        self.recursive = recursive
        
        # Setup logging
        self.setup_logging()
        
        # Check tshark
        self.check_tshark()
        
        # Create output directory
        self.out_dir.mkdir(parents=True, exist_ok=True)
    
    def setup_logging(self):
        """Setup logging configuration."""
        log_file = self.out_dir / 'pcap_analysis.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_tshark(self):
        """Check if tshark is installed."""
        if not shutil.which('tshark'):
            self.logger.error("tshark is not installed!")
            self.logger.error("Install with: sudo apt-get update && sudo apt-get install -y tshark")
            sys.exit(1)
        
        # Check version
        try:
            result = subprocess.run(
                ['tshark', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            version = result.stdout.split('\n')[0]
            self.logger.info(f"Found {version}")
        except Exception as e:
            self.logger.warning(f"Could not verify tshark version: {e}")
    
    def find_pcap_files(self):
        """Find all PCAP files in the directory."""
        self.logger.info(f"Searching for PCAP files in: {self.pcap_dir}")
        
        if not self.pcap_dir.exists():
            self.logger.error(f"Directory not found: {self.pcap_dir}")
            sys.exit(1)
        
        # Find files
        if self.recursive:
            pattern = '**/*.pcap'
        else:
            pattern = '*/*.pcap'
        
        files = list(self.pcap_dir.glob(pattern))
        
        # Filter by size
        filtered_files = []
        for f in files:
            if f.stat().st_size >= self.min_size:
                filtered_files.append(f)
            else:
                self.logger.warning(f"Skipping {f.name} (size: {f.stat().st_size} bytes < {self.min_size})")
        
        self.logger.info(f"Found {len(filtered_files)} PCAP files (filtered from {len(files)})")
        
        return filtered_files
    
    def extract_experiment_id(self, pcap_path):
        """Extract experiment ID from path structure."""
        # Assuming structure: pcaps/<experiment-id>/<file>.pcap
        relative = pcap_path.relative_to(self.pcap_dir)
        if len(relative.parts) > 1:
            return relative.parts[0]
        return 'unknown'
    
    def run_tshark_command(self, pcap_path, args, timeout=30):
        """Run tshark command and return output."""
        cmd = ['tshark', '-r', str(pcap_path)] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                self.logger.warning(f"tshark error for {pcap_path.name}: {result.stderr}")
                return None
            
            return result.stdout
        
        except subprocess.TimeoutExpired:
            self.logger.error(f"tshark timeout for {pcap_path.name}")
            return None
        
        except Exception as e:
            self.logger.error(f"tshark exception for {pcap_path.name}: {e}")
            return None
    
    def get_packet_count(self, pcap_path):
        """Get total packet count."""
        output = self.run_tshark_command(pcap_path, ['-T', 'fields', '-e', 'frame.number'])
        if output:
            lines = output.strip().split('\n')
            return len([l for l in lines if l.strip()])
        return 0
    
    def get_timestamps(self, pcap_path):
        """Get first and last timestamps."""
        # Get first timestamp
        output = self.run_tshark_command(
            pcap_path, 
            ['-T', 'fields', '-e', 'frame.time_epoch', '-c', '1']
        )
        
        start_time = None
        if output and output.strip():
            try:
                start_time = float(output.strip())
            except:
                pass
        
        # Get last timestamp
        output = self.run_tshark_command(
            pcap_path,
            ['-T', 'fields', '-e', 'frame.time_epoch']
        )
        
        end_time = None
        if output:
            lines = [l.strip() for l in output.strip().split('\n') if l.strip()]
            if lines:
                try:
                    end_time = float(lines[-1])
                except:
                    pass
        
        return start_time, end_time
    
    def get_protocol_stats(self, pcap_path):
        """Get protocol statistics using frame.len."""
        # Method 1: Get total bytes directly from frame.len
        output = self.run_tshark_command(
            pcap_path,
            ['-T', 'fields', '-e', 'frame.len', '-e', 'frame.protocols']
        )
        
        stats = {
            'total_bytes': 0,
            'protocols': {}
        }
        
        if not output:
            return stats
        
        # Parse output line by line
        for line in output.strip().split('\n'):
            if not line.strip():
                continue
            
            parts = line.split('\t')
            if len(parts) >= 1:
                try:
                    frame_len = int(parts[0])
                    stats['total_bytes'] += frame_len
                    
                    # Extract protocols from frame.protocols field
                    if len(parts) >= 2:
                        protocols = parts[1].split(':')
                        for proto in protocols:
                            proto = proto.strip().upper()
                            if proto:
                                if proto not in stats['protocols']:
                                    stats['protocols'][proto] = {'packets': 0, 'bytes': 0}
                                stats['protocols'][proto]['packets'] += 1
                                stats['protocols'][proto]['bytes'] += frame_len
                except ValueError:
                    continue
        
        # Fallback: If total_bytes is still 0, try alternative method
        if stats['total_bytes'] == 0:
            stats = self.get_protocol_stats_fallback(pcap_path)
        
        return stats
    
    def get_protocol_stats_fallback(self, pcap_path):
        """Fallback method using capinfos for total bytes."""
        stats = {
            'total_bytes': 0,
            'protocols': {}
        }
        
        # Try using capinfos (more reliable for total size)
        try:
            result = subprocess.run(
                ['capinfos', '-M', str(pcap_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'Number of packets:' in line:
                        try:
                            packets = int(line.split(':')[1].strip())
                        except:
                            pass
                    elif 'File size:' in line or 'Capture file size:' in line:
                        try:
                            # Extract bytes from format like "1234567 bytes"
                            size_str = line.split(':')[1].strip()
                            bytes_val = int(size_str.split()[0])
                            stats['total_bytes'] = bytes_val
                        except:
                            pass
        except:
            pass
        
        # If capinfos failed, sum frame.len manually
        if stats['total_bytes'] == 0:
            output = self.run_tshark_command(
                pcap_path,
                ['-T', 'fields', '-e', 'frame.len']
            )
            
            if output:
                for line in output.strip().split('\n'):
                    try:
                        stats['total_bytes'] += int(line.strip())
                    except:
                        continue
        
        # Get protocol distribution with simple method
        for proto in ['eth', 'ip', 'tcp', 'udp', 'icmp', 'arp', 'dns', 'http', 'https', 'tls']:
            output = self.run_tshark_command(
                pcap_path,
                ['-Y', proto, '-T', 'fields', '-e', 'frame.len']
            )
            
            if output:
                proto_bytes = 0
                proto_packets = 0
                for line in output.strip().split('\n'):
                    if line.strip():
                        try:
                            proto_bytes += int(line.strip())
                            proto_packets += 1
                        except:
                            continue
                
                if proto_packets > 0:
                    stats['protocols'][proto.upper()] = {
                        'packets': proto_packets,
                        'bytes': proto_bytes
                    }
        
        return stats
    
    def get_protocol_counts(self, pcap_path):
        """Get TCP/UDP/ICMP packet counts."""
        counts = {'tcp': 0, 'udp': 0, 'icmp': 0}
        
        for proto in ['tcp', 'udp', 'icmp']:
            output = self.run_tshark_command(
                pcap_path,
                ['-Y', proto, '-T', 'fields', '-e', 'frame.number']
            )
            
            if output:
                lines = [l for l in output.strip().split('\n') if l.strip()]
                counts[proto] = len(lines)
        
        return counts
    
    def get_unique_ips(self, pcap_path):
        """Get unique source and destination IPs."""
        src_ips = set()
        dst_ips = set()
        
        # Get source IPs
        output = self.run_tshark_command(
            pcap_path,
            ['-T', 'fields', '-e', 'ip.src']
        )
        
        if output:
            for line in output.strip().split('\n'):
                ip = line.strip()
                if ip and ip != '':
                    src_ips.add(ip)
        
        # Get destination IPs
        output = self.run_tshark_command(
            pcap_path,
            ['-T', 'fields', '-e', 'ip.dst']
        )
        
        if output:
            for line in output.strip().split('\n'):
                ip = line.strip()
                if ip and ip != '':
                    dst_ips.add(ip)
        
        return len(src_ips), len(dst_ips)
    
    def get_top_flows(self, pcap_path, limit=10):
        """Get top flows by bytes."""
        flows = []
        
        # Get TCP conversations
        output = self.run_tshark_command(
            pcap_path,
            ['-q', '-z', 'conv,tcp']
        )
        
        if output:
            flows.extend(self.parse_conversations(output, 'TCP'))
        
        # Get UDP conversations
        output = self.run_tshark_command(
            pcap_path,
            ['-q', '-z', 'conv,udp']
        )
        
        if output:
            flows.extend(self.parse_conversations(output, 'UDP'))
        
        # Sort by bytes and return top
        flows.sort(key=lambda x: x['bytes'], reverse=True)
        return flows[:limit]
    
    def parse_conversations(self, output, protocol):
        """Parse tshark conversation output."""
        flows = []
        in_section = False
        
        for line in output.split('\n'):
            if '<->' in line and 'Frames' in line:
                in_section = True
                continue
            
            if in_section and '<->' in line:
                try:
                    parts = line.split()
                    if len(parts) >= 6:
                        # Parse addresses and ports
                        addr_parts = parts[0].split(':')
                        src = addr_parts[0] if len(addr_parts) > 0 else 'unknown'
                        sport = addr_parts[1] if len(addr_parts) > 1 else '0'
                        
                        addr_parts = parts[2].split(':')
                        dst = addr_parts[0] if len(addr_parts) > 0 else 'unknown'
                        dport = addr_parts[1] if len(addr_parts) > 1 else '0'
                        
                        packets = int(parts[3])
                        bytes_val = int(parts[4])
                        
                        flows.append({
                            'src': src,
                            'dst': dst,
                            'sport': sport,
                            'dport': dport,
                            'protocol': protocol,
                            'packets': packets,
                            'bytes': bytes_val
                        })
                except Exception as e:
                    continue
        
        return flows
    
    def analyze_pcap(self, pcap_path):
        """Analyze a single PCAP file."""
        experiment_id = self.extract_experiment_id(pcap_path)
        
        self.logger.info(f"Analyzing: {pcap_path.name} (Experiment: {experiment_id})")
        
        result = {
            'experiment_id': experiment_id,
            'pcap_file': pcap_path.name,
            'pcap_path': str(pcap_path.absolute()),
            'total_packets': 0,
            'total_bytes': 0,
            'duration_s': 0.0,
            'start_time': '',
            'end_time': '',
            'avg_bitrate_bps': 0,
            'tcp_packets': 0,
            'udp_packets': 0,
            'icmp_packets': 0,
            'unique_src_ips': 0,
            'unique_dst_ips': 0,
            'top_5_protocols': '',
            'notes': 'ok'
        }
        
        errors = []
        
        try:
            # Get packet count
            result['total_packets'] = self.get_packet_count(pcap_path)
            
            # Get timestamps
            start_time, end_time = self.get_timestamps(pcap_path)
            
            if start_time and end_time:
                result['start_time'] = datetime.fromtimestamp(start_time).isoformat()
                result['end_time'] = datetime.fromtimestamp(end_time).isoformat()
                result['duration_s'] = round(end_time - start_time, 3)
            else:
                errors.append('invalid timestamps')
            
            # Get protocol stats
            proto_stats = self.get_protocol_stats(pcap_path)
            result['total_bytes'] = proto_stats['total_bytes']
            
            # Calculate bitrate
            if result['duration_s'] > 0:
                result['avg_bitrate_bps'] = int((result['total_bytes'] * 8) / result['duration_s'])
            
            # Top protocols
            top_protocols = sorted(
                proto_stats['protocols'].items(),
                key=lambda x: x[1]['bytes'],
                reverse=True
            )[:5]
            result['top_5_protocols'] = ';'.join([p[0] for p in top_protocols])
            
            # Get protocol counts
            proto_counts = self.get_protocol_counts(pcap_path)
            result['tcp_packets'] = proto_counts['tcp']
            result['udp_packets'] = proto_counts['udp']
            result['icmp_packets'] = proto_counts['icmp']
            
            # Get unique IPs
            src_count, dst_count = self.get_unique_ips(pcap_path)
            result['unique_src_ips'] = src_count
            result['unique_dst_ips'] = dst_count
            
            # Get top flows for detailed JSON
            top_flows = self.get_top_flows(pcap_path)
            
            # Save detailed JSON
            json_output = self.out_dir / experiment_id
            json_output.mkdir(parents=True, exist_ok=True)
            json_file = json_output / f"{pcap_path.stem}.json"
            
            detailed_result = result.copy()
            detailed_result['protocol_details'] = proto_stats['protocols']
            detailed_result['top_flows'] = top_flows
            
            with open(json_file, 'w') as f:
                json.dump(detailed_result, f, indent=2)
            
            if errors:
                result['notes'] = ';'.join(errors)
            
        except Exception as e:
            self.logger.error(f"Error analyzing {pcap_path.name}: {e}")
            result['notes'] = f"error: {str(e)}"
        
        return result
    
    def analyze_all(self):
        """Analyze all PCAP files."""
        pcap_files = self.find_pcap_files()
        
        if not pcap_files:
            self.logger.warning("No PCAP files found!")
            return
        
        self.logger.info(f"Starting analysis of {len(pcap_files)} files with {self.workers} workers")
        
        start_time = datetime.now()
        
        # Analyze files in parallel
        with Pool(processes=self.workers) as pool:
            results = list(tqdm(
                pool.imap(self.analyze_pcap, pcap_files),
                total=len(pcap_files),
                desc="Analyzing PCAPs"
            ))
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        self.logger.info(f"Analysis completed in {duration:.2f} seconds")
        
        # Save results
        self.save_results(results)
        
        # Print summary
        self.print_summary(results, duration)
    
    def save_results(self, results):
        """Save results to CSV and JSON."""
        # Save CSV
        csv_file = self.out_dir / 'summary.csv'
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        self.logger.info(f"Saved CSV summary: {csv_file}")
        
        # Save JSON
        json_file = self.out_dir / 'summary.json'
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        self.logger.info(f"Saved JSON summary: {json_file}")
    
    def print_summary(self, results, duration):
        """Print analysis summary."""
        total_packets = sum(r['total_packets'] for r in results)
        total_bytes = sum(r['total_bytes'] for r in results)
        total_duration = sum(r['duration_s'] for r in results)
        
        experiments = set(r['experiment_id'] for r in results)
        
        print("\n" + "="*70)
        print("  ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nFiles analyzed:      {len(results)}")
        print(f"Experiments:         {len(experiments)}")
        print(f"Total packets:       {total_packets:,}")
        print(f"Total bytes:         {total_bytes:,} ({total_bytes/1024/1024:.2f} MB)")
        print(f"Total duration:      {total_duration:.2f} seconds")
        print(f"Analysis time:       {duration:.2f} seconds")
        print(f"Processing speed:    {len(results)/duration:.2f} files/second")
        
        print(f"\nOutput files:")
        print(f"  - {self.out_dir / 'summary.csv'}")
        print(f"  - {self.out_dir / 'summary.json'}")
        print(f"  - {self.out_dir}/<experiment-id>/*.json")
        print("="*70 + "\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Analyze PCAP files using tshark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--pcap-dir',
        type=str,
        default=DEFAULT_PCAP_DIR,
        help='Directory containing PCAP files'
    )
    
    parser.add_argument(
        '--out-dir',
        type=str,
        default=DEFAULT_OUT_DIR,
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--workers',
        type=int,
        default=DEFAULT_WORKERS,
        help='Number of parallel workers'
    )
    
    parser.add_argument(
        '--min-size',
        type=int,
        default=DEFAULT_MIN_SIZE,
        help='Minimum file size in bytes'
    )
    
    parser.add_argument(
        '--recursive',
        action='store_true',
        help='Search recursively for PCAP files'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("  PCAP ANALYSIS TOOL")
    print("="*70 + "\n")
    
    # Create analyzer
    analyzer = PcapAnalyzer(
        pcap_dir=args.pcap_dir,
        out_dir=args.out_dir,
        workers=args.workers,
        min_size=args.min_size,
        recursive=args.recursive
    )
    
    # Run analysis
    try:
        analyzer.analyze_all()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
