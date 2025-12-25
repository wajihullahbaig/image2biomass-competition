import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import defaultdict

# ==========================================
# LOG PARSER
# ==========================================
class KFoldLogParser:
    """Parse K-Fold training logs and extract metrics per fold/stage/epoch"""
    
    def __init__(self):
        # Structure: {fold: {stage: {component: {train: [], val: []}}}}
        self.data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'train': [], 'val': []})))
        self.metadata = {}
        # For handling Stage 2's separate Train/Val lines
        self.pending_train_data = None
        
    def parse_log_file(self, filepath):
        """Parse the log file and extract all metrics"""
        if not os.path.exists(filepath):
            print(f"❌ File {filepath} not found!")
            return False
            
        current_fold = None
        current_stage = None
        
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            
            # Extract metadata
            if "Backbone:" in line:
                self.metadata['backbone'] = line.split("Backbone:")[-1].strip()
            elif "Learning Rate:" in line:
                self.metadata['lr'] = line.split("Learning Rate:")[-1].strip()
            elif "Total Samples:" in line:
                self.metadata['total_samples'] = line.split("Total Samples:")[-1].strip()
            
            # Detect fold changes
            fold_match = re.search(r'Fold (\d+)/\d+', line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                continue
            
            # Detect stage changes
            if "STAGE 1:" in line or "Stage 1" in line:
                current_stage = "stage1"
            elif "STAGE 2:" in line or "Stage 2" in line:
                current_stage = "stage2"
            
            # Parse metric lines
            if current_fold is not None and current_stage is not None:
                # Stage 1 format: Stage1 F1 E12 | Train: [...] | Val: [...]
                stage1_match = re.search(
                    r'Stage1\s+F\d+\s+E(\d+)\s+\|\s+Train:\s+\[(.*?)\]\s+\|\s+Val:\s+\[(.*?)\]',
                    line
                )
                
                if stage1_match:
                    epoch = int(stage1_match.group(1))
                    train_metrics = stage1_match.group(2)
                    val_metrics = stage1_match.group(3)
                    
                    train_dict = self._parse_metrics(train_metrics)
                    val_dict = self._parse_metrics(val_metrics)
                    
                    self._store_metrics(current_fold, current_stage, epoch, train_dict, val_dict)
                    continue
                
                # Stage 2 format (separate lines):
                # Line 1: Stage2 E1 | Train: [...]
                # Line 2: Stage2 E1 | Val:   [...]
                stage2_train_match = re.search(
                    r'Stage2\s+E(\d+)\s+\|\s+Train:\s+\[(.*?)\]',
                    line
                )
                
                stage2_val_match = re.search(
                    r'Stage2\s+E(\d+)\s+\|\s+Val:\s+\[(.*?)\]',
                    line
                )
                
                if stage2_train_match:
                    epoch = int(stage2_train_match.group(1))
                    train_metrics = stage2_train_match.group(2)
                    train_dict = self._parse_metrics(train_metrics)
                    # Store temporarily, waiting for Val line
                    self.pending_train_data = (current_fold, current_stage, epoch, train_dict)
                    continue
                
                if stage2_val_match and self.pending_train_data:
                    epoch = int(stage2_val_match.group(1))
                    val_metrics = stage2_val_match.group(2)
                    val_dict = self._parse_metrics(val_metrics)
                    
                    # Combine with pending train data
                    fold, stage, train_epoch, train_dict = self.pending_train_data
                    if epoch == train_epoch:  # Verify epochs match
                        self._store_metrics(fold, stage, epoch, train_dict, val_dict)
                    self.pending_train_data = None
                    continue
        
        return len(self.data) > 0
    
    def _store_metrics(self, fold, stage, epoch, train_dict, val_dict):
        """Store train and val metrics"""
        for component, value in train_dict.items():
            self.data[fold][stage][component]['train'].append({
                'epoch': epoch,
                'value': value
            })
        
        for component, value in val_dict.items():
            self.data[fold][stage][component]['val'].append({
                'epoch': epoch,
                'value': value
            })
    
    def _parse_metrics(self, metric_string):
        """Parse metrics from string
        Stage 1: 'Tot:1.7429 Sp:2.673 Nd:0.283 H:2.190'
        Stage 2: 'Clover: 0.2246 | Dead: 0.3030 | Green: 0.5128 | Total: 1.8972'
        """
        metrics = {}
        
        # Match pattern: ComponentName: FloatValue (with optional spaces and pipes)
        # This handles both formats
        pattern = r'([A-Za-z]+)\s*:\s*([\d.]+)'
        matches = re.findall(pattern, metric_string)
        
        for name, value in matches:
            metrics[name] = float(value)
        
        return metrics
    
    def get_summary(self):
        """Get a summary of parsed data"""
        summary = {
            'folds': list(self.data.keys()),
            'stages': set(),
            'components': set(),
            'metadata': self.metadata
        }
        
        for fold_data in self.data.values():
            for stage, stage_data in fold_data.items():
                summary['stages'].add(stage)
                for component in stage_data.keys():
                    summary['components'].add(component)
        
        summary['stages'] = sorted(list(summary['stages']))
        summary['components'] = sorted(list(summary['components']))
        
        return summary


# ==========================================
# PLOTTER
# ==========================================
class KFoldPlotter:
    """Create plots for K-Fold training metrics"""
    
    def __init__(self, parser_data, output_dir="plots"):
        self.data = parser_data
        self.output_dir = output_dir
        
    def plot_all(self):
        """Generate all plots organized by stage and fold"""
        print("\n🎨 Generating plots...")
        
        for fold in sorted(self.data.keys()):
            for stage in sorted(self.data[fold].keys()):
                self._plot_fold_stage(fold, stage)
        
        print(f"✅ All plots saved to '{self.output_dir}/'")
    
    def _plot_fold_stage(self, fold, stage):
        """Plot all components for a specific fold and stage"""
        stage_data = self.data[fold][stage]
        
        # Create output directory
        save_dir = os.path.join(self.output_dir, stage, f"fold{fold}")
        os.makedirs(save_dir, exist_ok=True)
        
        # Plot each component
        for component in sorted(stage_data.keys()):
            self._plot_component(fold, stage, component, save_dir)
    
    def _plot_component(self, fold, stage, component, save_dir):
        """Plot train vs validation for a single component"""
        comp_data = self.data[fold][stage][component]
        
        # Convert to arrays
        train_epochs = [d['epoch'] for d in comp_data['train']]
        train_values = [d['value'] for d in comp_data['train']]
        val_epochs = [d['epoch'] for d in comp_data['val']]
        val_values = [d['value'] for d in comp_data['val']]
        
        # Skip if no data
        if not train_epochs and not val_epochs:
            return
        
        # Create figure
        plt.figure(figsize=(10, 6))
        
        if train_epochs:
            plt.plot(train_epochs, train_values, 
                    label='Train', marker='o', linewidth=2, markersize=4)
        
        if val_epochs:
            plt.plot(val_epochs, val_values, 
                    label='Validation', marker='s', linewidth=2, 
                    markersize=4, linestyle='--')
        
        # Styling
        plt.title(f'Fold {fold} - {stage.upper()} - {component.upper()}', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel(f'{component} Loss', fontsize=12)
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        
        # Save
        filename = os.path.join(save_dir, f"{component}_loss.png")
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()


# ==========================================
# STATISTICS PRINTER
# ==========================================
class StatsPrinter:
    """Print training statistics in a clean format"""
    
    def __init__(self, parser_data, metadata):
        self.data = parser_data
        self.metadata = metadata
    
    def print_summary(self):
        """Print comprehensive training summary"""
        print("\n" + "="*70)
        print("  📊 K-FOLD TRAINING SUMMARY")
        print("="*70)
        
        # Print metadata
        if self.metadata:
            print("\n📋 Configuration:")
            for key, value in self.metadata.items():
                print(f"  • {key.replace('_', ' ').title()}: {value}")
        
        # Print fold statistics
        for fold in sorted(self.data.keys()):
            print(f"\n{'─'*70}")
            print(f"📁 FOLD {fold}")
            print(f"{'─'*70}")
            
            for stage in sorted(self.data[fold].keys()):
                self._print_stage_stats(fold, stage)
    
    def _print_stage_stats(self, fold, stage):
        """Print statistics for a specific stage"""
        stage_data = self.data[fold][stage]
        
        print(f"\n  🔹 {stage.upper()}")
        print(f"  {'─'*60}")
        
        for component in sorted(stage_data.keys()):
            comp_data = stage_data[component]
            
            # Get final and best values
            if comp_data['train']:
                final_train = comp_data['train'][-1]['value']
                best_train = min(d['value'] for d in comp_data['train'])
            else:
                final_train = best_train = None
            
            if comp_data['val']:
                final_val = comp_data['val'][-1]['value']
                best_val = min(d['value'] for d in comp_data['val'])
                total_epochs = comp_data['val'][-1]['epoch']
            else:
                final_val = best_val = None
                total_epochs = len(comp_data['train']) if comp_data['train'] else 0
            
            # Print component stats
            print(f"\n    {component.upper()} (Epochs: {total_epochs})")
            
            if final_train is not None:
                print(f"      Train  → Final: {final_train:.4f}  |  Best: {best_train:.4f}")
            
            if final_val is not None:
                print(f"      Val    → Final: {final_val:.4f}  |  Best: {best_val:.4f}")


# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    # Configuration
    LOG_FILE = "./logs/originals/KFold_20251216_073644.log"
    OUTPUT_DIR = "logs/originals/KFold_20251216_073644_logs_plots"
    print("🚀 K-Fold Log Analysis Tool")
    print("="*70)
    
    # Parse log file
    print(f"\n📖 Parsing log file: {os.path.basename(LOG_FILE)}")
    parser = KFoldLogParser()
    
    if not parser.parse_log_file(LOG_FILE):
        print("❌ No data parsed. Check log file format.")
        return
    
    # Print summary of what was found
    summary = parser.get_summary()
    print(f"\n✅ Successfully parsed:")
    print(f"  • Folds: {summary['folds']}")
    print(f"  • Stages: {summary['stages']}")
    print(f"  • Components: {summary['components']}")
    
    # Print statistics
    stats_printer = StatsPrinter(parser.data, parser.metadata)
    stats_printer.print_summary()
    
    # Generate plots
    plotter = KFoldPlotter(parser.data, output_dir=OUTPUT_DIR)
    plotter.plot_all()
    
    print("\n" + "="*70)
    print("✨ Analysis Complete!")
    print(f"📁 Plots saved to: {OUTPUT_DIR}/")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()