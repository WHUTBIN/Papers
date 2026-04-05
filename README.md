# ================= 可视化 =================
def plot_results(history, pred, truth, lower, upper):
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Loss曲线
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_title('Training Curve (MSE Loss)')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. 预测区间宽度分布
    ax2 = plt.subplot(3, 2, 2)
    widths = (upper - lower).flatten()
    ax2.hist(widths, bins=50, alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean(widths), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(widths):.1f}mm')
    ax2.set_title('CP Interval Width Distribution')
    ax2.set_xlabel('Width (mm)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. 预测对比
    ax3 = plt.subplot(3, 2, 3)
    n_show = min(500, len(pred))
    ax3.plot(truth[:n_show], label='True', alpha=0.7, linewidth=1.5)
    ax3.plot(pred[:n_show], label='Predicted', alpha=0.7, linewidth=1.5)
    ax3.fill_between(range(n_show),
                     lower[:n_show].flatten(),
                     upper[:n_show].flatten(),
                     alpha=0.2, label='95% CP Interval')
    ax3.set_title('Prediction vs Truth with CP Intervals')
    ax3.set_xlabel('Time Steps')
    ax3.set_ylabel('Liquid Level (mm)')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. 散点图
    ax4 = plt.subplot(3, 2, 4)
    ax4.scatter(truth, pred, alpha=0.3, s=5)
    ax4.plot([truth.min(), truth.max()], [truth.min(), truth.max()],
             'r--', linewidth=2, label='Perfect')
    ax4.set_title('Predicted vs True')
    ax4.set_xlabel('True (mm)')
    ax4.set_ylabel('Predicted (mm)')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. 覆盖率随时间变化
    ax5 = plt.subplot(3, 2, 5)
    window = 100
    coverage = []
    for i in range(0, len(pred) - window, 10):
        hits = ((truth[i:i+window] >= lower[i:i+window]) & 
                (truth[i:i+window] <= upper[i:i+window]))
        coverage.append(np.mean(hits) * 100)
    ax5.plot(coverage, alpha=0.7, linewidth=2)
    ax5.axhline(95, color='red', linestyle='--', linewidth=2, label='Target 95%')
    ax5.set_title('Coverage Rate Over Time')
    ax5.set_xlabel('Window Index')
    ax5.set_ylabel('Coverage (%)')
    ax5.legend()
    ax5.grid(alpha=0.3)
    
    # 6. 误差分布
    ax6 = plt.subplot(3, 2, 6)
    errors = (pred - truth).flatten()
    ax6.hist(errors, bins=50, alpha=0.7, edgecolor='black')
    ax6.axvline(0, color='red', linestyle='--', linewidth=2)
    ax6.set_title('Prediction Error Distribution')
    ax6.set_xlabel('Error (mm)')
    ax6.set_ylabel('Frequency')
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_cti_breakdown(cti_results):
    """可视化CTI各维度得分"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 子图1: 雷达图
    ax1 = axes[0]
    categories = ['精确性\n(Accuracy)', '物理一致性\n(Physics)', '不确定性\n(UQ)']
    values = [cti_results['S_acc'], cti_results['S_phys'], cti_results['S_uq']]
    
    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    values += values[:1]
    angles += angles[:1]
    
    ax1 = plt.subplot(121, projection='polar')
    ax1.plot(angles, values, 'o-', linewidth=2, label=f'CTI = {cti_results["CTI"]:.3f}')
    ax1.fill(angles, values, alpha=0.25)
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories)
    ax1.set_ylim(0, 1)
    ax1.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax1.legend(loc='upper right')
    ax1.set_title('CTI 三维度雷达图', pad=20)
    ax1.grid(True)
    
    # 子图2: 柱状图
    ax2 = axes[1]
    x = np.arange(len(categories))
    bars = ax2.bar(x, [cti_results['S_acc'], cti_results['S_phys'], cti_results['S_uq']], 
                   color=['#2E86AB', '#A23B72', '#F18F01'], alpha=0.7, edgecolor='black')
    ax2.axhline(y=0.85, color='green', linestyle='--', linewidth=2, label='优秀线 (0.85)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    ax2.set_ylabel('得分')
    ax2.set_ylim(0, 1.0)
    ax2.set_title(f'CTI维度得分对比 (总分: {cti_results["CTI"]:.3f})')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 在柱子上显示数值
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
