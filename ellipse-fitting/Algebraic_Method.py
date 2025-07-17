import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import warnings
import glob

warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- CONSTANTS ---
h = 45.0 # effective radiator height

def fit_ellipse_outer_ring(x, y, n_bins=12):
    if len(x) < 10:
        return None, None, None
    cx, cy = np.mean(x), np.mean(y)
    dx, dy = x - cx, y - cy
    angles = np.mod(np.arctan2(dy, dx), 2*np.pi)
    radii = np.hypot(dx, dy)
    bin_indices = np.floor(angles / (2*np.pi) * n_bins).astype(int)
    outer_x, outer_y = [], []
    for b in range(n_bins):
        mask = bin_indices == b
        if not np.any(mask):
            continue
        # pick the hit with max radius in this angular bin
        local_idxs = np.nonzero(mask)[0]
        idx_max = local_idxs[np.argmax(radii[mask])]
        outer_x.append(x[idx_max])
        outer_y.append(y[idx_max])
    outer_x = np.array(outer_x)
    outer_y = np.array(outer_y)
    if len(outer_x) < 6:
        return None, None, None
    fit = fit_ellipse_direct(outer_x, outer_y)
    if fit is None:
        return None, None, None
    # inner points (optional, not used for fit): those significantly inside
    distances = np.hypot(x - cx, y - cy)
    # A simple threshold for inner points could be the minimum radius of the outer points
    if len(outer_x) > 0:
        thresh = np.min(np.hypot(outer_x - cx, outer_y - cy)) * 0.8
    else:
        thresh = np.min(radii)
    inner_mask = distances < thresh
    inner_x, inner_y = x[inner_mask], y[inner_mask]
    return fit, (outer_x, outer_y), (inner_x, inner_y)

def fit_ellipse_direct(x, y):
    # returns (center_x, center_y, semi_a, semi_b) or None
    try:
        D = np.vstack([x**2, x*y, y**2, x, y, np.ones_like(x)]).T
        S = D.T @ D
        if np.linalg.matrix_rank(S) < 6:
            return None
        C = np.zeros((6, 6)); C[0, 2] = C[2, 0] = 2; C[1, 1] = -1
        # Use pseudo-inverse for more stability
        M = np.linalg.pinv(S) @ C
        vals, vecs = np.linalg.eig(M)
        con = 4 * vecs[0, :] * vecs[2, :] - vecs[1, :]**2
        idx = np.where(con > 0)[0]
        if not len(idx):
            return None
        a_vec = vecs[:, idx[0]].real # Use real part
        A, B, Cc, Dc, Ec, F = a_vec
        B2_4AC = B**2 - 4*A*Cc
        if B2_4AC >= 0: # This should be < 0 for an ellipse
            return None
        cx = (2*Cc*Dc - B*Ec) / B2_4AC
        cy = (2*A*Ec - B*Dc) / B2_4AC
        num = 2 * (A*Ec**2 + Cc*Dc**2 - B*Dc*Ec + B2_4AC * F)
        term = np.sqrt((A - Cc)**2 + B**2)
        den_a = B2_4AC * (term - (A + Cc))
        den_b = B2_4AC * (-term - (A + Cc))
        if den_a == 0 or den_b == 0:
            return None
        sa = np.sqrt(np.abs(num / den_a))
        sb = np.sqrt(np.abs(num / den_b))
        # ensure semi_a >= semi_b
        if sa < sb:
            sa, sb = sb, sa
        params = (cx, cy, sa, sb)
        if not all(np.isfinite(p) for p in params):
            return None
        return params
    except np.linalg.LinAlgError:
        return None

def pca_angle(outer_x, outer_y, cx, cy):
    # compute PCA on outer-ring points to get major-axis orientation
    dx = outer_x - cx
    dy = outer_y - cy
    cov = np.cov(dx, dy)
    eigvals, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, np.argmax(eigvals)]
    angle = np.degrees(np.arctan2(major[1], major[0]))
    return angle

def calculate_incident_angle(a, b, h):
    """
    Calculates the muon incident angle based on the semi-axes of the projected
    Cherenkov ellipse and the effective radiator height.
    """
    if a is None or b is None or h is None or a <= 0 or b <= 0 or h <= 0:
        return None
    if a**2 < b**2:
        return None # Physically impossible
    e = np.sqrt(1-(b**2/a**2))
    # Add a safe guard for the arccos argument
    cos_arg = np.cos(np.arctan(b**2 / (a * h)))
    if not -1 <= e * cos_arg <= 1:
        return None
    angle = np.arcsin(e * cos_arg)
    return np.degrees(angle)


def plot_reconstructed_ellipse(points_outer, points_inner, cx, cy, a, b, angle_deg, true_angle, event_id, run_id, ax):
    xo, yo = points_outer
    xi, yi = points_inner
    ax.scatter(xi, yi, alpha=0.3, s=5, color='gray')
    ax.scatter(xo, yo, alpha=0.6, s=8, color='royalblue')
    e = Ellipse(xy=(cx, cy), width=2*a, height=2*b, angle=angle_deg,
                edgecolor='r', fc='None', lw=1)
    ax.add_patch(e)
    ax.set_title(f'{true_angle:.1f}°', fontsize=6)
    ax.set_aspect('equal', 'box')
    ax.axis('off')

def process_cherenkov_data(filename, sigma):
    df = pd.read_csv(filename)
    has_angle = 'angle_deg' in df.columns
    grouped = df.groupby(['run_id', 'event_id'])

    stats = {
        'total_events': len(grouped),
        'events_processed': 0,
        'failed_min_hits': 0,
        'failed_fit': 0,
        'failed_angle_calc': 0,
    }

    examples = {}
    results = []

    for (run_id, event_id), group in grouped:
        if not has_angle:
            continue

        stats['events_processed'] += 1
        true_angle = group['angle_deg'].iloc[0]
        x, y = group['x_mm'].values, group['y_mm'].values
        
        # Gaussian Smear
        sigma = 0.5 #units of mm
        x = x + np.random.normal(0, sigma, size=x.shape)
        y = y + np.random.normal(0, sigma, size=y.shape)

        if len(x) < 10:
            stats['failed_min_hits'] += 1
            continue

        fit, pts_o, pts_i = fit_ellipse_outer_ring(x, y)
        if fit is None:
            stats['failed_fit'] += 1
            continue
        cx, cy, a, b = fit

        recon = calculate_incident_angle(a, b, h)

        if recon is None:
            stats['failed_angle_calc'] += 1
            continue

        outer_x, outer_y = pts_o
        angle_deg = pca_angle(outer_x, outer_y, cx, cy)
        bin_key = round(true_angle)
        if bin_key not in examples:
            examples[bin_key] = (pts_o, pts_i, cx, cy, a, b, angle_deg)
        results.append({
            'run_id': run_id,
            'event_id': event_id,
            'true_angle': true_angle,
            'reconstructed_angle': recon,
            'a': a, 'b': b,
            'num_photons': len(x)
        })
    return pd.DataFrame(results), examples, stats

def plot_examples_grid(examples):
    if not examples:
        print("No examples for grid plotting."); return
    bins = sorted(examples.keys())
    n = len(bins)
    if n == 0: return
    ncols = int(np.ceil(np.sqrt(n)))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*2, nrows*2), constrained_layout=True)
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis('off')
    for idx, bin_key in enumerate(bins):
        ax = axes[idx]
        pts_o, pts_i, cx, cy, a, b, angle_deg = examples[bin_key]
        plot_reconstructed_ellipse(pts_o, pts_i, cx, cy, a, b, angle_deg, bin_key, None, None, ax)
    plt.suptitle("Example Reconstructions at Different Angles")
    plt.show()

def plot_results(results_df):
    df = results_df.dropna(subset=['true_angle', 'reconstructed_angle']).copy()
    if df.empty:
        print("No valid entries to plot."); return
    angle_diff = df['reconstructed_angle'] - df['true_angle']
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Reconstruction Performance Analysis', fontsize=16)

    ax1 = axes[0, 0]
    ax1.hist(angle_diff, bins=50, range=(-10, 10), alpha=0.75, edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Angle Difference (Reconstructed - True) [deg]')
    ax1.set_ylabel('Count')
    ax1.set_title('Error Distribution')
    mean_diff, std_diff = np.mean(angle_diff), np.std(angle_diff)
    ax1.text(0.05, 0.95, f'Mean: {mean_diff:.2f}°\nσ: {std_diff:.2f}°',
             transform=ax1.transAxes, va='top', bbox=dict(boxstyle='round', fc='wheat', alpha=0.5))

    ax2 = axes[0, 1]
    sc = ax2.scatter(df['true_angle'], df['reconstructed_angle'], alpha=0.5, s=10, c=df['num_photons'], cmap='viridis', vmin=0, vmax=np.percentile(df['num_photons'], 95))
    plt.colorbar(sc, ax=ax2, label='Num Photons')
    mn = min(df['true_angle'].min(), df['reconstructed_angle'].min())
    mx = max(df['true_angle'].max(), df['reconstructed_angle'].max())
    ax2.plot([0, mx], [0, mx], 'r--', linewidth=2)
    ax2.set_xlabel('True Angle [deg]'); ax2.set_ylabel('Reconstructed Angle [deg]')
    ax2.set_title('True vs Reconstructed'); ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal', 'box')

    ax3 = axes[1, 0]
    ax3.scatter(df['true_angle'], angle_diff, alpha=0.4, s=10)
    ax3.axhline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('True Angle [deg]'); ax3.set_ylabel('Difference [deg]')
    ax3.set_title('Error vs True Angle'); ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    max_angle = df['true_angle'].max()
    bins = np.arange(0, max(10, int(max_angle)) + 6, 5) # 5 degree bins
    df['angle_bin'] = pd.cut(df['true_angle'], bins=bins, right=False)
    box_data = [angle_diff[df['angle_bin'] == b].dropna().values for b in df['angle_bin'].cat.categories]
    labels = [f'{int(b.left)}-{int(b.right)}' for b in df['angle_bin'].cat.categories]
    valid = [(d, l) for d, l in zip(box_data, labels) if len(d) > 0]
    if valid:
        data_f, labels_f = zip(*valid)
        ax4.boxplot(data_f, tick_labels=labels_f)
    ax4.set_xlabel('True Angle Range [deg]'); ax4.set_ylabel('Difference [deg]')
    ax4.set_title('Error by Angle Range')
    ax4.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()
    

def print_summary_statistics(results_df, stats):
    """Prints a formatted summary of the fitting and reconstruction statistics."""
    print("\n" + "="*60)
    print("--- Overall Fit and Reconstruction Statistics ---")
    print("="*60)

    total_events = stats['total_events']
    events_processed = stats['events_processed']
    successful_fits = len(results_df)

    if events_processed == 0:
        print(f"Total events found in file: {total_events}")
        print("No events were processed (e.g., 'angle_deg' column might be missing).")
        print("="*60)
        return

    print(f"Total Events in File:         {total_events:6d}")
    print(f"Events with True Angle Data:  {events_processed:6d}")
    print("-" * 40)

    fail_min_hits = stats['failed_min_hits']
    fail_fit = stats['failed_fit']
    fail_angle = stats['failed_angle_calc']
    total_failed = fail_min_hits + fail_fit + fail_angle

    print("Failure Breakdown:")
    print(f"  - Insufficient photons (<10): {fail_min_hits:6d}")
    print(f"  - Ellipse fit failed:         {fail_fit:6d}")
    print(f"  - Physically invalid result:  {fail_angle:6d}")
    print(f"Total Failed Events:          {total_failed:6d}")
    print("-" * 40)

    success_rate = (successful_fits / events_processed) * 100 if events_processed > 0 else 0
    print(f"Successfully Reconstructed:     {successful_fits:6d} ({success_rate:.1f}% of processed)")

    if successful_fits > 0:
        print("\n--- Performance on Successfully Reconstructed Events ---")

        angle_diff = results_df['reconstructed_angle'] - results_df['true_angle']
        mean_err = np.mean(angle_diff)
        std_err = np.std(angle_diff)
        mae = np.mean(np.abs(angle_diff))
        rmse = np.sqrt(np.mean(angle_diff**2))

        print("\nAngle Error (Recon - True):")
        print(f"  - Mean Error (Bias):      {mean_err:8.3f} °")
        print(f"  - Std. Dev (Resolution):  {std_err:8.3f} °")
        print(f"  - Mean Absolute Error:    {mae:8.3f} °")
        print(f"  - Root Mean Square Error: {rmse:8.3f} °")

        print("\nReconstructed Ellipse Parameters (mm):")
        a_stats = results_df['a'].describe()
        b_stats = results_df['b'].describe()
        print(f"  - Semi-major axis 'a':  Mean={a_stats['mean']:.1f}, Std={a_stats['std']:.1f}, Median={a_stats['50%']:.1f}")
        print(f"  - Semi-minor axis 'b':  Mean={b_stats['mean']:.1f}, Std={b_stats['std']:.1f}, Median={b_stats['50%']:.1f}")

        print("\nPhoton Counts for Successful Events:")
        photon_stats = results_df['num_photons'].describe()
        print(f"  - Mean={photon_stats['mean']:.1f}, Std={photon_stats['std']:.1f}, Median={photon_stats['50%']:.0f}")

    print("="*60 + "\n")


if __name__ == "__main__":
    file_pattern = "../cherenkov_hits_*.csv"
    try:
        files = glob.glob(file_pattern)
        if not files:
            raise IndexError
        filename = files[0]
        print(f"Found data file: '{filename}'")

        #Amount to smear data
        smear = 0.0 # units of mm

        if (smear == 0):
            print(f"\nNo gaussian smearing applied.")
        else:
            print(f"\nGaussian smearing of {smear}mm applied.")

        results_df, examples, stats = process_cherenkov_data(filename, smear)

        print_summary_statistics(results_df, stats)

        if not results_df.empty:
            print(f"\nNow generating plots...")
            plot_examples_grid(examples)
            plot_results(results_df)
            
        else:
            print("\nANALYSIS FAILED: No events could be reconstructed.")
            
    except IndexError:
        print(f"Error: No file found matching the pattern '{file_pattern}'.")
        print("Please ensure your data CSV file is in the same directory.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
