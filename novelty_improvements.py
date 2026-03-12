
from weasyprint import HTML

html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
    font-size: 10.5pt;
    color: #1a1a2e;
    background: #fff;
    line-height: 1.6;
    padding: 0;
  }

  /* Header Banner */
  .header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    color: white;
    padding: 36px 48px 28px 48px;
    page-break-inside: avoid;
  }
  .header .badge {
    display: inline-block;
    background: rgba(229, 57, 53, 0.85);
    color: white;
    font-size: 7.5pt;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 3px;
    margin-bottom: 10px;
  }
  .header h1 {
    font-size: 20pt;
    font-weight: 700;
    color: #ffffff;
    line-height: 1.25;
    margin-bottom: 8px;
  }
  .header .subtitle {
    font-size: 9.5pt;
    color: #a8b2d8;
    margin-top: 6px;
  }

  /* Main content area */
  .content {
    padding: 28px 48px 36px 48px;
  }

  /* Section heading */
  h2 {
    font-size: 10pt;
    font-weight: 700;
    color: #0f3460;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    border-left: 4px solid #e53935;
    padding-left: 10px;
    margin: 26px 0 10px 0;
    page-break-after: avoid;
  }

  /* Table styles */
  table {
    width: 100%;
    border-collapse: collapse;
    margin-bottom: 14px;
    font-size: 9.5pt;
    page-break-inside: avoid;
  }
  thead tr {
    background: #0f3460;
    color: white;
  }
  thead th {
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
    font-size: 9pt;
  }
  tbody tr:nth-child(even) { background: #f0f4ff; }
  tbody tr:nth-child(odd)  { background: #ffffff; }
  tbody td {
    padding: 7px 12px;
    border-bottom: 1px solid #dce3f5;
    vertical-align: top;
  }
  tbody td:first-child {
    font-weight: 600;
    color: #0f3460;
    white-space: nowrap;
    width: 200px;
  }

  /* Highlight box for top picks */
  .highlight-box {
    background: linear-gradient(135deg, #e8f4fd 0%, #f0e8ff 100%);
    border: 1.5px solid #7c4dff;
    border-radius: 6px;
    padding: 14px 18px;
    margin-top: 10px;
    page-break-inside: avoid;
  }
  .highlight-box h3 {
    font-size: 9.5pt;
    font-weight: 700;
    color: #4527a0;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
  }
  .highlight-box ol {
    padding-left: 18px;
  }
  .highlight-box li {
    font-size: 9.5pt;
    color: #2d1b69;
    margin-bottom: 5px;
    line-height: 1.5;
  }
  .highlight-box li strong {
    color: #4527a0;
  }

  /* Footer */
  .footer {
    background: #1a1a2e;
    color: #a8b2d8;
    font-size: 7.5pt;
    text-align: center;
    padding: 10px 48px;
    margin-top: 20px;
  }

  /* Page numbering */
  @page {
    size: A4;
    margin: 0;
    @bottom-center {
      content: "Page " counter(page) " of " counter(pages);
      font-size: 8pt;
      color: #888;
    }
  }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="badge">Research Novelty Report</div>
  <h1>Proposed Improvements &amp; Novel Contributions<br>for the RIS-MISO DRL Paper</h1>
  <div class="subtitle">
    Based on: <em>"Deep Reinforcement Learning Based Joint Downlink Beamforming and RIS Configuration<br>
    in RIS-aided MU-MISO Systems Under Hardware Impairments and Imperfect CSI"</em><br>
    IEEE ICC Workshops 2023 &nbsp;|&nbsp; Saglam, Gurgunoglu, Kozat
  </div>
</div>

<div class="content">

<!-- Section 1 -->
<h2>1 &nbsp; Algorithm-Level Improvements</h2>
<table>
  <thead><tr><th>Idea</th><th>Details</th></tr></thead>
  <tbody>
    <tr><td>Replace SAC with TD3 or PPO</td><td>The paper only uses SAC. Benchmarking TD3 (Twin Delayed DDPG) or PPO in this specific RIS setting is a direct novel contribution.</td></tr>
    <tr><td>Multi-Agent DRL (MARL)</td><td>Treat each user or RIS element as a separate agent using MADDPG or QMIX — especially relevant for distributed RIS deployments.</td></tr>
    <tr><td>Model-Based RL</td><td>Add a learned dynamics/channel model (e.g., Dreamer, MBPO) to reduce sample complexity beyond the 20,000-step training budget.</td></tr>
    <tr><td>Hierarchical RL</td><td>Separate high-level beam selection from low-level phase tuning with a hierarchical policy — addresses different timescales of adaptation.</td></tr>
    <tr><td>Transformer-based Policy</td><td>Replace MLP actor/critic with a Transformer backbone to exploit correlations across antenna and RIS element dimensions.</td></tr>
    <tr><td>Curriculum Learning</td><td>Start training on easy channel conditions (low noise, perfect CSI) and progressively increase difficulty to improve convergence speed.</td></tr>
  </tbody>
</table>

<!-- Section 2 -->
<h2>2 &nbsp; System Model Extensions</h2>
<table>
  <thead><tr><th>Idea</th><th>Details</th></tr></thead>
  <tbody>
    <tr><td>Multi-RIS / Distributed RIS</td><td>Current paper has a single RIS. Extending to multiple cooperative RIS surfaces is a highly publishable novel direction.</td></tr>
    <tr><td>Active RIS (vs. Passive)</td><td>Active RIS elements amplify signals; combining passive and active elements is a very hot current research direction.</td></tr>
    <tr><td>OFDM / Wideband Channels</td><td>The paper assumes narrowband. Extending to frequency-selective wideband channels adds significant realism and complexity.</td></tr>
    <tr><td>Non-Linear Energy Harvesting</td><td>Add SWIPT (Simultaneous Wireless Information and Power Transfer) objective alongside sum-rate maximization.</td></tr>
    <tr><td>Intelligent Omni-Surface (IOS)</td><td>IOS can transmit and reflect simultaneously; modeling this significantly extends the current PDA hardware model.</td></tr>
    <tr><td>Full-Duplex Base Station</td><td>Allow the BS to operate full-duplex with self-interference cancellation for higher spectral efficiency.</td></tr>
  </tbody>
</table>

<!-- Section 3 -->
<h2>3 &nbsp; Channel Model &amp; CSI Improvements</h2>
<table>
  <thead><tr><th>Idea</th><th>Details</th></tr></thead>
  <tbody>
    <tr><td>3GPP / Saleh-Valenzuela Channel</td><td>Replace Rayleigh fading with spatially correlated or geometry-based stochastic channels for higher realism.</td></tr>
    <tr><td>Deep CSI Feedback</td><td>Use autoencoders to compress and reconstruct CSI before feeding into the RL agent — reduces feedback overhead.</td></tr>
    <tr><td>Federated Learning for CSI</td><td>Distributed CSI acquisition with privacy constraints — FL + DRL hybrid is a novel and timely direction.</td></tr>
    <tr><td>Online Channel Tracking</td><td>Adapt to non-stationary channels using a Kalman filter or RNN-based channel predictor as state augmentation.</td></tr>
    <tr><td>Quantized Feedback</td><td>Extend beyond Gaussian channel estimation error to 1-bit or few-bit quantized CSI feedback models.</td></tr>
  </tbody>
</table>

<!-- Section 4 -->
<h2>4 &nbsp; Hardware Impairment Extensions</h2>
<table>
  <thead><tr><th>Idea</th><th>Details</th></tr></thead>
  <tbody>
    <tr><td>Non-Linear Power Amplifier (HPA)</td><td>Add amplifier distortion model at the BS in addition to the existing RIS phase-dependent amplitude model.</td></tr>
    <tr><td>Phase Noise at BS</td><td>Include oscillator phase noise for a more realistic hardware impairment model in the transmitter chain.</td></tr>
    <tr><td>Discrete Phase Shifts</td><td>Restrict to 1-bit, 2-bit, or 3-bit discrete phases and develop a quantization-aware DRL policy.</td></tr>
    <tr><td>Mutual Coupling (Extended PDA)</td><td>Extend the PDA model to include electromagnetic mutual coupling between adjacent RIS elements.</td></tr>
  </tbody>
</table>

<!-- Section 5 -->
<h2>5 &nbsp; Objective Function Improvements</h2>
<table>
  <thead><tr><th>Idea</th><th>Details</th></tr></thead>
  <tbody>
    <tr><td>Multi-Objective Optimization</td><td>Jointly optimize sum-rate AND energy efficiency (EE) using Pareto-based reward shaping or scalarization.</td></tr>
    <tr><td>Fairness Constraints</td><td>Add per-user minimum rate constraints (max-min fairness) on top of the current sum-rate objective.</td></tr>
    <tr><td>Secure Beamforming</td><td>Add eavesdroppers and optimize the physical layer secrecy rate instead of or alongside sum-rate.</td></tr>
    <tr><td>Latency-Aware Reward</td><td>Add delay or queue backlog penalty terms suitable for URLLC (Ultra-Reliable Low-Latency) scenarios.</td></tr>
  </tbody>
</table>

<!-- Section 6 -->
<h2>6 &nbsp; Training &amp; Sample Efficiency</h2>
<table>
  <thead><tr><th>Idea</th><th>Details</th></tr></thead>
  <tbody>
    <tr><td>Meta-Learning (MAML)</td><td>Train a meta-policy that quickly adapts to new channel environments with very few gradient steps.</td></tr>
    <tr><td>Transfer Learning</td><td>Pre-train on one scenario (e.g., L=16 RIS elements) and fine-tune on another (L=64) — compare convergence speed.</td></tr>
    <tr><td>Prioritized Experience Replay (PER)</td><td>Replace uniform sampling in utils.py with TD-error prioritized sampling for more efficient use of replay buffer.</td></tr>
    <tr><td>Offline RL / Imitation</td><td>Use expert trajectories from convex optimization solvers (e.g., CVX/CVXPY) as initial demonstrations for the agent.</td></tr>
  </tbody>
</table>

<!-- Section 7 -->
<h2>7 &nbsp; Benchmarking &amp; Baselines</h2>
<table>
  <thead><tr><th>Idea</th><th>Details</th></tr></thead>
  <tbody>
    <tr><td>Analytical Baselines (SDR / AO)</td><td>Compare DRL against Semi-Definite Relaxation (SDR) or Alternating Optimization (AO) near-optimal solutions.</td></tr>
    <tr><td>Classical + DRL Hybrid</td><td>Use convex optimization only for beamforming (convex sub-problem) and DRL only for RIS phase design — best of both worlds.</td></tr>
    <tr><td>Zero-Forcing + RL</td><td>Combine Zero-Forcing beamforming with RL-based RIS phase optimization as a strong hybrid baseline.</td></tr>
    <tr><td>Theoretical Upper Bound</td><td>Add capacity upper bound curves to learning figures to contextualize how close the agent gets to optimal.</td></tr>
  </tbody>
</table>

<!-- Section 8 -->
<h2>8 &nbsp; Code Architecture &amp; Reproducibility</h2>
<table>
  <thead><tr><th>Idea</th><th>Details</th></tr></thead>
  <tbody>
    <tr><td>Vectorized Environments</td><td>Use gym.vector.AsyncVectorEnv for parallel environment stepping — directly speeds up wall-clock training time.</td></tr>
    <tr><td>WandB / TensorBoard Logging</td><td>Replace manual .npy saves with experiment tracking dashboards for better monitoring and reproducibility.</td></tr>
    <tr><td>Hyperparameter Sweep</td><td>Systematic search over beta_min, batch_size, and learning rate using Optuna or Ray Tune.</td></tr>
    <tr><td>Docker Container</td><td>Package the full environment in a Docker image for perfectly reproducible results regardless of hardware.</td></tr>
  </tbody>
</table>

<!-- Highlight Box -->
<div class="highlight-box">
  <h3>🏆 Highest-Impact Novel Combinations (Recommended for a New Paper)</h3>
  <ol>
    <li><strong>Multiple RIS + Active RIS + MARL</strong> — Distributed multi-agent configuration with active elements added to passive surfaces.</li>
    <li><strong>Discrete Phase Shifts + Transformer Policy</strong> — Quantization-aware DRL with an attention-based architecture that understands inter-element correlations.</li>
    <li><strong>Meta-RL for Fast Channel Adaptation</strong> — Few-shot adaptation across channel conditions; addresses the paper's stochasticity limitation in a principled way.</li>
    <li><strong>Hybrid Optimization (AO + DRL)</strong> — Convex optimization for beamforming + RL for RIS phase shifts; show it outperforms pure DRL with fewer training steps.</li>
  </ol>
</div>

</div>

<!-- Footer -->
<div class="footer">
  Generated from analysis of /home/m-suman/IRS/new/conf_paper &nbsp;|&nbsp; March 2026
</div>

</body>
</html>
"""

HTML(string=html_content).write_pdf("/home/m-suman/IRS/new/conf_paper/novelty_improvements.pdf")
print("PDF generated successfully!")
