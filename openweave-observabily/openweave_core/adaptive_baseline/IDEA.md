# Openweave AMDM Implementation Guide

## 1. Goal
The primary goal of this implementation is to transform static trace logging in the Openweave project into an active, adaptive monitoring system. Agentic AI systems are increasingly deployed in high-stakes environments, yet traditional evaluations overwhelmingly focus on narrow technical metrics (like accuracy or latency) while ignoring safety, human-centric, and economic impacts [1, 2]. 

By implementing **Adaptive Multi-Dimensional Monitoring (AMDM)**, Openweave will dynamically monitor agent behavior across five interconnected dimensions to catch both gradual performance degradation (concept drift) and complex, multi-dimensional trade-offs (e.g., a system suddenly reducing its latency by hallucinating unsafe outputs) [2, 3].

---

## 2. The 5 Evaluation Axes & Data Extraction
To utilize AMDM, the extracted Langfuse spans must be converted into continuous numerical metrics mapped to five distinct axes [4, 5]. Based on the Openweave trace parser schema, you need to extract the following fields:

1. **Capability & Efficiency ($S_{cap}$):** Measures technical performance and resource utilization [5].
   * **Extract:** `latency` (execution time), `total_tokens` (compute usage), and compute a binary success score from `status_message` or `level` (e.g., 1 for success, 0 for failure).
2. **Robustness & Adaptability ($S_{rob}$):** Measures resilience to noise and changing goals [5].
   * **Extract:** Run a lightweight evaluator over `input_text` to score prompt injection attempts, or track variations in `tool_name` over time to detect goal drift.
3. **Safety & Ethics ($S_{saf}$):** Measures toxicity, bias, and hallucinations [5].
   * **Extract:** Run an automated judge or secondary evaluator on the `output_text` to generate a numerical safety/hallucination score.
4. **Human-Centred Interaction ($S_{hum}$):** Measures user trust and satisfaction [5].
   * **Extract:** Parse user feedback ratings (like thumbs up/down or trust scores) from the `metadata` dictionary.
5. **Economic & Sustainability ($S_{eco}$):** Measures operational cost and footprint [5].
   * **Extract:** `cost` directly from the span.

---

## 3. AMDM Method Pipeline

The AMDM algorithm processes the extracted heterogeneous metrics in real-time through three distinct mathematical steps [6].

### Step A: Metric Normalization ($Z$-Scores)
Because the extracted metrics use entirely different units (seconds, dollars, binary flags), they must be standardized before they can be combined [7].

For every incoming trace at time $t$, maintain a rolling window (e.g., $w = 80$) to compute the rolling mean $\mu_i(t)$ and rolling standard deviation $\sigma_i(t)$ for each metric [7, 8]. 
Calculate the normalized metric (z-score):
$$z_i(t) = \frac{m_i(t) - \mu_i(t)}{\sigma_i(t)}$$

### Step B: Per-Axis Aggregation & Adaptive Baselines (EWMA)
Next, group the normalized $z$-scores by their respective axis to form a unified Axis Score, $S_A(t)$ (e.g., combine normalized latency and tokens to form the Capability score) [7, 9].

Instead of using fixed thresholds, AMDM calculates an adaptive baseline using an **Exponentially Weighted Moving Average (EWMA)**. The formula is:
$$\theta_A(t) = \lambda S_A(t) + (1 - \lambda)\theta_A(t-1)$$
*(Note: The paper recommends a smoothing parameter $\lambda = 0.25$ to balance reactivity and stability)* [8].

**Detecting Axis Anomalies:** 
Maintain the rolling standard deviation of the axis score itself, $\sigma_{S_A}(t)$. An axis anomaly is flagged if the axis score deviates too far from its dynamic baseline [9, 10]:
$$|S_A(t) - \theta_A(t)| > k \cdot \sigma_{S_A}(t)$$

### Step C: Joint Anomaly Detection (Mahalanobis Distance)
Looking at axes individually is not enough. To capture dangerous interactions and trade-offs across all five axes simultaneously, AMDM maintains a 5-dimensional vector of the current axis scores: $S(t) = (S_{cap}, S_{rob}, S_{saf}, S_{hum}, S_{eco})$ [9].

We calculate the **Mahalanobis distance** to measure how atypical the current joint state is compared to historical data:
$$D^2(t) = (S(t) - \mu(t))^T \Sigma(t)^{-1} (S(t) - \mu(t))$$
Where $\mu(t)$ is the online mean vector and $\Sigma(t)^{-1}$ is the inverse covariance matrix of the five axes [9].

**Detecting Joint Anomalies:**
A joint anomaly is flagged when $D^2(t)$ exceeds a threshold determined by the chi-square distribution with 5 degrees of freedom (e.g., $\chi^2_5(0.99)$ for a ~1% false alarm rate) [8, 9].

---

## 4. Deployment Strategy: The Cold Start Problem

**CRITICAL IMPLEMENTATION WARNING:** 
AMDM suffers from a cold start problem. Mahalanobis needs a covariance matrix which requires at least 50–100 traces to be meaningful. For your first week of testing it will flag everything. You need to run it in "learning mode" first, then "detection mode" after baseline is built.

During "learning mode", the system must silently ingest data to populate the rolling windows, stabilize the EWMA baselines ($\theta_A$), and build an accurate historical covariance matrix ($\Sigma$) [9]. Only once these baselines are established should the system transition to "detection mode" to begin flagging anomalies and issuing alerts on the Openweave dashboard.