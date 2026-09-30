# Case Study Project Plan: Improving Network Security through Intrusion Detection Systems (IDS)
## 21CSC302J – Computer Networks | SRM Institute of Science and Technology | Topic 10

## TL;DR
- Build the report as a **Computer Networks project first, ML project second**: Modules 1–3 are packets/protocols/topology/signature detection in an isolated VirtualBox lab; the published IEEE paper (NSL-KDD + Random Forest + SHAP + FGSM/PGD + adversarial training) slots in as Modules 4–6, the "case study" proving how a modern IDS gets improved.
- **Recommended stack (no Packet Tracer):** VirtualBox host-only network with promiscuous mode "Allow All" on a dedicated IDS sensor VM; **Suricata 8.0.x** as primary IDS (Snort 3.12.x as comparison); Kali attacker VM + Ubuntu victim VM; Python + scikit-learn + SHAP + IBM ART for the ML case study. **GPU is NOT needed** — Random Forest trains on CPU in seconds.
- **Differentiator:** almost every other student will submit a static Packet Tracer topology or a copy-paste Snort tutorial. This project runs live attacks, catches them two ways (signature AND ML), and then attacks the ML detector itself — demonstrating the adversarial-robustness frontier that no other submission will have.

## Key Findings

**The re-anchoring strategy is the whole game.** The paper is ML-heavy. To be defensible in a Computer Networks viva, the first half of the report must be genuine networking: OSI/TCP-IP layering, the TCP three-way handshake, how a SYN scan / SYN flood / brute-force actually looks on the wire, where an IDS sits, and why a sensor needs promiscuous mode to see traffic not addressed to it. The ML paper then answers a networking question: "how do we make the anomaly-detection half of an IDS better and trustworthy?"

**Tool versions (verified, August 2026):**
- **Suricata:** 8.0.6 / 7.0.17 are the current stable releases (updated 13 July 2026) per the OISF GitHub Releases page; Suricata 7 has reached end-of-life at 7.0.17. Use 8.0.x (the `ppa:oisf/suricata-stable` PPA installs 8.0.x).
- **Snort 3:** 3.12.2.0, released 23 April 2026 (Cisco/Talos, GPLv2+), per snort.org and Wikipedia ("Snort 3.x: 3.12.2.0 / April 23, 2026"); note that per the Snort blog all Snort 3 versions up to and including 3.1.9.0 reached end-of-life 18 December 2025, so build on 3.12.x.
- **scikit-learn:** 1.8.0, released 10 December 2025, which "supports Python versions 3.11 to 3.14 and features support of free-threaded CPython," per the official scikit-learn 1.8.0 release announcement.
- **IBM Adversarial Robustness Toolbox (ART):** v1.20.1, released July 2025, per the Trusted-AI/adversarial-robustness-toolbox GitHub — MIT-licensed and an LF AI & Data **graduated** project (the highest maturity level in the Linux Foundation AI & Data Foundation). ART "implements 55+ attack methods and 30+ defense mechanisms across four threat categories: evasion, poisoning, extraction, and inference." **Use ART, not CleverHans** (CleverHans still runs but is no longer actively developed).

**NSL-KDD ~77% is not a bug, it's the honest number.** KDDTrain+ contains 22 attack types and KDDTest+ adds exactly **17 novel attack types absent from training** (per UNB CIC and multiple NSL-KDD analyses: "there are 17 types of novel attacks in the test dataset of NSL-KDD"). So the ~77% reflects generalisation to unseen attacks — a deliberate distribution-shift stress test. The 99% figures other students quote come from randomly splitting KDDTrain+ (train and test drawn from the same distribution), which leaks and inflates accuracy. This is a viva-winning point.

## Details

### 1. Module-Wise Breakdown

**Module 1 — Network Environment & Threat Modeling (networking-heavy)**
- *Objective:* Define the protected network, its assets, trust boundaries, and the threat model the IDS must address.
- *Inputs:* Course networking concepts (OSI/TCP-IP, IP addressing, subnetting, TCP/UDP/ICMP), a chosen topology.
- *Processing:* Draw the logical topology (attacker, victim/server, IDS sensor, virtual switch). Define the IP scheme (e.g., 192.168.56.0/24 host-only). Enumerate assets and likely attacks (recon, DoS, credential attacks) mapped loosely to MITRE ATT&CK tactics. Identify the IDS placement point.
- *Outputs:* Topology diagram, IP addressing table, threat model table, IDS placement justification.
- *Tools:* draw.io / diagrams.net (or Mermaid) for the diagram.

**Module 2 — Traffic Capture & Analysis (networking-heavy)**
- *Objective:* Capture and dissect normal vs malicious traffic at packet level.
- *Inputs:* Live traffic on the host-only segment.
- *Processing:* Use Wireshark/tshark/tcpdump to capture; analyse the TCP three-way handshake, a SYN scan's half-open connections, ICMP behaviour; apply display filters; export PCAPs.
- *Outputs:* Annotated PCAPs, screenshots of handshake and scan, protocol-hierarchy statistics.
- *Tools:* Wireshark, tshark, tcpdump.

**Module 3 — Signature-Based Detection (networking-heavy)**
- *Objective:* Detect known attacks via rule matching, demonstrating NIDS fundamentals.
- *Inputs:* Live/replayed traffic; Emerging Threats ruleset + custom rules.
- *Processing:* Deploy Suricata (primary) and optionally Snort 3 (comparison) on the sensor VM in promiscuous mode; write/enable rules; trigger alerts with live attacks; read fast.log / eve.json.
- *Outputs:* Alert logs, rule files, screenshots correlating each attack to its alert; a Suricata-vs-Snort comparison table.
- *Tools:* Suricata 8, Snort 3, Emerging Threats rules.

**Module 4 — ML-Based Anomaly Detection (case study begins)**
- *Objective:* Show how anomaly detection generalises beyond signatures using the NSL-KDD Random Forest.
- *Inputs:* NSL-KDD KDDTrain+ (125,973 records) and KDDTest+ (22,544 records), 41 features.
- *Processing:* Preprocess (one-hot encode protocol_type/service/flag, scale), train Random Forest (100 trees, max_depth=15, min_samples_split=5), evaluate on KDDTest+.
- *Outputs:* Baseline 77.04% clean accuracy, confusion matrix, per-class metrics, feature importances.
- *Tools:* Python, scikit-learn, pandas, numpy, matplotlib, seaborn.

**Module 5 — Adversarial Robustness & Explainability (the differentiator)**
- *Objective:* Attack the ML detector and defend it; keep explanations stable.
- *Inputs:* Trained RF model, test set, SHAP explainer.
- *Processing:* Generate FGSM (ε=0.2) and PGD (5 iterations, α=0.01, ε=0.2) adversarial examples; measure degradation; apply adversarial training; recompute SHAP explanations and check stability.
- *Outputs:* FGSM 70.60% → 80.60% and PGD 70.80% → 80.40% after adversarial training (clean stable at 76.75%); SHAP summary/force plots before and after; explanation-stability finding.
- *Tools:* IBM ART, SHAP, scikit-learn.

**Module 6 — Evaluation & Results / Integration**
- *Objective:* Tie signature + ML detection into one defense-in-depth narrative and evaluate.
- *Inputs:* All prior outputs.
- *Processing:* Compute accuracy, precision, recall, F1, FPR/FNR; compare signature vs anomaly detection strengths/weaknesses; discuss operational deployment.
- *Outputs:* Consolidated results tables, comparison discussion, limitations, future work.
- *Tools:* Python, spreadsheets.

### 2. Hardware Requirements

Random Forest on NSL-KDD is tiny by ML standards (125k rows × ~122 columns after encoding) — it trains on CPU in seconds to a couple of minutes. **No GPU is required or useful for Random Forest** (scikit-learn's RF is CPU-only; a GPU would sit idle). State this explicitly in the report — it pre-empts a common viva question.

*Minimum laptop:* quad-core CPU (Intel i5 8th-gen / Ryzen 5 or better), 8 GB RAM, 60 GB free SSD space, virtualization (VT-x/AMD-V) enabled in BIOS.
*Recommended:* 6–8 core CPU, 16 GB RAM, 100+ GB SSD. 16 GB makes running two-to-three VMs simultaneously comfortable.

*Virtual lab (host-only network, promiscuous "Allow All" on sensor):*
- **Attacker VM (Kali):** 2 vCPU, 2–4 GB RAM, ~25 GB disk.
- **Victim VM (Ubuntu Server / Metasploitable-style):** 1–2 vCPU, 2 GB RAM, ~15 GB disk.
- **IDS Sensor VM (Ubuntu + Suricata):** 2 vCPU, 4 GB RAM (Suricata is RAM-hungry with full rulesets), ~25 GB disk.
- On 8 GB RAM, run two VMs at a time (e.g., attacker + sensor) rather than all three; on 16 GB run all three.

### 3. Software Requirements

**Operating systems:** Host OS = Windows 11 or Linux. Guest OSes = Kali Linux (attacker; ships with nmap/hping3/hydra/Metasploit), Ubuntu 24.04 LTS (victim + IDS sensor).

**Virtualization — comparison and recommendation:**
- **VirtualBox (recommend):** free, GPLv3, easiest promiscuous-mode config for a student, host-only networking with a GUI toggle. Best fit.
- **VMware Workstation Player:** now free for personal use, slightly better performance, but promiscuous mode is fiddlier on host-only.
- **GNS3 / Mininet:** powerful for topology emulation but overkill and steeper; Mininet is SDN-oriented (good optional extension, not core).
- **Docker:** great for spinning up Suricata quickly, but container networking hides the L2 packet-visibility story you want to demonstrate — use VMs for the core lab, Docker optionally.
- Recommendation: **VirtualBox** for the core lab.

**IDS software — Suricata vs Snort:**
- **Suricata 8.0.x (recommend as primary):** multi-threaded, native JSON (eve.json) output, easy to parse, protocol detection, GPLv2. Current stable 8.0.6.
- **Snort 3.12.x (secondary/comparison):** Lua config, huge Talos community; GPLv2+. Current stable 3.12.2.0. Including a short Snort comparison strengthens the networking depth.
- Recommendation: build on **Suricata 8**, add a Snort 3 comparison section.

**Traffic capture/analysis:** Wireshark (GUI), tshark (CLI), tcpdump (lightweight capture).

**Attack generation (isolated lab only):** nmap (port/SYN scans), hping3 (SYN flood / custom packets), slowhttptest (slow-DoS), Hydra (SSH brute force), Metasploit (optional exploit demo). Keep all traffic inside the host-only network.

**Python + libraries:** Python 3.11–3.12; scikit-learn (RF, metrics), pandas, numpy, matplotlib, seaborn, SHAP, and **IBM ART 1.20.x** for FGSM/PGD/adversarial training (actively maintained LF AI graduated project; prefer over CleverHans).

**Diagramming:** draw.io / diagrams.net (free, offline desktop app) or Mermaid (text-based, version-controllable) for topology and data-flow figures.

### 4. Lab Setup Guide (isolated virtual network)

1. In VirtualBox, create a host-only network (File → Host Network Manager) — e.g., vboxnet0, 192.168.56.0/24, DHCP off.
2. Attach all three VMs' adapters to that host-only network. For the victim, optionally add a second NAT adapter for internet/package installs.
3. Assign static IPs: attacker .10, victim .20, sensor .30.
4. **Set the sensor VM's adapter Promiscuous Mode to "Allow All"** (Settings → Network → Advanced). This is the crux.
5. On the sensor, put the NIC in promiscuous mode at the OS level and point Suricata at it.
6. Verify with Wireshark on the sensor: you should see attacker→victim traffic even though it is not addressed to the sensor.

**The networking concept (write this up carefully — it demonstrates real understanding):** A normal NIC drops frames whose destination MAC isn't its own. Promiscuous mode disables that filter so the NIC passes *all* frames up the stack. But VirtualBox's virtual switch, by default, behaves like a real switch and only forwards frames to the port owning the destination MAC — so even a promiscuous guest NIC won't see other VMs' unicast traffic unless the *virtual switch* is told to allow it. Setting VirtualBox promiscuous mode to "Allow All" makes the virtual switch flood frames to the sensor port, emulating a **SPAN/mirror port** on a physical managed switch. That is exactly how a real out-of-band NIDS is fed traffic. "Allow VMs" restricts visibility to inter-VM traffic; "Allow All" is what an IDS sensor needs.

### 5. Demonstration Scenarios

**A. Nmap SYN scan**
- Command (attacker): `sudo nmap -sS -p 1-1000 192.168.56.20`
- Custom Suricata rule: `alert tcp any any -> $HOME_NET any (msg:"Nmap SYN Scan Detected"; flags:S; threshold:type threshold, track by_src, count 10, seconds 3; sid:1000001; rev:1;)`
- Expected alert in fast.log: an alert line firing on the SYN threshold from the attacker IP. (Emerging Threats also ships scan rules in the 2000000–2799999 SID range under the BSD license, e.g. SID 2100624 "SCAN SYN" derived from the arachnids-198 signature.)
- Wireshark: capture on sensor, filter `tcp.flags.syn==1 && tcp.flags.ack==0` to see the half-open SYNs.

**B. hping3 SYN flood**
- Command: `sudo hping3 -S -p 80 --flood 192.168.56.20`
- Custom rule: `alert tcp $EXTERNAL_NET any -> $HOME_NET any (msg:"Possible SYN flood / DoS"; flow:to_server; flags:S; threshold:type both, track by_dst, count 5000, seconds 5; classtype:misc-activity; sid:1000002; rev:1;)`
- Wireshark filter: `tcp.flags.syn==1 && tcp.flags.ack==0` with a high packet rate; use Statistics → I/O Graph to show the spike.

**C. Hydra SSH brute force**
- Command: `hydra -l root -P rockyou.txt ssh://192.168.56.20`
- Detection: Emerging Threats SSH brute-force rules, or a custom threshold rule on repeated connections to port 22.
- Wireshark filter: `tcp.port==22` to show repeated SSH sessions.

For each, screenshot: the attacker command, the Suricata alert, and the Wireshark capture — the three-way corroboration that makes the report credible.

### 6. IDS Placement & Architecture (for the report)

- **Perimeter:** IDS/IPS behind the edge firewall watching internet↔internal traffic — the baseline.
- **DMZ:** monitor public-facing servers separately.
- **Internal segments (east-west):** catch lateral movement; perimeter-only is no longer enough.
- **Inline IPS vs out-of-band IDS:** an IPS sits in the forwarding path and can drop packets (adds latency, can be a failure point); an IDS gets a *copy* via a **tap or SPAN port**, out-of-band, zero latency, but can only alert. SPAN is convenient but can drop packets under load; a tap gives cleaner copies. Many organisations run in IDS mode first to baseline traffic, then switch to inline IPS.
- **Segmentation & VLANs:** limit blast radius and give the IDS defined choke points to monitor; part of **defense-in-depth** (layered controls: firewall + segmentation + IDS/IPS + host controls + monitoring/SIEM).
- **Good topology diagram should show:** internet → edge firewall → DMZ + internal segments, the IDS sensor hanging off a SPAN/mirror port (out-of-band), VLAN boundaries, and where the ML anomaly engine consumes flow features. Mirror this in the virtual lab (host-only switch in "Allow All" = the SPAN port).

### 7. Dataset Section (NSL-KDD)

- **Lineage:** NSL-KDD (Tavallaee et al., 2009) is a cleaned version of KDD Cup 99, itself derived from the 1998 DARPA IDS evaluation. It removed KDD'99's massive record duplication that let trivial models hit 99%.
- **Splits:** KDDTrain+ (125,973 records), KDDTest+ (22,544 records), plus KDDTest-21 (a harder subset that drops the easiest-to-classify records). 41 features + label (+ difficulty score).
- **Feature groups:** basic (e.g., duration, protocol_type, service, flag, src_bytes, dst_bytes), content (e.g., hot, num_failed_logins, logged_in, root_shell), time-based traffic (2-second window: count, srv_count, error rates), host-based traffic (dst_host_* features over the last 100 connections).
- **Attack categories:** DoS, Probe, R2L, U2R.
- **Criticisms (be honest):** ~1999-era traffic; synthetic/simulated; no modern protocols, encryption, or app-layer attacks; severe class imbalance (U2R/R2L tiny — U2R has only 52 training records); poor representation of low-footprint attacks; unrealistic for today's networks.
- **Modern alternatives:** CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15 (49 features, 9 attack families), CIC-IoT2023.
- **Viva-proof justification for using it:** it is the reproducibility/benchmark standard for adversarial-ML-on-IDS research, small enough to run on a laptop with no GPU, and its built-in train/test distribution shift (17 novel attack types in the test set) makes it a *harder, more honest* benchmark — which is exactly the point when studying robustness. The contribution is the robustness/XAI method, not the dataset.

### 8. Viva Preparation (model answers)

- **IDS vs IPS:** IDS detects and alerts (out-of-band, via tap/SPAN, no latency, can't block); IPS sits inline and can drop/reset malicious traffic (adds latency, can fail closed/open).
- **Signature vs anomaly detection:** signature matches known patterns (low false positives, blind to novel attacks); anomaly models "normal" and flags deviations (catches zero-days, more false positives). Modern IDS combine both.
- **IDS placement & why:** perimeter for external threats, internal segments for lateral movement, DMZ for public servers — placement determines visibility; you monitor at choke points.
- **Promiscuous mode:** a NIC mode that accepts all frames, not just those addressed to its MAC — required for a sniffer/IDS.
- **SPAN port:** a switch feature that mirrors traffic from other ports to a monitor port so an out-of-band IDS can see it; my VirtualBox "Allow All" host-only switch emulates this.
- **Snort/Suricata rule structure:** action + header (protocol, src IP/port → dst IP/port) + options in parentheses (msg, flags, content, threshold, sid, rev). Example given in Module 3.
- **False positive vs false negative:** FP = benign flagged as attack (alert fatigue); FN = attack missed (dangerous). Tune thresholds to balance; in security an FN is usually costlier.
- **Why only ~77% on NSL-KDD when others report 99%?** The 99% papers randomly split KDDTrain+, so train and test come from the same distribution (leakage). KDDTest+ deliberately includes 17 attack types unseen in training, creating a distribution shift; ~77% is honest generalisation to unseen attacks.
- **FGSM (simple):** a one-step attack that nudges each input feature in the direction that most increases the model's error (the sign of the gradient), scaled by ε.
- **PGD (simple):** iterative FGSM — many small steps, each projected back into an ε-ball — a stronger attack.
- **SHAP (simple):** a game-theory method (Shapley values) that assigns each feature a contribution to a specific prediction, making the model's decision explainable.
- **Why explanation stability matters operationally:** a SOC analyst trusts the IDS only if the *reasons* for an alert stay consistent; if a tiny adversarial perturbation flips both the prediction and its explanation, the tool is untrustworthy. Stable SHAP under adversarial training means the defence preserves interpretability, not just accuracy.

### 9. Report Structure (25–35 pages)

1. Title, abstract, TOC — 2 pp.
2. Introduction & problem statement (network security context) — 2 pp.
3. Networking background: OSI/TCP-IP, protocols, attack anatomy — 4 pp.
4. IDS fundamentals: types, detection methods, placement, architecture (networking) — 4 pp.
5. Lab setup: VMs, host-only network, promiscuous/SPAN concept — 3 pp.
6. Signature detection: Suricata/Snort, rules, live demos — 4 pp.
7. Dataset (NSL-KDD, criticisms, alternatives) — 2 pp.
8. Case study — the IEEE paper: RF + SHAP + FGSM/PGD + adversarial training — 5 pp.
9. Results & evaluation (signature + ML integrated) — 3 pp.
10. Discussion, limitations, future work — 2 pp.
11. Conclusion + references — 2 pp.

Networking content ≈ pp. 1–17; the IEEE paper's contribution slots into section 8 (case study) and section 9.

### 10. References (verify DOIs resolve before submission)

1. Diana, L.; Dini, P.; Paolini, D. "Overview on Intrusion Detection Systems for Computers Networking Security." *Computers* 2025, 14(3), 87. DOI 10.3390/computers14030087. (Open access, CC BY — your primary networking-anchor reference.)
2. Gopa, P.; Grace Shalini, T. "Adversarial Robustness and Explainability in Intrusion Detection Systems: Maintaining XAI Integrity Under Adversarial Attacks." *2026 International Conference on Air Defense and Security (ICADS)*, IEEE (ISBN 9798331563219). DOI 10.1109/ICADS69450.2026.11545463. (Your case-study core.)
3. Tavallaee, M.; Bagheri, E.; Lu, W.; Ghorbani, A. "A Detailed Analysis of the KDD CUP 99 Data Set." *IEEE CISDA* 2009. DOI 10.1109/CISDA.2009.5356528. (The NSL-KDD paper.)
4. Goodfellow, I.; Shlens, J.; Szegedy, C. "Explaining and Harnessing Adversarial Examples." *ICLR* 2015. arXiv:1412.6572. (FGSM.)
5. Madry, A.; Makelov, A.; Schmidt, L.; Tsipras, D.; Vladu, A. "Towards Deep Learning Models Resistant to Adversarial Attacks." *ICLR* 2018. arXiv:1706.06083. (PGD.)
6. Lundberg, S.; Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." *NeurIPS* 2017. arXiv:1705.07874. (SHAP.)
7. Breiman, L. "Random Forests." *Machine Learning* 2001, 45(1), 5–32. DOI 10.1023/A:1010933404324.
8. Nicolae, M.-I. et al. "Adversarial Robustness Toolbox v1.0.0." arXiv:1807.01069. (ART.)
9. Sharafaldin, I.; Lashkari, A.; Ghorbani, A. "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization (CIC-IDS2017)." *ICISSP* 2018. DOI 10.5220/0006639801080116.
10. Moustafa, N.; Slay, J. "UNSW-NB15: A Comprehensive Data Set for Network Intrusion Detection Systems." *MilCIS* 2015. DOI 10.1109/MilCIS.2015.7348942.
11. Khan, N.; Ahmad, K.; Al Tamimi, A.; Alani, M. M.; Bermak, A.; Khalil, I. "Explainable AI-Based Intrusion Detection Systems for Industry 5.0 and Adversarial XAI: A Systematic Review." *Information* 2025, 16(12), 1036. DOI 10.3390/info16121036. (Recent, verified XAI-IDS survey.)
12. Ouazza, H.; Khennou, F.; Abdellaoui, A. "Adversarial Machine Learning for Secure and Explainable AI Systems: A Comprehensive Review." *Journal of Cybersecurity and Privacy* 2026, 6(4), 132. DOI 10.3390/jcp6040132.
13. Ribeiro, M.; Singh, S.; Guestrin, C. "Why Should I Trust You? Explaining the Predictions of Any Classifier (LIME)." *KDD* 2016. DOI 10.1145/2939672.2939778.
14. Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." *JMLR* 2011, 12, 2825–2830.
15. Suricata official documentation, OISF. https://docs.suricata.io
16. Snort 3 official documentation, Cisco/Talos. https://docs.snort.org
17. Wireshark User's Guide, Wireshark Foundation. https://www.wireshark.org/docs/
18. Emerging Threats Open ruleset, Proofpoint. https://rules.emergingthreats.net
19. NSL-KDD dataset, Canadian Institute for Cybersecurity, UNB. https://www.unb.ca/cic/datasets/nsl.html

## Recommendations

1. **Now:** Build the VirtualBox lab (attacker/victim/sensor on host-only, sensor promiscuous "Allow All"). Get Suricata alerting on an nmap scan — that single screenshot proves the whole networking premise. *Benchmark to advance:* a live alert appears in fast.log.
2. **Next:** Run all three demo scenarios and capture PCAPs. *Benchmark:* three attacks, three alerts, three Wireshark captures.
3. **Then:** Reproduce the paper's RF baseline (77.04% clean) and the FGSM/PGD + adversarial-training numbers (70.60→80.60% FGSM, 70.80→80.40% PGD, clean 76.75%); generate SHAP plots. *Benchmark:* your numbers land within a couple of points of the paper.
4. **Finally:** Write up with networking front-loaded. If a reviewer signals the report is "too ML," expand Modules 1–3 and the placement/architecture section; if they want more novelty, expand Module 5 (adversarial robustness).
5. **Differentiation check:** before submitting, ask "would a lazy student produce this?" If any section reads like a Packet Tracer screenshot or a copy-paste Snort quickstart, replace it with your live-lab evidence or the adversarial-robustness analysis. The combination of live signature detection + ML anomaly detection + attacking-the-detector is what no one else will have.

## Caveats
- **Verify the IEEE DOI resolves on IEEE Xplore before final submission.** The conference, publisher (IEEE), year (2026), ISBN (9798331563219), and the DOI numbering scheme are all confirmed real and internally consistent (sibling papers like …11545469 and …11545486 from the same proceedings are indexed), but as a Feb 2026 conference the specific record for …11545463 may still be propagating to third-party indexes; resolve the DOI in a browser to confirm.
- Only run attack tools inside the isolated host-only network. Never point nmap/hping3/Hydra at anything you don't own.
- Suricata 7 is EOL — use 8.0.x; Snort 3 versions up to 3.1.9.0 are EOL — use 3.12.x. Confirm current point-release versions at install time.
- Random Forest results vary slightly with random_state and library version; fix the seed (`random_state`) for reproducibility, and note your exact scikit-learn version in the report.
- Some sources on NSL-KDD report the KDDTest+ record count as 22,543 vs 22,544 and U2R test counts as 67 vs 200 depending on the file variant used; use the exact numbers from the files you actually load and state them.