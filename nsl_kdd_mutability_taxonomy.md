# NSL-KDD 41-Feature Adversarial Mutability Taxonomy
### The networking spine of the case study — which adversarial perturbations are physically realizable over TCP/IP, and which are math fictions

## The core claim

Your paper's FGSM/PGD attacks perturb a **41-dimensional feature vector** by ε=0.2. But a feature vector is not traffic. An adversarial example is only a real threat if some actual packet sequence, sent by a real attacker against a real victim, *produces* that vector. The gradient doesn't know that:

- you can't send 0.2 of a byte,
- you can't lower a connection count below the connections your own attack makes,
- and you can't reach into a 100-connection history that includes traffic other machines sent to the victim.

So the question this project answers — a **networking** question, not an ML one — is: **of the 41 features, how many can an attacker actually control?** The perturbation is only as dangerous as its realizable subset.

## Classification scheme

Each feature gets a control class based on: *in a live attack over TCP/IP, can the attacker independently set this feature to an arbitrary target value?*

| Class | Meaning | Perturbation status |
|---|---|---|
| **A — Freely mutable** | Attacker sets it directly and independently | Realizable (subject to integer/validity rounding) |
| **B — Constrained / semi-mutable** | Attacker influences it, but bounded by protocol semantics, coupled to other features, asymmetric (can raise not lower), or binary/self-incriminating | Partially realizable |
| **C — Non-mutable** | Victim-controlled, emergent aggregate over other connections, categorical launch-time choice, or a fixed dataset constant | Not realizably perturbable |

## The 41 features

### Basic features (per-connection, from packet headers) — features 1–9

| # | Feature | Controlled by | Class | Realizability constraint |
|---|---|---|---|---|
| 1 | duration | Attacker (hold time) | **B** | ≥0, integer seconds; can pad but that changes attack timing/behaviour |
| 2 | protocol_type | Attacker (launch choice) | **C** | Categorical (tcp/udp/icmp). Flipping ≠ small perturbation — it's a different attack. Not gradient-perturbable |
| 3 | service | Attacker (launch choice) | **C** | Categorical (target port/service). Same as above |
| 4 | flag | Victim-coupled | **B** | Connection-resolution status (SF/S0/REJ…). Attack type biases it (SYN scan → S0/REJ) but it reflects how the victim responded — not freely settable |
| 5 | **src_bytes** | **Attacker** | **A** | The one genuinely free feature. Integer ≥0 — so sub-byte ε=0.2 is unrealizable; must round |
| 6 | dst_bytes | **Victim** | **C** | Bytes the victim sends back. Attacker only influences indirectly via the request |
| 7 | land | Attacker (spoof) | **B** | Binary; craftable via src=dst spoofing, but land=1 is itself an anomaly (Land attack) |
| 8 | wrong_fragment | Attacker | **B** | Small non-negative int; craftable via malformed fragmentation, but self-incriminating |
| 9 | urgent | Attacker | **B** | Small int; URG-flag packets; craftable but unusual |

### Content features (payload / host-state, need DPI) — features 10–22

| # | Feature | Controlled by | Class | Realizability constraint |
|---|---|---|---|---|
| 10 | hot | Victim/context | **C** | "Hot" behaviour indicators; emergent from session content |
| 11 | num_failed_logins | Attacker (attempts) | **B** | Attacker chooses how many tries; integer; but each try is real traffic |
| 12 | logged_in | Victim (auth) | **C** | 1 only if auth actually succeeded — can't be dialled |
| 13 | num_compromised | Victim state | **C** | Post-compromise count; success-dependent |
| 14 | root_shell | Victim state | **C** | Binary; 1 only if a root shell was actually obtained |
| 15 | su_attempted | Attacker | **B** | Attacker action, but binary and rare |
| 16 | num_root | Victim state | **C** | Root-access count; success-dependent |
| 17 | num_file_creations | Victim state (attacker-driven post-compromise) | **C** | Only meaningful after compromise; not independently settable |
| 18 | num_shells | Victim state | **C** | Same |
| 19 | num_access_files | Victim state | **C** | Same |
| 20 | num_outbound_cmds | — | **C** | **Constant 0 across the entire dataset** — a dead feature, zero variance, non-mutable by definition |
| 21 | is_host_login | Victim config | **C** | Depends on account type; ~always 0 in dataset |
| 22 | is_guest_login | Victim config | **C** | Depends on which account was used |

### Time-based traffic features (2-second window over other connections) — features 23–31

| # | Feature | Controlled by | Class | Realizability constraint |
|---|---|---|---|---|
| 23 | count | Attacker (volume), asymmetric | **B** | Attacker can *raise* it by generating connections (that's what a scan/flood does) but can't precisely *set* or *lower* it below their own footprint |
| 24 | srv_count | Attacker (volume), asymmetric | **B** | Same asymmetry, per-service |
| 25 | serror_rate | Victim + window | **C** | % SYN errors over the window; depends on victim port state + other connections |
| 26 | srv_serror_rate | Victim + window | **C** | Same, per-service |
| 27 | rerror_rate | Victim + window | **C** | % REJ errors; victim decides rejections |
| 28 | srv_rerror_rate | Victim + window | **C** | Same, per-service |
| 29 | same_srv_rate | Window aggregate | **C** | Ratio over the window; emergent (heavy single-service flooding can bias it — note this) |
| 30 | diff_srv_rate | Window aggregate | **C** | Emergent ratio |
| 31 | srv_diff_host_rate | Window aggregate | **C** | Emergent ratio |

### Host-based traffic features (window of last 100 connections to the destination) — features 32–41

Every one of these is an aggregate over the **last 100 connections to the victim host — including connections the attacker did not make.** The attacker cannot control what other machines send to the victim. This entire block is the strongest part of the argument.

| # | Feature | Controlled by | Class | Realizability constraint |
|---|---|---|---|---|
| 32 | dst_host_count | Third-party window | **C** | Includes other hosts' traffic to the victim |
| 33 | dst_host_srv_count | Third-party window | **C** | Same |
| 34 | dst_host_same_srv_rate | Third-party window | **C** | Emergent ratio over 100-conn history |
| 35 | dst_host_diff_srv_rate | Third-party window | **C** | Same |
| 36 | dst_host_same_src_port_rate | Third-party window | **C** | Same |
| 37 | dst_host_srv_diff_host_rate | Third-party window | **C** | Same |
| 38 | dst_host_serror_rate | Third-party window + victim | **C** | Victim-determined error rate over history |
| 39 | dst_host_srv_serror_rate | Third-party window + victim | **C** | Same |
| 40 | dst_host_rerror_rate | Third-party window + victim | **C** | Same |
| 41 | dst_host_srv_rerror_rate | Third-party window + victim | **C** | Same |

## The result

| Class | Count | Features |
|---|---|---|
| **A — freely mutable** | **1** | src_bytes |
| **B — constrained / semi-mutable** | **9** | duration, flag, land, wrong_fragment, urgent, num_failed_logins, su_attempted, count, srv_count |
| **C — non-mutable** | **31** | everything else |

**One feature of forty-one is freely attacker-settable. Nine are constrained. Thirty-one are effectively outside the attacker's control.**

Your paper's FGSM/PGD spread a perturbation budget across all 41 dimensions. In reality that budget collapses into ~10 dimensions — and most of those are integer-constrained, asymmetric (raise-only), or self-incriminating (a malformed-fragment spike is itself an IDS signature). The aggregate rate features that carry the most gradient signal (23–41) are precisely the ones the attacker controls least.

That's the finding. It reframes the paper's weakest point — stale synthetic data, 77% accuracy — into the object of study instead of something to defend.

## The experiment that proves it

Re-run the paper's attacks under a **realizability mask**:

1. **Zero the gradient on all Class C features** — the attack may only move A and B features.
2. **Integer-round** Class A/B features after each step (no fractional bytes/counts).
3. **Sign-constrain** the asymmetric ones — `count`/`srv_count`/`num_failed_logins` may only increase.
4. Compare accuracy degradation: unconstrained FGSM/PGD (your published numbers) vs. constrained.

Two possible outcomes, both publishable-grade for a case study:
- **Degradation mostly vanishes** → the real-world threat is far smaller than the paper implies; your adversarial-training defence looks even stronger under a realistic threat model.
- **Degradation survives on ~10 features** → sharper, scarier finding: an attacker needs to control only a handful of features to break the model, so those are the ones to harden.

Either way you've said something no other submission can: you interrogated an adversarial-ML result with **protocol semantics**.

## Why this is unbreakable for a Computer Networks course

- It's a **networking critique of an ML claim** — TCP/IP behaviour, connection semantics, windowing. Dead centre of the syllabus.
- It can't be replicated without your paper *plus* packet-level understanding — a lazy student can't copy it.
- It ties directly to the live lab: run the nmap scan, capture the PCAP, extract the features, and show on-screen which ones you could and couldn't manipulate on the wire.

## Open calls to pressure-test (be honest in the report)

- **flag (4):** arguably closer to C — it's mostly victim-determined. Defensible either way; state the reasoning.
- **count / srv_count (23–24):** the asymmetry (raise-only) is the crux — don't overclaim these as mutable.
- **same_srv_rate (29):** a sustained single-service flood *can* bias it, so a footnote rather than a hard C.
- The class boundaries are a **threat-model assumption**, not ground truth. Saying so out loud is what makes it unbreakable — you're inviting the reviewer to test it, and the experiment does.

## Next steps

1. Lock the taxonomy (this doc) — the report's Module 5 spine.
2. Implement the realizability mask in the FGSM/PGD code (a per-feature boolean vector + rounding + sign clamp).
3. Run constrained vs unconstrained; tabulate degradation.
4. From the live lab: capture one real attack, extract NSL-KDD-style features, annotate which the wire actually let you set — the money screenshot.