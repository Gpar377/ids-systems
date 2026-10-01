# Viva cheat-sheet — IDS Realizability Case Study

SRM 21CSC302J · Computer Networks · Topic 10 · extends Gopa & Grace Shalini, ICADS 2026

---

## The 30-second pitch

> ML intrusion detectors are usually attacked by nudging all 41 NSL-KDD features freely.
> But an attacker can't edit numbers; they can only send packets, and over TCP/IP only
> **9 of 41** features can be moved. We re-ran the attacks limited to what the wire allows,
> measured the threat with the right metric (**evasion rate**), and built an IDS that is
> both **more accurate (77.9% → 89.6%)** and **far harder to evade (31% → 5%)**.

---

## Numbers to know cold

| | Paper's IDS | Adv-trained | **Our IDS** |
|---|---|---|---|
| Accuracy on KDDTest+ (unseen attacks) | 77.9% | 76.1% | **89.6%** |
| Precision / recall | 0.97 / 0.63 | — | **0.92 / 0.89** |
| Evasion, unrestricted PGD | 31.1% | 0.0% | **4.9%** |
| Evasion, realistic PGD | 13.0% | 7.1% | **3.0%** |

- Dataset: KDDTrain+ **125,973**, KDDTest+ **22,544**, **17** attack types appear only in test.
- Random split of the training file gives **99.8%**. That's the misleading number other people quote.
- Mutable features: **9 / 41** raw, **9 / 122** encoded (7.4%).
- Our threshold: **0.04** (paper's: 0.5). Chosen as the median over 5 splits → **89.6% ± 1.0%**.
- SHAP: **80.7%** of the model's importance is in features no attacker controls.
- Adaptive attacker's copy agreed with the defended model on **93.4%** of test data.
- 46 unit tests. Every number comes from `python src/experiment.py` → `results/`.

---

## Feature classes (be ready to defend any one)

| Class | Count | Features | Why |
|---|---|---|---|
| **A** free | 1 | `src_bytes` | attacker picks how much to send (whole bytes only) |
| **B** limited | 9 | `duration`, `flag`, `land`, `wrong_fragment`, `urgent`, `num_failed_logins`, `su_attempted`, `count`, `srv_count` | attacker influences, within rules |
| **C** uncontrollable | 31 | everything else | victim decides, or it depends on *other hosts'* traffic |

Constraints: `count`, `srv_count`, `num_failed_logins` **can only increase**. Bytes and counts must be **whole numbers**. Rates stay in **[0, 1]**. `flag` is B but is frozen in the attack because it's a category, so no small nudge exists.

---

## Demo flow (3 minutes) — `python demo/app.py` → http://127.0.0.1:8765

1. **Overview:** read the four cards, then point at the grey vs red bars.
2. **Attack lab:** click *"Pick an example that fools the paper's IDS"*. It's **EVADED**, but point at the red tiles: "22 of these values can't exist in a real packet."
3. Switch to **Realistic** and launch: **32 locks**, and only real changes remain.
4. Switch the target to **Our IDS**: it holds both ways.
5. **Network simulation → Recon scan:** `mscan` (never seen in training) is **MISSED** by the paper's IDS and caught by ours.
6. *(If captured)* **Live capture:** drop in the pcap, and real packets get scored.

---

## Likely questions and short answers

**Q: Why only 77.9%? Isn't that bad?**
KDDTest+ contains 17 attack types the model never saw, so it tests generalisation to new attacks. A random split gives 99.8% because it tests on attacks the model has already seen, which is data leakage.

**Q: Isn't lowering the threshold to 0.04 cheating?**
No. The threshold was chosen on *training* data: we held out whole attack types, trained on the rest, and picked the threshold that caught the hidden ones. The test set was never used. Across 5 different held-out selections it stays between 0.02 and 0.05, and accuracy stays within 88–91%. The trade-off is that precision drops from 0.97 to 0.92 (more false alarms), while missed attacks fall from 37% to 11%. For an IDS that's the right trade.

**Q: What are FGSM and PGD?**
Attacks that nudge input values in the direction that most lowers the model's "attack" score. FGSM makes one step of size ε. PGD makes many small steps and projects back into the allowed range after each one, which makes it stronger.

**Q: A Random Forest has no gradients. How do you attack it?**
With a transfer attack. The attacker trains their own neural-network copy of the IDS, crafts the attack on that copy, and sends it to the real IDS. That's also the realistic threat model, because a real attacker doesn't have your model.

**Q: Why not a simpler linear copy?**
We tried it. A linear model's gradient always points the same way, ±w, so the nudge direction gave away each sample's label, and adversarial training scored a fake 100%. We caught it, measured it (93/93 dimensions pushed opposite ways per class), and switched to an MLP.

**Q: How do you know the mask actually works?**
It applies constraints in *real units* (bytes, seconds, counts), not in scaled model space. That's the classic trap, because rounding a scaled value does nothing. Unit tests perturb a frozen feature and check the model's input is byte-identical, and check that bytes stay whole numbers and counts never decrease. They pass in both feature spaces.

**Q: Why can't the attacker control `dst_host_serror_rate`?**
It's computed over the **last 100 connections to the victim, from anyone**, including other users. The attacker can't reach into other people's traffic.

**Q: Why isn't `flag` attacker-controlled?**
It records how the connection *ended*: SF = completed normally, REJ = refused with a reset, S0 = SYN with no reply. The **victim's response** decides that, not the sender.

**Q: What's evasion rate, and why not accuracy under attack?**
Evasion rate asks: of the attacks the IDS catches, what fraction can the attacker disguise as normal? The paper's metric perturbs *every* connection, including normal users' traffic, which no attacker controls. When we did that to our IDS, accuracy fell 33 points, yet recall *rose* to 0.9995. The drop was false alarms, not attacks getting through.

**Q: Why is the adversarially trained model worse on realistic attacks (7.1%) than unrestricted ones (0%)?**
It was trained on impossible perturbations, so it learned to block those and not the realistic ones. Train against the threat you actually face.

**Q: Did you check that the defense holds if the attacker adapts?**
Yes. The attacker retrained their copy to mimic the *defended* model (93.4% agreement) and attacked again. Evasion barely changed, so there was no collapse.

**Q: Anything wrong with the original paper's attacks?**
PGD used α = 0.01 × 5 steps = 0.05 total movement, but ε = 0.2, so it never reached its own budget and was weaker than its own FGSM. We use α = 0.02 × 10 steps.

**Q: Where does an IDS sit in the network?**
Off a switch **mirror (SPAN) port**, or a network tap. It sees a copy of the traffic without being in its path, so it can't slow or break traffic. The sensor's interface runs in **promiscuous mode** so it accepts packets not addressed to it.

**Q: How do you get the 41 features from real packets?**
`lab/extract/pcap_to_nslkdd.py` groups packets into TCP connections. It derives `flag` from the handshake (SYN, SYN-ACK, FIN, RST) and computes the two NSL-KDD windows: connections in the **last 2 seconds**, and the **last 100 connections** to the same host. Content features (`logged_in`, `num_root`, …) never appear on the wire, so they come out as 0, which is itself why they're Class C.

**Q: Why not Cisco Packet Tracer?**
It simulates devices; it doesn't produce real packet captures with these features. We used an isolated Docker network with a real attacker, victim, third-party host and tcpdump sensor.

**Q: Is this novel? Hasn't constrained adversarial ML been done?**
The general idea has, and we cite it: Teuffenbach et al. 2020, Sheatsley et al., Merzouk et al. 2020, Apruzzese et al. 2022, Simonetto et al. 2022. Our contributions are (1) showing the standard metric confuses evasion with benign false alarms, (2) showing a calibrated threshold beats adversarial training under realistic constraints, and (3) checking feature controllability on real packets.

**Q: Limitations?**
- NSL-KDD is old (1999 traffic).
- One dataset; single-seed runs.
- Feature classes are assigned by hand (the live capture is the check on that).
- Transfer attacks only.
- **Functional realizability:** e.g. a `pod` (ping-of-death) attack's only realistic change is `src_bytes` 1480 → 0, which stops it being an attack. So our realistic evasion numbers are an **upper bound**, and the real threat is smaller still.

---

## Don't say

- ❌ "We're the first to consider realistic attacks." ✅ "We build on constrained-attack work and show…"
- ❌ "Our model gets 99%." ✅ "89.6% on unseen attacks, the honest protocol."
- ❌ "The attack helps accuracy." ✅ "The attack no longer transfers to the defended model."
- ❌ "The simulation shows live packets." ✅ "It replays real test-set connections through both IDSs."
- ❌ "SHAP is our contribution." ✅ "SHAP is supporting evidence: the model relies on features attackers can't touch."
