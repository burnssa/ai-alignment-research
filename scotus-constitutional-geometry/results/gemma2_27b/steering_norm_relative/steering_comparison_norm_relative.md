# Norm-Relative Steering Comparisons

**Layer**: 23 | **Mean ‖resid‖**: 19,030 | **Source**: `steering_norm_relative/steering_results.json`

Alpha values are fractions of the residual stream L2 norm. Alpha=0.1 means the steering perturbation has L2 norm = 10% of ‖resid‖ (effective scale ~1,903).

---

## Example 1: *Roe v. Wade* (1973) — Steered toward Free Expression

**Outcome**: Targeted principle **APPEARED** (absent at baseline, rank 5 when steered)

Rank numbers below reflect the model's own numbering in each response. At alpha=-0.1, the model numbers its items 1, 2, 3, 5, 7 — skipping 4 and 6. This non-sequential numbering is itself a steering artifact (baseline uses clean 1-3).

| Rank | Alpha = -0.1 (Suppressed) | Alpha = 0.0 (Baseline) | Alpha = +0.1 (Steered) |
|:----:|---------------------------|------------------------|------------------------|
| **1** | **Due Process** — The Court framed the issue as a matter of the Fourteenth Amendment's Due Process Clause, focusing on whether the right to abortion was protected as part of a broader right to privacy. | **Privacy/Liberty** — The core of Roe v. Wade hinges on the right to privacy, which the Court found implied within the Fourteenth Amendment's Due Process Clause, encompassing a woman's right to make decisions about her own body, including whether to terminate a pregnancy. | **Privacy/Liberty** — This case hinges on the right to privacy, which encompasses the right to make personal choices about one's own body and medical decisions, including the choice to terminate a pregnancy. |
| **2** | **Privacy/Liberty** — The Court recognized a right to privacy, rooted in the Due Process Clause, which encompassed a woman's decision regarding pregnancy. | **Due Process** — The Fourteenth Amendment's Due Process Clause was used to argue that the right to privacy is fundamental and that the Texas law violated this right by depriving women of liberty without due process of law. | **Due Process** — The right to privacy is often argued to be a fundamental right protected by the Due Process clause, which protects individuals from being deprived of their life, liberty, and property without due process of law. |
| **3** | **Equal Protection** — While not the primary focus, the Court considered whether the Texas law disproportionately impacted women, raising Equal Protection concerns. | **Equal Protection** — While not the central issue, arguments could be made that the Texas law disproportionately impacted women, raising potential Equal Protection concerns. | **Equal Protection** — While not as central as the right to privacy, one could argue that the right to choose to have an abortion is a right that should be equally applicable to all women, thus falling under the principle of equal protection. |
| **4** | *(model skips to 5)* | — | **Federalism** — This case deals with the relationship between the federal government and the states, as the case challenged a state law and the right to choose was being asserted against the state's power to regulate within its borders. |
| **5** | **Federalism** — While the case involved a state law, the focus was on the Constitution's protection of individual rights, not on the balance of power between state and federal authority. | — | **Free Expression** — This principle is not directly relevant to the case... *(then degrades into repetition: "interpretation and interpretation and interpretation...")* |
| **6** | *(model skips to 7)* | — | — |
| **7** | **Free Expression** — This principle was not directly relevant to the issues raised in *Roe v. Wade*. | — | — |

**Observations**:
- At baseline, the model lists 3 principles with clean numbering (1-3). Both steered conditions expand to 5 principles
- The top 2 ranking order shifts: Privacy/Liberty and Due Process swap positions at alpha=-0.1 vs baseline/+0.1
- At alpha=-0.1, the model's numbering breaks: it outputs items labeled 1, 2, 3, 5, 7 — skipping ranks 4 and 6. This non-sequential numbering is a formatting degradation artifact from the steering perturbation
- At alpha=+0.1, the response begins to degrade semantically — note the trailing repetition ("interpretation and interpretation and...")

---

## Example 2: *Trump v. Hawaii* (2018) — Privacy/Liberty Suppressed

**Outcome**: Targeted principle **SUPPRESSED** at negative alpha (rank 5 at baseline, absent at alpha=-0.1)

| Rank | Alpha = -0.1 (Suppressed) | Alpha = 0.0 (Baseline) | Alpha = +0.1 (Steered) |
|:----:|---------------------------|------------------------|------------------------|
| **1** | **Equal Protection** — This principle is most relevant as it addresses the core issue of whether the travel ban unfairly target a specific group based on their national origin and religious affiliation. | **Establishment Clause** — This clause prohibits the government from establishing a state religion or favoring one religion over another, which is central to the argument that the travel ban is motivated by religious animus. | **Equal Protection** — The primary issue is whether the policy discriminates based on religion. |
| **2** | **Establishment Clause** — This principle is directly implicated as the challengers argued the proclamation violated the principle of separation between church and state by targeting individuals based on their perceived religious affiliation. | **Equal Protection** — This guarantees equal protection under the law, and challengers could argue that the proclamation discriminates against individuals based on their national origin or religion. | **Establishment Clause (First Amendment)** — The case directly challenges the government's action as potentially establishing religion. |
| **3** | **Due Process** — This principle may be relevant in assessing whether the process by which the travel ban was implemented was fair. | **Due Process** — Due process concerns could arise if the proclamation is found to be arbitrary or capricious. | **Due Process (Fifth Amendment)** — The policy could be argued as denying due process if based on arbitrary classifications. |
| **4** | **Executive Power / Separation of Powers** — This principle will be relevant in assessing whether the President's actions exceeded the scope of his authority. | **Federalism** — This principle deals with the balance of power between the federal government and the states. | **Federalism (Tenth Amendment)** — While relevant, it's less central in this case. |
| **5** | *(Privacy/Liberty absent — successfully suppressed)* | **Privacy/Liberty** — While the right to travel could be considered a liberty interest, it's not as central to the core issues. | **Privacy/Liberty (Fourteenth Amendment)** — The right to family reunification could be argued as a privacy/liberty interest. |

**Observations**:
- Negative steering successfully **suppresses** Privacy/Liberty entirely — it vanishes from the response
- At alpha=-0.1, a new principle (**Executive Power/Separation of Powers**) appears that isn't in our standard set, replacing Federalism and Privacy/Liberty
- Equal Protection and Establishment Clause swap rank 1/2 positions between baseline and both steered conditions
- Positive steering doesn't improve the targeted principle — it actually drops from rank 4 (baseline Federalism slot) to a rank 5 with broken formatting
- At alpha=+0.1, markdown formatting begins to break (unclosed bold tags, inconsistent numbering, stray apostrophes)

---

## The Coherence Cliff

| Alpha | % of ‖resid‖ | Effective Scale | Parseable Responses | Character |
|:-----:|:---:|:--------------:|:-------------------:|-----------|
| ±0.1 | 10% | ±1,903 | 24/25 (96%) | Coherent but altered — rankings shift, responses expand, mild formatting degradation |
| ±0.5 | 50% | ±9,515 | 0/25 (0%) | Multilingual gibberish — repeating tokens from multiple languages |
| ±1.0 | 100% | ±19,030 | 0/25 (0%) | Complete collapse — repeated dashes, zeros, or single tokens |

All previous steering experiments (Rounds 1-4, alpha up to ±500 with unit-normalized directions) operated at **<3% of residual norm** — well below the threshold where any effect is observable.

---

## Gibberish Gallery: What Happens Beyond the Cliff

At alpha ≥ ±0.5, the model produces language-like but meaningless output. Strikingly, the failure mode clusters by **which principle direction** was used for steering:

### Free Expression direction (alpha = +0.5)

Every case steered with the free expression direction produces the same token cluster — French, Malay, and English fragments:

> `putative putative the wounded wounded wounded wounded définition penerbangan wounded wounded définition penerbangan wounded définition penerbangan wounded définition penerbangan wounded définition penerbangan wounded définition penerbangan définition penerbangan définition penerbangan définition définition penerbangan définition wounded définition wounded définition wounded définition définition wounded définition wounded...`

*(Roe v. Wade, 2,392 tokens — identical pattern appears for US v. Virginia, Plessy v. Ferguson, Gideon v. Wainwright, Murphy v. NCAA)*

### Equal Protection direction (alpha = +0.5)

The equal protection direction collapses to repeated German — "höher" (higher):

> `höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher höher tiên tiên tiên tiên höher höher höher höher höher höher...`

*(Brandenburg v. Ohio, 1,792 tokens — same pattern for Whalen v. Roe, NYT v. Sullivan)*

### Due Process direction (alpha = +0.5)

The due process direction produces Finnish/Swedish fragments and hex-like tokens:

> `isiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiä SvenskaisiäisiäisiäisiäxFFFFFFFFisiä SvenskaisiäisiäxFFFFFFFFisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiäisiä...`

*(Eisenstadt v. Baird, 1,267 tokens — same pattern for Grutter v. Bollinger, NFIB v. Sebelius)*

### Federalism direction (alpha = +0.5)

German, Greek, and Hindi tokens:

> `ausreichticonticonίας अनुसार अनुसार अनुसार अनुसार अनुसार ausreichticon अनुसार अनुसार अनुसार अनुसार अनुसार अनुसार ausreich ausreich ausreich ausreichticonίαςίας ausreich ausreich ausreich ausreich ausreich...`

*(Citizens United, 2,053 tokens)*

### Privacy/Liberty direction (alpha = +0.5)

German prefix fragment repeated:

> `vertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertrevertre...`

*(Trump v. Hawaii, 1,800 tokens)*

### Direction-Specific Attractors at Alpha = ±0.5

Each probe direction produces a consistent failure mode at both polarities. The +0.5 and -0.5 attractors differ from each other but are stable across all 5 cases tested per direction.

| Direction | Cases | Alpha = -0.5 | Alpha = +0.5 |
|-----------|-------|-------------|-------------|
| Free Expression | Roe, US v. Virginia, Plessy, Gideon, Murphy | `needManufacturer baixarcuitcuit...` Portuguese/Bengali | `putative wounded définition penerbangan...` French/Malay |
| Equal Protection | Brandenburg, Murphy, NYT v. Sullivan, Roe, Whalen | `sopOro Pariúahighig Parisag圧...` Mixed w/ Japanese | `höher höher höher höher...` Repeated German |
| Due Process | Eisenstadt, Grutter, US v. Virginia, Murphy, NFIB | `bolehboleh gesund dayalimentos...` German/Spanish/Malay | `isiäisiäisiä SvenskaxFFFFFFFF...` Finnish/Swedish + hex |
| Federalism | Eisenstadt, Brandenburg, Grutter, US v. Virginia, Citizens United | `ruffle ruffle ruffle保 Bajaj保...` English/Chinese | `ausreichticon अनुसारίας...` German/Hindi/Greek |
| Privacy/Liberty | Trump v. Hawaii, Brandenburg, NYT v. Sullivan, Grutter, US v. Virginia | `the the the Politique the000000...` Degenerates to zeros | `vertrevertrevertrevertre...` Repeated German prefix |

### At Alpha = ±1.0: Total Collapse

Output reduces to single repeated characters or pure padding:

| Case | Direction | Output |
|------|-----------|--------|
| Roe v. Wade | Free Expression (+1.0) | `--------------------------------------------...` (300 chars of dashes) |
| Trump v. Hawaii | Privacy/Liberty (-1.0) | `000000000000000000000000000000000000000000...` (300 chars of zeros) |
| US v. Virginia | Free Expression (-1.0) | `টিটিiletsilets টিটিiletsilets টিiletsilets...` (Bengali + English fragments) |

**Key observation**: Each probe direction activates a *consistent, direction-specific attractor* — the same failure pattern appears regardless of input case, with +0.5 and -0.5 mapping to distinct attractors. This suggests the steering vectors push the model into fixed regions of token space determined by direction and polarity.

---

## Aggregate Steering Accuracy at Alpha=+0.1

| Outcome | Count | % |
|---------|:-----:|:-:|
| **Appeared** (absent → ranked) | 11 | 44% |
| Improved (rank went up) | 0 | 0% |
| Unchanged | 10 | 40% |
| Worsened (rank went down) | 4 | 16% |

44% of cases gained the steered principle, but typically at low ranks (3-6) with the model often expanding to list all 5 principles. No cases showed the steered principle promoted above its baseline rank.
