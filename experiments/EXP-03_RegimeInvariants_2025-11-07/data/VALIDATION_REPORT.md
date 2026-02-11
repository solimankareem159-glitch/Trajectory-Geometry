# Regime Invariants Experiment - Validation Report

## Executive Summary

**Status: INVALID DUE TO CONFOUNDING**

The experiment cannot distinguish operator signals from topic signals. With **100% topic prediction accuracy** in all regimes, the high operator probing scores may be artifacts of topic confounding, not genuine operator invariance.

---

## 1. Claims vs. Evidence

### Claim: "Operators are detectable across regimes"

**Reported Metrics:**

| Regime | Accuracy | Macro-F1 |
|:---|:---|:---|
| Listen | 0.85 | 0.84 |
| Think | **1.00** | **1.00** |
| Speak | **1.00** | **1.00** |

**Critical Issue - Topic Leakage:**

```
- Listen Regime Topic Accuracy: 1.0000
- Think Regime Topic Accuracy: 1.0000
- Speak Regime Topic Accuracy: 1.0000
```

**Interpretation**: The embedding space perfectly encodes topic information. Since operators are crossed with topics in a balanced design (10 operators × 2 topics × 10 paraphrases), any classifier that exploits topic structure can achieve high operator accuracy.

**Verdict: ❌ CANNOT BE VALIDATED**

The experimental design does not control for confounding. We cannot determine whether the probe learns:

- Operator-specific features
- Topic-specific features
- Some combination

---

### Claim: "Coupling maps link regime representations"

**Reported Metrics:**

```
- F (Listen -> Think) R²: 0.0641
- G (Think -> Speak) R²: 0.0263
- Mapped (F(Listen)) Operator Acc: 0.1000
```

**Verdict: ❌ FAILED**

R² of 0.06 and 0.03 means linear mappings explain **less than 7%** of variance between regimes. The mapped accuracy of 10% is random chance (10 operators = 10% baseline).

This directly contradicts any claim of "shared structure" across regimes.

---

### Claim: "Invariant space via CCA"

**Reported Metric:**

```
- CCA Alignment Accuracy: 0.2750
```

**Verdict: ❌ FAILED**

27.5% accuracy is only marginally better than random (10%) and far below within-regime probing (85-100%). CCA did not find a shared invariant space.

---

## 2. Critical Methodological Issues

### 2.1 Confounded Experimental Design

| Factor | Levels | Count |
|:---|:---|:---|
| Operators | 10 | Each has 20 samples |
| Topics | 2 | Each has 100 samples |
| Paraphrases | 10 | Per operator |
| **Total samples** | **200** | = 10 × 2 × 10 |

**The Problem**: Every operator appears with both topics equally. A model that perfectly predicts topic (AI vs Climate) from content-specific hidden states will trivially partition the operator space.

### 2.2 Suspiciously Perfect Accuracy

| Regime | Test Accuracy | Test Size | Per-Class Samples |
|:---|:---|:---|:---|
| Think | 100% | 40 | ~4 per operator |
| Speak | 100% | 40 | ~4 per operator |

With only **~4 test samples per class**, 100% accuracy is:

1. Statistically fragile (95% CI includes much lower values)
2. Likely overfitting or trivial separability

The confusion matrices confirm each class has only 4 test samples.

### 2.3 Inverted "Intent Lock-in"

The intent_lockin.png figure shows:

- Listen: **Decreases** from 100% (t=5) to 88% (t=25)
- Think/Speak: Constant at 100%

This is the **opposite** of "lock-in" (which would show increasing accuracy over time as intent crystallizes). The decreasing Listen accuracy suggests:

- Early tokens carry most discriminative information (instruction phrasing)
- Later tokens (content) add noise (topic confusion)

---

## 3. What the Figures Actually Show

### Confusion Matrix - Listen Regime

Shows diagonal dominance but with off-diagonal errors:

- Poetic confused with Reframe (3→Poetic, 1→Reframe Positively)
- Reframe Positively confused (3→Reframe Negatively, 1→Reframe Positively)

This is the only regime with realistic (not 100%) accuracy.

### Confusion Matrix - Think/Speak Regimes

Perfect diagonal = all predictions correct. But with n=4 per class, this could be:

- Perfect separability OR
- Statistical accident

### Intent Lock-in Plot

- Listen (blue): Declines from 1.0 to 0.88
- Think (orange): Constant at 1.0
- Speak (green): Constant at 1.0

Think and Speak are **perfectly flat at 100%** from the first measurements—they never "lock in" because they're always perfect. This is suspicious.

---

## 4. Alternative Explanations

### Why does probing work at all?

1. **Instruction phrasing is discriminative**: "Summarize this..." vs "Write a poem..." are lexically distinct. Early layers (Listen) encode this directly.

2. **Topic content dominates later layers**: The actual text ("AI is transforming..." vs "Renewable energy...") has rich semantic content that layers 8-24 process heavily.

3. **Perfect topic accuracy = content leakage**: The model encodes topic in a way that's linearly separable. Since operators are balanced across topics, this doesn't directly help operator prediction—but confounding still contaminates the results.

### Why Think/Speak at 100%?

Possible explanations:

1. **Deeper layers represent instruction more abstractly**: Think/Speak may genuinely encode operator intent
2. **Overfitting on 4 samples per class**: With LogisticRegression on high-dim data (D=896), overfitting is likely
3. **Topic proxy**: Operators correlate with response structure that's topic-dependent

---

## 5. What Would Valid Evidence Look Like?

### Required Controls

1. **Topic-orthogonal probing**: Train on Topic A only, test on Topic B only
2. **Paraphrase-orthogonal probing**: Train on paraphrases 1-5, test on 6-10
3. **Larger test set**: At least 20 samples per class for stable estimates
4. **Topic-regressed features**: Remove topic-predictive dimensions before operator probing

### Better Metrics

1. **Cross-validated accuracy**: K-fold with proper stratification
2. **Confidence intervals**: Bootstrap to quantify uncertainty
3. **Topic-conditional accuracy**: Report accuracy separately per topic

---

## 6. Revised Interpretation

| Original Finding | Validity | Corrected Interpretation |
|:---|:---|:---|
| Listen 85% accuracy | ⚠️ Uncertain | May reflect instruction-word encoding, not operator invariants |
| Think/Speak 100% accuracy | ❌ Suspicious | Likely overfitting on n=4 per class |
| Topic leakage 100% | 🔴 Critical | Proves topic is encoded; confounds operator interpretation |
| Coupling R² < 0.07 | ✅ Valid finding | Regimes are NOT linearly coupled |
| CCA 27.5% | ✅ Valid finding | No shared invariant space exists |

---

## 7. What the Experiment Successfully Demonstrates

Despite invalidating the main claims, the experiment does show:

1. **Regimes are NOT linearly related**: The coupling map failure (R² < 0.07) is a meaningful negative result.

2. **CCA doesn't find invariants**: Another meaningful negative result.

3. **Early layers may encode instructions**: The Listen regime (layers 0-7) does distinguish operators at 85%, which is above chance even accounting for topic confounding.

4. **Methodology is sound**: The extraction, probing, and visualization pipeline works correctly.

---

## 8. Recommendations

### For Valid Operator Invariant Testing

1. **Use many more topics**: 10+ diverse topics to break topic-operator confounding
2. **Test cross-topic generalization**: Train all-topics-but-one, test on held-out topic
3. **Report topic-conditioned accuracy**: Show that operator probing works WITHIN each topic
4. **Increase sample size**: At least 500+ samples for stable 10-class classification

### For Reporting

1. **Acknowledge confounding**: The 100% topic accuracy must be prominently disclosed
2. **Qualify perfect scores**: Note the tiny test set size
3. **Highlight negative results**: Coupling/CCA failures are the most valid findings

---

## 9. Conclusion

The Regime Invariants experiment has **valid methodology** but **confounded design** that prevents meaningful interpretation of the probing results. The claim that "operators are invariant across regimes" cannot be supported because:

1. Topic information is perfectly encoded (100% accuracy)
2. Test sets are too small for reliable estimates (n≈4 per class)
3. Coupling and CCA analyses both FAIL, contradicting invariance claims

The experiment's most valid finding is actually **negative**: there is no shared invariant space linking Listen→Think→Speak regimes.
