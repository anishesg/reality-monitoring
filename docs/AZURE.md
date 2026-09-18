# Azure compute for this project (state as of 2026-09-18)

## The money
- Subscription: "Azure subscription 1", id `eae276d8-7251-4b9f-a24f-a69393830747`, owner vkakaria11@gmail.com.
- Offer: Azure Sponsorship (`quotaId Sponsored_2016-01-01`) on a Microsoft Customer Agreement (individual).
- Credit lot: **USD 10,000.00**, source "Azure startup sponsorship credit", start 2026-07-27, **expires 2028-07-27**. Spent to date: $0.00.
- Billing profile `JT34-BWUL-BG7-PGB`. A Visa (exp 10/2029) is attached and the spending limit is OFF, which means Azure *would* bill the card after credits run out. Mitigations below.

Where to see the balance in the portal: **Cost Management + Billing → Billing scopes → "Vikram Kakaria" (billing account) → Billing profiles → JT34-BWUL-BG7-PGB → Credits** (left menu). Direct: portal.azure.com → search "Cost Management + Billing" → "Credits". The Founders Hub site (foundershub.startups.microsoft.com → Benefits → Azure) shows the same number.

## Zero-card-charge guardrails (in place)
1. Budget `credit-lifetime-10k` (annual, $10,000): email at 25/50/75%; **kill switch fires at 85% and 95%**.
2. (monthly guard removed 2026-09-18 at user request; only the lifetime budget remains)
3. Kill switch = Automation account `rm-infra/rm-killswitch`, runbook `StopAllVMs` (deallocates every VM and VM scale set in the subscription via managed identity with Virtual Machine Contributor). Action group `rm-killswitch-ag`. Test-fired 2026-09-18 07:11 UTC: authenticated and completed.
4. Rule: only first-party Azure compute/storage/network. **No Marketplace/third-party offers** (those are not covered by sponsorship credit and bill the card).
5. Budget evaluation lags actual usage by up to 24h, so the 85% trigger ≈ $8,500 leaves ~$1,500 headroom. Planned project spend is ≈ $300–600.
6. Manual stop before $10k: portal → Subscriptions → Cancel subscription. Do this if balance < $500.

## GPU quota (the blocker)
All GPU families are at **0 vCPUs** in every region. Spot ("low priority") cores are at 3. Automatic Quota-API requests for 96/48/24 A100 vCPUs in eastus2 and westus3 were all rejected; the support-ticket API is closed to the free Developer support plan. **Request via the portal (free):**

1. portal.azure.com → search **Quotas** → **Compute** → Region = **East US 2** → filter "NCADS_A100_v4" → select **Standard NCADSA100v4 Family vCPUs** → **New Quota Request** → **96** → submit. Repeat for **West US 3 → 48** as fallback.
2. Also request **Low priority cores** (spot) → 96 in East US 2 (spot A100 is $0.68/hr vs $3.67/hr and fine for evals).
3. If the portal quota page rejects it, use **Help + support → Create a support request → Issue type "Service and subscription limits (quotas)" → Compute-VM (cores-vCPUs)**. Quota tickets are free on every plan when opened in the portal.
Justification text to paste: "LLM post-training research (DPO/GRPO fine-tuning of 7–8B open models with vLLM evaluation), short bursts totalling ~100 GPU-hours over two weeks, funded by Azure startup sponsorship credit."

## What it costs (eastus2 list price, Linux)
| SKU | GPUs | On-demand | Spot |
|---|---|---|---|
| NC24ads_A100_v4 | 1× A100 80GB | $3.67/h | $0.68/h |
| NC48ads_A100_v4 | 2× A100 80GB | $7.35/h | $1.36/h |
| NC96ads_A100_v4 | 4× A100 80GB | $14.69/h | $2.72/h |
| NC40ads_H100_v5 | 1× H100 94GB | $6.98/h | $1.29/h (not offered in the regions checked) |
| NC4as_T4_v3 | 1× T4 16GB | $0.53/h | $0.15/h (smoke tests only) |

Whole ladder (2 backbones × 6 arms + evals) ≈ 60–100 A100-hours ≈ **$250–400 on-demand**, well under 5% of the credit.

## Resources that exist
- Resource group `rm-infra` (eastus2): Automation account `rm-killswitch` (Free tier), action group `rm-killswitch-ag`. Cost: $0.
- Registered providers: Compute, Network, Storage, MachineLearningServices, ContainerRegistry, KeyVault, Insights, Automation, Quota.
