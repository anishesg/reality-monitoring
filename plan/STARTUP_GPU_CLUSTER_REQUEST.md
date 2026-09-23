# Startup GPU Cluster: what it is, how to get it, and the $2 credit-only test (V, 2026-09-23)

## What the primary sources say (learn.microsoft.com, read 2026-09-23)
- The Startup GPU Cluster is real: ND-H100 (1-13 VMs) and NDm-A100 (1-62 VMs, 8 GPUs each) reserved for short-term startup use, max 60-day
  reservation, deploy within 48 h of approval or the quota may be revoked, spin down at the end date.
- Credits: "Microsoft for Startups credits can be used to pay for it. Your Azure Sponsorship credits are deducted to cover these costs.
  **Once your charges surpass your sponsorship balance, you are responsible for the additional cost.**" (cluster page, verbatim)
- Access is invitation-driven: "Eligible startups receive weekly invitations to request access to the cluster." The request form is only in that
  email. No public form URL. Escalation: Microsoft for Startups portal (https://aka.ms/startuphelp-mfs-portal) -> support ticket, or the
  program representative. One asker with a $5,000 sponsorship never received an invitation and was routed to an Azure support ticket.
- Nothing I can do from the CLI reaches this: it is a Founders Hub action, not an Azure API. Every GPU family quota on eae276d8 is 0 in every
  region (re-verified tonight), so until either the cluster invitation or a quota grant lands there is no GPU to test.

## What to send (paste into the Founders Hub support ticket / reply to the invitation)
Subject: Startup GPU Cluster access (or NC A100 v4 / ND H100 v5 quota) for an ICLR submission, sponsorship subscription eae276d8-7251-4b9f-a24f-a69393830747

We are a Founders Hub startup (Ablation Labs) running an LLM post-training study for an ICLR 2027 submission due 2026-09-25 AoE. We need
short-term interconnected GPU access billed to our Azure Sponsorship credits (current balance ~$9.8k of $10k).
- Workload: DPO and GRPO fine-tuning of 7-8B open models (OLMo-2-7B-SFT, Tulu-3-8B-SFT), 3 seeds x 4 arms, plus vLLM evaluation of 7B-72B models.
- Request: ND H100 v5 (1 VM, 8x H100) for 7 days, or NC A100 v4 (Standard_NC96ads_A100_v4, 4x A100 80GB) quota of 96 vCPUs for 14 days.
- Regions in order: East US 2, South Central US, West US 3.
- Estimated spend: 150-250 GPU-hours, about $1,000 in sponsorship credits; we will spin down at the end date.
- Every quota request on this subscription (17 so far, all NC/ND families, all regions) has been auto-rejected with limit 0; Azure support says
  the subscription cannot be granted GPU quota. Please either invite us to the Startup GPU Cluster or tag the subscription for GPU quota.

## The $2 credit-only test (run the moment any GPU capacity appears)
1. `az vm create` the smallest granted GPU size (e.g. Standard_NC24ads_A100_v4 or the cluster's smallest ND) with `--priority Regular`, run
   `nvidia-smi`, deallocate after <= 10 minutes (about $0.60-$0.80 at list price).
2. Next day: `az costmanagement query` grouped by PublisherType and MeterCategory. The charge must appear with PublisherType = Microsoft
   under "Virtual Machines" on subscription eae276d8. The Sponsorship subscription has no payment instrument: when the credit is exhausted
   the subscription is disabled, not billed (quotaId Sponsored_2016-01-01; billing account is MCA-Individual but the sponsorship offer does
   not invoice). Marketplace SaaS resources on the subscription: 0 (checked).
3. Only then: `slurm/submit_everything.sh` equivalents from `azure/queue_ladder.sh` (rewritten for the granted SKU), with the existing kill
   switch and a $1,000 budget alert, wall-clock limits DPO 4 h / GRPO 12 h / eval 2 h, checkpoint resume.
