---
name: ascend-310p-accuracy-troubleshooting
description: Diagnose, reproduce, bisect, and verify vLLM/vLLM-Ascend inference accuracy regressions on Ascend 310P. Use for environment setup, ais_bench or curl reproduction, feature-isolation experiments, and commit/PR-level root-cause analysis; do not use as a general performance-tuning guide.
---

# Ascend 310P Accuracy Troubleshooting

Build an evidence-backed path from the reported accuracy symptom to the smallest reproducible configuration and, when requested, the introducing commit or code path. Keep accuracy diagnosis separate from performance tuning unless the user explicitly asks for both.

## Route the work

- For installing or verifying vLLM, vLLM-Ascend, CANN, and 310P-specific prerequisites, read [references/deployment.md](references/deployment.md).
- For launching the service and reproducing with curl or ais_bench, read [references/accuracy-testing.md](references/accuracy-testing.md).
- For narrowing features, comparing good and bad baselines, bisecting commits, and closing the regression, read [references/triage-workflow.md](references/triage-workflow.md).
- Read more than one reference only when the task spans those stages.

## Operating rules

1. Ask whether a compatible vLLM/vLLM-Ascend diagnosis environment already exists. If it does, obtain the vLLM-Ascend source path and active branch before proceeding, and run service launch and source-level debugging from that path. If it does not, inspect the container's installed versions before changing it; ask which vLLM-Ascend branch is required because it may not be `main`. Before launching a service, honor any user-specified NPU; otherwise inspect device health, active processes, available memory, and TP requirements, then select suitable idle device(s) with a process-scoped `ASCEND_RT_VISIBLE_DEVICES` value.
2. Start with evidence. Obtain the last-known-good and first-known-bad vLLM and vLLM-Ascend versions or commit IDs, the complete launch command, model/weight identity, CANN, PTA/`torch_npu`, firmware, and driver information, hardware topology, dataset or minimal curl, expected and actual outputs, and generation/evaluation settings.
3. Reproduce before modifying code. First preserve the reported setup, then build a graduated curl suite: a structured factual answer, a longer deterministic prompt/output, and—when relevant—an actual failed dataset sample with image/input payload. Exercise enough decode tokens, prefix reuse, repeated requests, batch transitions, or concurrency to cover the reported feature path. A trivial one-token answer is only a health check. If the first curl does not reproduce, expand the minimal suite or run a small targeted dataset slice; do not jump directly to a costly full evaluation. Use a stable distinguishing case as the primary diagnosis and fix-acceptance criterion.
4. Before broad experimental localization, inspect the code changes between the supplied dates or commits and reason about which 310P paths could affect the symptom. Turn that analysis into explicit hypotheses and discriminating tests; static inspection prioritizes experiments but does not prove root cause.
5. Compare one variable at a time. Isolate graph mode, quantization, MTP, prefix cache, tensor parallelism, concurrency, thinking mode, post-processing, and related features rather than changing a bundle of flags.
6. Treat branches, versions, and command-line flags as version-dependent. Verify the active branch, checked-out source, package metadata, and local `--help` before issuing commands. Do not assume `main` or silently upgrade dependencies while investigating a regression.
7. Keep service and evaluator settings aligned: model name, host, port, output length, temperature, EOS handling, chat template/thinking mode, batch size, and dataset revision.
8. Separate model-quality changes from serving defects. Preserve raw responses and evaluator artifacts; confirm suspicious scores with a minimal request and compare token/output behavior before attributing the issue to an operator or kernel.
9. If commit bisect is infeasible or cannot produce a reliable boundary, compare tensor dumps from controlled good and bad runs. Narrow from model-stage or layer boundaries to the first divergent layer, then to operator inputs and outputs. Align execution coordinates and use dtype-appropriate error metrics; do not treat any nonzero floating-point difference as root cause.
10. After identifying a feature, tensor divergence, or commit, explain the evidence and likely code path. Implement a fix only when the user asks. Verify any fix first against the graduated minimal reproducer and targeted failing samples. After those pass, run the original full accuracy suite with the documented evaluator command and original service/evaluator settings; do not report the fix complete without that full regression or an explicit external blocker.
11. Once the user authorizes network access and diagnosis-environment changes for the named server/container, treat that as standing authorization for in-scope clone/fetch, clean checkout/worktree switching, submodule updates, and installation of vLLM, vLLM-Ascend, ais_bench, and their ordinary Python/build dependencies. Execute these steps without repeated conversational confirmation. Preserve dirty trees by using a clean worktree or separate clone, record every commit and install action, and do not extend this authorization to driver, firmware, CANN, PTA/`torch_npu`, destructive cleanup, or unrelated system packages.

## Safety and reporting

- Never automate bulk deletion of build trees, installed toolchains, CANN/NNAL directories, datasets, or user outputs. If cleanup is required, list the exact paths and ask the user to remove them manually or explicitly authorize individual-file removal.
- Never update, downgrade, reinstall, or uninstall the Ascend driver as part of this workflow, even if driver differences are suspected. Record the evidence and report the driver dependency as an unresolved external constraint.
- Update, downgrade, reinstall, or uninstall CANN or PTA/`torch_npu` only after controlled evidence localizes the problem to that component and the user explicitly confirms the proposed change. Before requesting confirmation, state the current and target versions, supporting evidence, affected environment, risks, rollback plan, and regression test plan.
- When a container's installed vLLM/vLLM-Ascend pair is unusable for the requested branch, explain the incompatibility and install the matched pair under the standing diagnosis-environment authorization from rule 11; if that authorization has not been given, request it once for the whole scoped workflow. Do not uninstall merely because versions differ. Record rollback information before replacing the pair. Do not replace firmware without explicit authorization. Prefer a disposable container or isolated environment for comparisons.
- Do not claim root cause from score movement alone. Report the reproduction matrix, controlled variable, artifacts, confidence, remaining uncertainty, and the next discriminating experiment.

## Deliverable

Provide a compact diagnosis containing:

- environment and version fingerprint;
- exact reproducer and expected/actual result;
- good/bad comparison matrix;
- isolated feature, commit, PR, first divergent tensor/operator, or code path with supporting evidence;
- fix status and regression results, if a fix was requested;
- unresolved risks and the next test when root cause is not yet proven.
