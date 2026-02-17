# RESPONSIBLE_USE.md

This repository is released to support **defensive security research**, **password-strength evaluation (PSM)**, and **reproducible academic experiments** on password modeling. It must **not** be used to facilitate unauthorized access, credential stuffing, or any activity that harms users or systems.

## Allowed Uses (Examples)
- Evaluating password strength meters and defensive guessing curves in **offline** settings.
- Reproducing results reported in the accompanying paper using **publicly accessible** datasets.
- Research on modeling, measurement, and mitigation of weak-password risks.

## Prohibited Uses
You agree **not** to use this repository (or any derivative work) for:
1. **Unauthorized access** to any accounts, services, networks, or devices.
2. **Online guessing**, credential stuffing, or any form of attack against live systems.
3. Generating or distributing **high-throughput cracking pipelines**, large-scale guess lists, or automation intended to increase attack capability.
4. Collecting, retaining, disclosing, or redistributing **personally identifiable information (PII)** (e.g., emails), or attempting **re-identification** of individuals.
5. Redistributing breached/leaked password datasets or any data obtained from them, except where explicitly permitted by the dataset’s original terms and applicable laws.

## Sensitive Data Handling Expectations
If you work with breached password datasets, you must:
- Use data only in **controlled, offline** environments.
- Apply **data minimization**: process only what is necessary and avoid retaining raw identifiers.
- Protect data with **access controls** (and encryption at rest where feasible).
- Avoid releasing examples that could reveal real credentials; report only **aggregate statistics**.
- Follow your institution’s policies and applicable laws/regulations.

## No Dataset Redistribution
This repository does **not** include breached datasets. Users are responsible for obtaining datasets through lawful and ethically appropriate means. Do **not** upload or share datasets (or derived sensitive artifacts) in issues, pull requests, or public forks.

## No Warranty
This software is provided “AS IS”, without warranty of any kind. The authors and contributors are not responsible for misuse, damages, or legal consequences resulting from prohibited use.

## Contact / Reporting
If you believe this repository is being misused, or if you discover security/privacy issues in the code or documentation, please open a responsible disclosure report via the project’s preferred channel.
