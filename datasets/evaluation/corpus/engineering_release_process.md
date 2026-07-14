# Engineering Release Process

Process identifier: ENG-REL-302

## Standard Release Window

This process defines how engineering teams prepare, approve, and execute production releases. Production releases normally occur Wednesday between 2:00 PM and 4:00 PM IST. The standard window gives support, operations, and product stakeholders predictable coverage during deployment activity.

Release owners should verify scope, customer impact, observability dashboards, and operational readiness before requesting approval. A rollback plan is mandatory for every production release, including releases that appear low risk.

## Review And Approval Requirements

Two approving reviewers are required before a production release can proceed. One reviewer must be from outside the author's immediate team to provide independent review of risk, testing evidence, and operational readiness.

Database migrations require a verified backup. The release owner must confirm that backup completion, migration testing, and rollback steps are recorded before the deployment begins. Deployment error code DEPLOY-GATE-73 means required approval evidence is missing.

## Emergency And Automated Controls

Emergency releases require approval from the on-call engineering manager. The emergency path should be used only when delaying the fix would create unacceptable customer, security, or operational risk.

Failed health checks trigger automatic rollback. Release automation should stop rollout, restore the prior healthy version when possible, and notify the release owner and on-call responders. Release records must be retained for 180 days for auditability, incident review, and operational trend analysis.
