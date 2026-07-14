# Security Incident Handbook

Incident code: SEC-INC-417

## Credential Leak Classification

This handbook describes the immediate response process for suspected credential exposure. A suspected credential leak is classified as Severity 1 because unauthorized access could affect customer data, internal systems, or privileged operational tools. Employees should treat even partial evidence of credential exposure as urgent until the Security Operations Center determines otherwise.

Employees must notify the Security Operations Center within 15 minutes of discovering a suspected leak. The emergency email address is security-urgent@example.internal. The initial notification should include the system involved, the approximate time of discovery, known affected accounts, and any related screenshots or logs that can be preserved safely.

## Containment Actions

Compromised credentials must be rotated immediately. Affected sessions must be revoked so active access tokens, browser sessions, and long-lived automation credentials cannot continue to be used. Teams should coordinate containment with the Security Operations Center before changing system state that could affect evidence collection.

Evidence must not be deleted or modified. Employees should preserve relevant logs, alert identifiers, emails, chat messages, and system snapshots. If evidence includes sensitive material, it should be stored only in approved incident-response locations.

## Communication And Review

The incident commander owns external communication approval. No employee should contact customers, vendors, regulators, or public channels about an active incident unless the incident commander has approved the message and timing.

Post-incident review occurs within five business days. The review should document root cause, response timeline, containment actions, customer impact, and prevention tasks. Error identifier AUTH-TOKEN-91 refers to repeated invalid refresh-token use and should be included in the incident record when observed.
