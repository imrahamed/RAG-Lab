# Northstar access management

This is fictional sample knowledge for the RAG Lab.

New employees request application access through the internal service desk. Their manager approves the business need and the application owner approves the role. Production access requires multifactor authentication and a named user account.

Temporary elevated privileges expire after eight hours. A new approval is needed for an extension. Shared administrator passwords are prohibited. Service credentials must be stored in the approved secrets manager, not committed to a source repository.

When an employee leaves, the identity team disables interactive access by the end of the employee's final working day. Application owners review access lists every quarter and remove permissions that no longer match a user's duties.
