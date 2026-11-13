# Northstar database backup policy

This is fictional sample knowledge for the RAG Lab.

Production PostgreSQL databases receive a full backup every day at 02:00 UTC. Transaction logs are archived continuously to support point-in-time recovery. Daily backups are retained for 30 days and monthly backups are retained for 12 months.

To restore accidentally deleted customer records, open a restore request with the database operations team. The database operations lead must approve a production restore. Restore into an isolated database first, verify the requested records, and agree on the recovery plan before changing production.

The recovery point objective is 15 minutes. The recovery time objective is four hours. These are operational targets rather than promises that every incident will meet them. Test a full restore once each quarter and document its duration and validation results.
