"""Deterministic, idempotent database migrations and seed tooling."""

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from alembic.config import Config
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from alembic import command
from vehicle_risk_agent.config import Settings
from vehicle_risk_agent.persistence.models import (
    Base,
    PolicyCorpusRecord,
    PolicyCorpusSnapshotRecord,
    PolicyPassageRecord,
    PolicySnapshotRecord,
    PolicySourceRecord,
    RiskPolicyRecord,
)
from vehicle_risk_agent.risk.models import (
    create_default_policy_v1,
)

logger = logging.getLogger(__name__)


def run_migrations(database_url: str) -> None:
    """Run Alembic database migrations up to head."""
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    alembic_ini = root_dir / "alembic.ini"
    if alembic_ini.exists():
        cfg = Config(str(alembic_ini))
        cfg.set_main_option("sqlalchemy.url", database_url)
        command.upgrade(cfg, "head")


async def seed_database(
    database_url: str | None = None,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Apply migrations and seed active Risk Policy and Policy Corpus idempotently."""
    if settings is None:
        settings = Settings()
    db_url = database_url or settings.database_url

    # 1. Run migrations and ensure tables exist
    try:
        await asyncio.to_thread(run_migrations, db_url)
    except Exception as exc:
        logger.warning("Alembic upgrade through config failed: %s", exc)

    engine_bootstrap = create_async_engine(db_url, echo=False)
    async with engine_bootstrap.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    await engine_bootstrap.dispose()

    engine = create_async_engine(db_url, echo=False)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    seeded_risk_policy_id = "nz-risk-policy-v1"
    seeded_corpus_id = "nz-policy-corpus-v1"

    async with session_factory() as session:
        # 2. Seed Risk Policy if no active policy exists
        active_rp_stmt = select(RiskPolicyRecord).where(
            RiskPolicyRecord.lifecycle_state == "ACTIVE"
        )
        active_rp = (await session.execute(active_rp_stmt)).scalar_one_or_none()

        if active_rp is None:
            # Check if record exists in draft/ready
            existing_stmt = select(RiskPolicyRecord).where(
                RiskPolicyRecord.id == seeded_risk_policy_id
            )
            rp_record = (await session.execute(existing_stmt)).scalar_one_or_none()

            default_policy = create_default_policy_v1(policy_id=seeded_risk_policy_id)
            if rp_record is None:
                rp_record = RiskPolicyRecord(
                    id=default_policy.id,
                    version=default_policy.version,
                    name=default_policy.name,
                    description=default_policy.description,
                    lifecycle_state="ACTIVE",
                    factor_weights_json=json.dumps(
                        {k.value: v for k, v in default_policy.factor_weights.items()},
                        sort_keys=True,
                    ),
                    score_cap=default_policy.score_cap,
                    risk_bands_json=json.dumps(
                        [
                            {
                                "band": b.band.value,
                                "min_score": b.min_score,
                                "max_score": b.max_score,
                                "description": b.description,
                            }
                            for b in default_policy.risk_bands
                        ],
                        sort_keys=True,
                    ),
                    mandatory_review_rules_json=json.dumps(
                        list(default_policy.mandatory_review_rules), sort_keys=True
                    ),
                    required_evidence_fields_json=json.dumps(
                        list(default_policy.required_evidence_fields), sort_keys=True
                    ),
                    policy_hash=default_policy.policy_hash,
                    activated_by="system-seed",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                    activated_at=datetime.now(UTC),
                )
                session.add(rp_record)
            else:
                rp_record.lifecycle_state = "ACTIVE"
                rp_record.activated_at = datetime.now(UTC)
                rp_record.activated_by = "system-seed"

            await session.commit()

        # 3. Seed Policy Sources, Snapshots, Passages, and Corpus
        active_pc_stmt = select(PolicyCorpusRecord).where(
            PolicyCorpusRecord.lifecycle_state == "ACTIVE"
        )
        active_pc = (await session.execute(active_pc_stmt)).scalar_one_or_none()

        if active_pc is None:
            # Policy Source: FTA 1986
            fta_source_id = "nz-fta-1986"
            fta_source = (
                await session.execute(
                    select(PolicySourceRecord).where(PolicySourceRecord.id == fta_source_id)
                )
            ).scalar_one_or_none()

            if fta_source is None:
                fta_source = PolicySourceRecord(
                    id=fta_source_id,
                    title="Fair Trading Act 1986",
                    issuing_authority="Parliament of New Zealand",
                    jurisdiction="NZ",
                    canonical_origin="https://www.consumerprotection.govt.nz/general-help/consumer-laws/fair-trading-act",
                    authority_classification="PRIMARY_LEGISLATION",
                    reuse_terms="CC BY 4.0",
                    expected_update_cadence="ADHOC",
                    status="ACTIVE",
                    created_at=datetime.now(UTC),
                    updated_at=datetime.now(UTC),
                )
                session.add(fta_source)
                await session.flush()

            # Snapshot for FTA
            fta_content = (
                "# Fair Trading Act 1986\n"
                "## Section 9: Misleading and deceptive conduct\n"
                "No person shall, in trade, engage in conduct that is misleading or deceptive "
                "or is likely to mislead or deceive.\n"
                "## Section 13: False or misleading representations\n"
                "No person shall, in trade, in connection with the supply or possible supply "
                "of goods, make false or misleading representations concerning quality or status."
            )
            fta_content_hash = hashlib.sha256(fta_content.encode("utf-8")).hexdigest()
            fta_snap_id = f"snap-fta-{fta_content_hash[:16]}"

            fta_snap = (
                await session.execute(
                    select(PolicySnapshotRecord).where(PolicySnapshotRecord.id == fta_snap_id)
                )
            ).scalar_one_or_none()

            if fta_snap is None:
                fta_snap = PolicySnapshotRecord(
                    id=fta_snap_id,
                    source_id=fta_source_id,
                    content_hash=fta_content_hash,
                    raw_content=fta_content,
                    parser_version="policy-parser-v1",
                    validation_outcome="VALID",
                    metadata_json="{}",
                    created_at=datetime.now(UTC),
                    retrieved_at=datetime.now(UTC),
                )
                session.add(fta_snap)
                await session.flush()

                # Add passages
                p1_text = (
                    "No person shall, in trade, engage in conduct that is misleading or deceptive "
                    "or is likely to mislead or deceive."
                )
                p2_text = (
                    "No person shall, in trade, in connection with the supply or possible supply "
                    "of goods, make false or misleading representations concerning quality."
                )

                dummy_emb = [0.0] * 384
                passages = [
                    PolicyPassageRecord(
                        id=f"{fta_snap_id}-p1",
                        snapshot_id=fta_snap_id,
                        source_id=fta_source_id,
                        section_identifier="Section 9",
                        heading="Misleading and deceptive conduct",
                        text=p1_text,
                        sequence=1,
                        char_offset_start=0,
                        char_offset_end=len(p1_text),
                        content_hash=hashlib.sha256(p1_text.encode("utf-8")).hexdigest(),
                        embedding=dummy_emb,
                    ),
                    PolicyPassageRecord(
                        id=f"{fta_snap_id}-p2",
                        snapshot_id=fta_snap_id,
                        source_id=fta_source_id,
                        section_identifier="Section 13",
                        heading="False or misleading representations",
                        text=p2_text,
                        sequence=2,
                        char_offset_start=len(p1_text),
                        char_offset_end=len(p1_text) + len(p2_text),
                        content_hash=hashlib.sha256(p2_text.encode("utf-8")).hexdigest(),
                        embedding=dummy_emb,
                    ),
                ]
                session.add_all(passages)
                await session.flush()

            # Create Corpus
            corpus_record = (
                await session.execute(
                    select(PolicyCorpusRecord).where(PolicyCorpusRecord.id == seeded_corpus_id)
                )
            ).scalar_one_or_none()

            manifest_content = f"{seeded_corpus_id}:{fta_snap_id}"
            manifest_hash = hashlib.sha256(manifest_content.encode("utf-8")).hexdigest()

            if corpus_record is None:
                corpus_record = PolicyCorpusRecord(
                    id=seeded_corpus_id,
                    name="New Zealand Consumer Protection Policy Corpus v1",
                    description="Authoritative baseline corpus covering Fair Trading Act.",
                    lifecycle_state="ACTIVE",
                    retrieval_config_json=json.dumps(
                        {"vector_weight": 0.5, "keyword_weight": 0.5, "top_k": 5}
                    ),
                    manifest_hash=manifest_hash,
                    created_at=datetime.now(UTC),
                    activated_at=datetime.now(UTC),
                    activated_by="system-seed",
                )
                session.add(corpus_record)
                await session.flush()

                assoc = PolicyCorpusSnapshotRecord(
                    corpus_id=seeded_corpus_id,
                    snapshot_id=fta_snap_id,
                )
                session.add(assoc)
            else:
                corpus_record.lifecycle_state = "ACTIVE"
                corpus_record.activated_at = datetime.now(UTC)
                corpus_record.activated_by = "system-seed"

            await session.commit()

    await engine.dispose()
    return {
        "status": "seeded",
        "risk_policy_id": seeded_risk_policy_id,
        "corpus_id": seeded_corpus_id,
    }


def main() -> None:
    """CLI entrypoint for seed command."""
    parser = argparse.ArgumentParser(description="Seed vehicle risk agent database")
    parser.add_argument("--database-url", type=str, default=None, help="Database connection URL")
    args = parser.parse_args()

    result = asyncio.run(seed_database(database_url=args.database_url))
    print(f"Seeding completed successfully: {result}")
    sys.exit(0)


if __name__ == "__main__":
    main()
