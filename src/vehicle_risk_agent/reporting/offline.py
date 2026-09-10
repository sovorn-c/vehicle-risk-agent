"""Pure deterministic offline report drafting adapter."""

# story: e04s02

from __future__ import annotations

from vehicle_risk_agent.reporting.models import (
    AbstentionNotice,
    ClaimReference,
    ContributingFactorsSection,
    EvidenceSummarySection,
    ExecutiveSummarySection,
    LimitationsSection,
    MandatoryReviewSection,
    MissingEvidenceNotice,
    PolicyCitationsSection,
    ReportDraft,
    ReportDraftStatus,
    ReportSections,
    RiskScoreSection,
    SyntheticNotice,
    SyntheticNoticeSection,
    VehicleIdentitySection,
    compute_draft_hash,
)
from vehicle_risk_agent.reporting.protocol import (
    ReportDraftingContext,
    ReportDraftingProtocol,
)
from vehicle_risk_agent.risk.models import AssessmentOutcome, RiskBand, RiskFactor


class OfflineReportDraftingAdapter(ReportDraftingProtocol):
    """Deterministic, offline report drafting adapter generating reviewable cited drafts."""

    def __init__(self, drafter_version: str = "offline-v1") -> None:
        self.drafter_version = drafter_version

    async def draft_report(self, context: ReportDraftingContext) -> ReportDraft:
        """Draft a 9-section cited report deterministically without network or LLM dependencies."""
        return self._draft_report_sync(context)

    def _draft_report_sync(self, context: ReportDraftingContext) -> ReportDraft:
        """Synchronous drafting implementation."""
        risk_result = context.risk_result
        is_incomplete = risk_result.is_incomplete
        outcome = AssessmentOutcome.INCOMPLETE if is_incomplete else AssessmentOutcome.SCORED

        # 1. Executive Summary
        s1 = self._build_executive_summary(context)

        # 2. Vehicle Identity
        s2 = self._build_vehicle_identity(context)

        # 3. Risk Score & Band
        s3 = self._build_risk_score_and_band(context)

        # 4. Mandatory Review Findings
        s4 = self._build_mandatory_findings(context)

        # 5. Contributing Factors
        s5 = self._build_contributing_factors(context)

        # 6. Policy Citations
        s6 = self._build_policy_citations(context)

        # 7. Evidence Summary
        s7 = self._build_evidence_summary(context)

        # 8. Limitations & Missing Evidence
        s8 = self._build_limitations_and_missing_evidence(context)

        # 9. Synthetic Data Notice
        s9 = self._build_synthetic_notice(context)

        sections = ReportSections(
            executive_summary=s1,
            vehicle_identity=s2,
            risk_score_and_band=s3,
            mandatory_review_findings=s4,
            contributing_factors=s5,
            policy_citations=s6,
            evidence_summary=s7,
            limitations_and_missing_evidence=s8,
            synthetic_data_notice=s9,
        )

        draft_hash = compute_draft_hash(
            assessment_id=context.assessment_id,
            run_number=context.run_number,
            outcome=outcome,
            risk_result_id=risk_result.id,
            sections=sections,
        )

        draft_id = f"draft-{context.assessment_id}-{context.run_number}"

        return ReportDraft(
            id=draft_id,
            assessment_id=context.assessment_id,
            run_number=context.run_number,
            risk_result_id=risk_result.id,
            status=ReportDraftStatus.DRAFT,
            outcome=outcome,
            sections=sections,
            draft_hash=draft_hash,
            metadata={"drafter_id": self.drafter_version, "mode": "offline", **context.metadata},
        )

    # -------------------------------------------------------------------------
    # Section Builders
    # -------------------------------------------------------------------------

    def _build_executive_summary(self, context: ReportDraftingContext) -> ExecutiveSummarySection:
        rr = context.risk_result
        aid = context.assessment_id
        rnum = context.run_number
        vin = context.vin or context.vehicle_id
        make = context.get_canonical_field("make", "Unknown")
        model = context.get_canonical_field("model", "Unknown")
        year = context.get_canonical_field("year", "Unknown")
        claims: list[ClaimReference] = []
        ev_refs: set[str] = set()
        pol_refs: set[str] = set()
        rf_refs: set[str] = set()

        if rr.is_incomplete:
            missing_list = rr.missing_evidence or rr.missing_findings or ()
            missing_count = len(missing_list)
            missing_names = ", ".join(mf.field_name for mf in missing_list) or "required fields"
            summary_text = (
                f"# Executive Summary\n\n"
                f"⚠️ **ASSESSMENT INCOMPLETE — SCORING WITHHELD**\n\n"
                f"- **Assessment Outcome:** INCOMPLETE\n"
                f"- **Subject Vehicle:** {year} {make} {model} (VIN: {vin})\n"
                f"- **Risk Score:** Withheld (INCOMPLETE)\n"
                f"- **Risk Band:** N/A (Withheld)\n"
                f"- **Missing Required Evidence:** {missing_count} field(s) ({missing_names})\n\n"
                f"## Abstention Narrative\n"
                f"Under New Zealand Vehicle Risk Policy {rr.policy_version} invariants, "
                "numerical risk scoring and risk band classification are strictly withheld "
                "because required evidence fields could not be verified from upstream "
                "registers.\n\n"
                f"## Reviewer Guidance\n"
                f"The assessment cannot proceed to scored approval until the missing evidence is "
                f"obtained and resolved. A reviewer must either request reinvestigation or reject "
                f"the report draft. Partial evidence collected during this run is documented below."
            )
            claims.append(
                ClaimReference(
                    claim_id=f"claim-{aid}-{rnum}-exec-incomplete",
                    statement=(
                        f"Scoring withheld due to {missing_count} unverified "
                        f"field(s): {missing_names}"
                    ),
                    evidence_refs=tuple(sorted(ev_refs)),
                )
            )
            return ExecutiveSummarySection(
                summary_text=summary_text,
                outcome=AssessmentOutcome.INCOMPLETE,
                key_findings=(),
                recommendation="Withheld pending missing evidence",
                claims=tuple(claims),
                evidence_refs=tuple(sorted(ev_refs)),
                policy_citation_refs=(),
                risk_factor_refs=(),
            )

        # SCORED Executive Summary
        score = rr.score or 0
        band = rr.band or RiskBand.LOW
        findings_count = len(rr.findings)
        key_findings: list[str] = []

        if findings_count > 0:
            for idx, f in enumerate(rr.findings, 1):
                finding_str = (
                    f"{f.title} (+{f.weight} pts, {f.severity.value if f.severity else 'MEDIUM'})"
                )
                key_findings.append(finding_str)
                ev_refs.update(f.evidence_refs)
                pol_refs.update(f.policy_citation_refs)
                rf_refs.add(f.factor.value if isinstance(f.factor, RiskFactor) else str(f.factor))
                claims.append(
                    ClaimReference(
                        claim_id=f"claim-{aid}-{rnum}-exec-{idx}",
                        statement=f"{f.title}: {f.description}",
                        evidence_refs=f.evidence_refs,
                        policy_citation_refs=f.policy_citation_refs,
                        risk_factor_refs=(
                            f.factor.value if isinstance(f.factor, RiskFactor) else str(f.factor),
                        ),
                    )
                )
        else:
            key_findings.append(
                "All automated vehicle register checks (PPSR security interest, "
                "stolen vehicle register, write-off status) returned clear/negative results. "
                "No adverse records were detected."
            )

        summary_text = (
            f"# Executive Summary\n\n"
            f"- **Assessment Outcome:** SCORED\n"
            f"- **Subject Vehicle:** {year} {make} {model} (VIN: {vin})\n"
            f"- **Risk Score:** {score} / 100 ({band.value} Risk Band)\n"
            f"- **Policy Evaluated:** {rr.policy_id} (Version {rr.policy_version})\n"
            f"- **Mandatory Review Findings:** {findings_count} finding(s) triggered\n\n"
            f"## Key Findings Summary\n"
            + "\n".join(f"- {kf}" for kf in key_findings)
            + "\n\n## Reviewer Guidance\n"
            "This report requires human reviewer evaluation prior to report release. "
            "An approved report confirms register evidence verification and does not "
            "constitute a physical vehicle condition inspection or purchase endorsement."
        )

        recommendation = (
            "Manual review required prior to release" if findings_count > 0 else "Standard review"
        )

        return ExecutiveSummarySection(
            summary_text=summary_text,
            outcome=AssessmentOutcome.SCORED,
            key_findings=tuple(key_findings),
            recommendation=recommendation,
            claims=tuple(claims),
            evidence_refs=tuple(sorted(ev_refs)),
            policy_citation_refs=tuple(sorted(pol_refs)),
            risk_factor_refs=tuple(sorted(rf_refs)),
        )

    def _build_vehicle_identity(self, context: ReportDraftingContext) -> VehicleIdentitySection:
        aid = context.assessment_id
        rnum = context.run_number
        vin = context.vin or context.vehicle_id
        make = context.get_canonical_field("make")
        model = context.get_canonical_field("model")
        year_raw = context.get_canonical_field("year")
        year = int(year_raw) if year_raw is not None and str(year_raw).isdigit() else None
        plate = context.get_canonical_field("plate")
        body_style = context.get_canonical_field("body_style")
        color = context.get_canonical_field("color")
        engine_num = context.get_canonical_field("engine_number")
        chassis_num = context.get_canonical_field("chassis_number")

        conf_score = None
        conf_band = None
        if context.evidence_snapshot is not None:
            conf_score = context.evidence_snapshot.confidence.score
            conf_band = context.evidence_snapshot.confidence.band

        ev_refs: set[str] = set()
        for fld in ("vin", "make", "model", "year", "plate"):
            ev_refs.update(context.get_evidence_refs_for_field(fld))

        claims = (
            ClaimReference(
                claim_id=f"claim-{aid}-{rnum}-ident",
                statement=(
                    f"Vehicle identified as {year or ''} {make or ''} {model or ''} (VIN: {vin})"
                ),
                evidence_refs=tuple(sorted(ev_refs)),
            ),
        )

        return VehicleIdentitySection(
            vin=vin,
            make=str(make) if make is not None else None,
            model=str(model) if model is not None else None,
            year=year,
            plate=str(plate) if plate is not None else None,
            body_style=str(body_style) if body_style is not None else None,
            color=str(color) if color is not None else None,
            engine_number=str(engine_num) if engine_num is not None else None,
            chassis_number=str(chassis_num) if chassis_num is not None else None,
            confidence_score=conf_score,
            confidence_band=conf_band,
            identity_verified=True,
            claims=claims,
            evidence_refs=tuple(sorted(ev_refs)),
        )

    def _build_risk_score_and_band(self, context: ReportDraftingContext) -> RiskScoreSection:
        rr = context.risk_result
        aid = context.assessment_id
        rnum = context.run_number
        if rr.is_incomplete:
            return RiskScoreSection(
                score=None,
                band=None,
                raw_score=None,
                score_cap=100,
                policy_id=rr.policy_id,
                policy_version=rr.policy_version,
                calculation_hash=rr.calculation_hash,
                is_incomplete=True,
                scoring_withheld_reason="Missing required evidence fields",
                claims=(
                    ClaimReference(
                        claim_id=f"claim-{aid}-{rnum}-risk-score-withheld",
                        statement=(
                            "Numerical risk score withheld under policy incomplete evidence rule"
                        ),
                        risk_factor_refs=(),
                    ),
                ),
            )

        return RiskScoreSection(
            score=rr.score,
            band=rr.band,
            raw_score=rr.raw_score,
            score_cap=100,
            policy_id=rr.policy_id,
            policy_version=rr.policy_version,
            calculation_hash=rr.calculation_hash,
            is_incomplete=False,
            claims=(
                ClaimReference(
                    claim_id=f"claim-{aid}-{rnum}-risk-score",
                    statement=(
                        f"Deterministic risk score calculated as {rr.score} "
                        f"({rr.band.value if rr.band else 'LOW'})"
                    ),
                    risk_factor_refs=tuple(f.factor.value for f in rr.factors if f.triggered),
                ),
            ),
        )

    def _build_mandatory_findings(self, context: ReportDraftingContext) -> MandatoryReviewSection:
        rr = context.risk_result
        aid = context.assessment_id
        rnum = context.run_number
        findings = rr.findings
        claims: list[ClaimReference] = []
        ev_refs: set[str] = set()
        pol_refs: set[str] = set()
        rf_refs: set[str] = set()

        for f in findings:
            ev_refs.update(f.evidence_refs)
            pol_refs.update(f.policy_citation_refs)
            factor_name = f.factor.value if isinstance(f.factor, RiskFactor) else str(f.factor)
            rf_refs.add(factor_name)
            claims.append(
                ClaimReference(
                    claim_id=f"claim-{aid}-{rnum}-finding-{f.finding_id}",
                    statement=f"Mandatory Review Finding: {f.title} ({f.finding_id})",
                    evidence_refs=f.evidence_refs,
                    policy_citation_refs=f.policy_citation_refs,
                    risk_factor_refs=(factor_name,),
                )
            )

        return MandatoryReviewSection(
            findings=findings,
            claims=tuple(claims),
            evidence_refs=tuple(sorted(ev_refs)),
            policy_citation_refs=tuple(sorted(pol_refs)),
            risk_factor_refs=tuple(sorted(rf_refs)),
        )

    def _build_contributing_factors(
        self, context: ReportDraftingContext
    ) -> ContributingFactorsSection:
        rr = context.risk_result
        aid = context.assessment_id
        rnum = context.run_number
        factors = rr.factors
        triggered = tuple(f.factor for f in factors if f.triggered)

        claims: list[ClaimReference] = []
        ev_refs: set[str] = set()
        pol_refs: set[str] = set()
        rf_refs: set[str] = set()

        for f in factors:
            ev_refs.update(f.evidence_refs)
            pol_refs.update(f.policy_citation_refs)
            factor_name = f.factor.value if isinstance(f.factor, RiskFactor) else str(f.factor)
            rf_refs.add(factor_name)
            status = "TRIGGERED" if f.triggered else "CLEAR"
            claims.append(
                ClaimReference(
                    claim_id=f"claim-{aid}-{rnum}-factor-{factor_name}",
                    statement=(
                        f"Factor {factor_name}: {status} (weight={f.weight}, "
                        f"contribution={f.score_contribution})"
                    ),
                    evidence_refs=f.evidence_refs,
                    policy_citation_refs=f.policy_citation_refs,
                    risk_factor_refs=(factor_name,),
                )
            )

        return ContributingFactorsSection(
            factor_breakdown=factors,
            triggered_factors=triggered,
            claims=tuple(claims),
            evidence_refs=tuple(sorted(ev_refs)),
            policy_citation_refs=tuple(sorted(pol_refs)),
            risk_factor_refs=tuple(sorted(rf_refs)),
        )

    def _build_policy_citations(self, context: ReportDraftingContext) -> PolicyCitationsSection:
        citations = context.policy_citations
        aid = context.assessment_id
        rnum = context.run_number
        is_abstention = context.metadata.get("is_abstention", False) or not citations

        abstention_notice = None
        if is_abstention or len(citations) == 0:
            abstention_notice = AbstentionNotice(
                topic="NZ Vehicle Risk Policy Citations",
                reason="No authoritative policy passage met the minimum relevance threshold",
                impact="Policy citations omitted; standard policy baseline rules applied",
            )

        claims: list[ClaimReference] = []
        pol_refs: list[str] = []
        for c in citations:
            pol_refs.append(c.passage_id)
            claims.append(
                ClaimReference(
                    claim_id=f"claim-{aid}-{rnum}-cit-{c.passage_id}",
                    statement=(
                        f"Policy Grounding: {c.heading} ({c.passage_id}) from {c.source_title}"
                    ),
                    policy_citation_refs=(c.passage_id,),
                )
            )

        return PolicyCitationsSection(
            citations=citations,
            has_abstention=is_abstention,
            abstention_notice=abstention_notice,
            claims=tuple(claims),
            policy_citation_refs=tuple(sorted(set(pol_refs))),
        )

    def _build_evidence_summary(self, context: ReportDraftingContext) -> EvidenceSummarySection:
        snap = context.evidence_snapshot
        aid = context.assessment_id
        rnum = context.run_number
        if snap is None:
            return EvidenceSummarySection(
                revision_id="untracked",
                revision_number=1,
                material_hash="",
                canonical_fields={},
                conflict_count=0,
                conflicts=(),
                history_depth=0,
            )

        ev_refs: set[str] = set()
        for prov_list in snap.field_provenance.values():
            for p in prov_list:
                if p.observation_id:
                    ev_refs.add(p.observation_id)

        claims = (
            ClaimReference(
                claim_id=f"claim-{aid}-{rnum}-ev-summary",
                statement=(
                    f"Evidence snapshot revision {snap.revision_number} "
                    f"({snap.revision_id}) material hash verified"
                ),
                evidence_refs=tuple(sorted(ev_refs)),
            ),
        )

        return EvidenceSummarySection(
            revision_id=snap.revision_id,
            revision_number=snap.revision_number,
            material_hash=snap.material_hash,
            as_of=snap.as_of,
            canonical_fields=dict(snap.canonical_fields),
            conflict_count=len(snap.conflicts),
            conflicts=snap.conflicts,
            history_depth=len(snap.history),
            claims=claims,
            evidence_refs=tuple(sorted(ev_refs)),
        )

    def _build_limitations_and_missing_evidence(
        self, context: ReportDraftingContext
    ) -> LimitationsSection:
        rr = context.risk_result
        aid = context.assessment_id
        rnum = context.run_number
        missing_list = rr.missing_evidence or rr.missing_findings or ()
        missing_notices = tuple(
            MissingEvidenceNotice(
                field_name=mf.field_name,
                reason=mf.reason,
                details=mf.details,
            )
            for mf in missing_list
        )

        abstention_notices: list[AbstentionNotice] = []
        if context.metadata.get("is_abstention", False):
            abstention_notices.append(
                AbstentionNotice(
                    topic="Policy Retrieval",
                    reason="No policy passage met threshold",
                    impact="Standard policy defaults used",
                )
            )

        claims: list[ClaimReference] = []
        for mn in missing_notices:
            claims.append(
                ClaimReference(
                    claim_id=f"claim-{aid}-{rnum}-lim-missing-{mn.field_name}",
                    statement=(
                        f"Missing evidence disclosure for field {mn.field_name}: "
                        f"{mn.reason} ({mn.details})"
                    ),
                )
            )

        return LimitationsSection(
            missing_evidence_notices=missing_notices,
            abstention_notices=tuple(abstention_notices),
            claims=tuple(claims),
        )

    def _build_synthetic_notice(self, context: ReportDraftingContext) -> SyntheticNoticeSection:
        is_synthetic = context.is_synthetic_context()
        aid = context.assessment_id
        rnum = context.run_number
        if not is_synthetic:
            return SyntheticNoticeSection(
                is_synthetic=False,
                notice=None,
                disclaimer_text=None,
            )

        synthetic_sources: list[str] = []
        if context.evidence_snapshot is not None:
            for _field, prov_list in context.evidence_snapshot.field_provenance.items():
                for p in prov_list:
                    if p.synthetic and p.observation_id:
                        synthetic_sources.append(p.observation_id)

        notice = SyntheticNotice(
            is_synthetic=True,
            notice_text=(
                "This assessment contains synthetic demonstration data and must not "
                "be used for production credit, financing, or underwriting decisions."
            ),
            synthetic_sources=tuple(sorted(set(synthetic_sources))),
        )

        return SyntheticNoticeSection(
            is_synthetic=True,
            notice=notice,
            disclaimer_text=notice.notice_text,
            claims=(
                ClaimReference(
                    claim_id=f"claim-{aid}-{rnum}-synth-notice",
                    statement=(
                        "Synthetic demonstration data notice active for this assessment draft"
                    ),
                    evidence_refs=tuple(sorted(set(synthetic_sources))),
                ),
            ),
            evidence_refs=tuple(sorted(set(synthetic_sources))),
        )
