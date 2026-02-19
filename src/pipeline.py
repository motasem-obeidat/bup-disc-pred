"""Data Filtering"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from datetime import timedelta
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class BupConfig:
    min_bup_start: pd.Timestamp = pd.Timestamp("2018-01-01")
    max_bup_start: pd.Timestamp = pd.Timestamp("2022-12-31")
    followup_cutoff: pd.Timestamp = pd.Timestamp("2021-12-31")
    enrollment_days_pre: int = 180
    enrollment_days_post: int = 365
    followup_days: int = 365
    gap_days: int = 30
    pdc_days: int = 30
    min_age: int = 18


# -----------------------------------------------------------------------------
# Processor
# -----------------------------------------------------------------------------


class BupProcessor:
    def __init__(
        self,
        cohort_path: str,
        enrolid_detail_path: str,
        rx_path: str,
        diag_path: str,
        encounter_path: str,
        redbook_path: str,
        procedure_path: str,
        charlson_path: str,
        *,
        config: Optional[BupConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.paths = {
            "cohort": cohort_path,
            "enrolid_detail": enrolid_detail_path,
            "rx": rx_path,
            "diag": diag_path,
            "encounter": encounter_path,
            "redbook": redbook_path,
            "procedure": procedure_path,
            "charlson": charlson_path,
        }
        self.config = config or BupConfig()
        self.log = logger or logging.getLogger("BupProcessor")
        if not self.log.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(message)s")
            handler.setFormatter(formatter)
            self.log.addHandler(handler)
            self.log.setLevel(logging.INFO)

        # Placeholders for loaded data
        self.cohort: pd.DataFrame = pd.DataFrame()
        self.enrolid_detail: pd.DataFrame = pd.DataFrame()
        self.rx: pd.DataFrame = pd.DataFrame()
        self.diag: pd.DataFrame = pd.DataFrame()
        self.encounter: pd.DataFrame = pd.DataFrame()
        self.redbook: pd.DataFrame = pd.DataFrame()
        self.procedure: pd.DataFrame = pd.DataFrame()
        self.charlson: pd.DataFrame = pd.DataFrame()

        # Working tables
        self.bup_rx_all: pd.DataFrame = pd.DataFrame()
        self.switchers: set[str] = set()

    # Loading & Cleaning
    def load_all_datasets(self) -> None:
        cfg = self.config
        self.log.info("Loading input CSVs…")

        self.cohort = pd.read_csv(
            self.paths["cohort"],
            parse_dates=["FIRST_RECORD_DATE", "OUD_DATE", "BUP_START_DATE"],
            dtype={"ENROLID": str},
        )
        self.enrolid_detail = pd.read_csv(
            self.paths["enrolid_detail"],
            parse_dates=["ELIG_SPAN_BEGIN", "ELIG_SPAN_END"],
            dtype={"ENROLID": str, "DOBYR": "Int64"},
        )
        self.rx = pd.read_csv(
            self.paths["rx"],
            parse_dates=["FILLDATE"],
            dtype={
                "ENROLID": str,
                "NDCNUM": str,
                "DRUG_NAME": str,
                "TCGPI_ID": str,
                "TCGPI_NAME": str,
            },
        )
        self.diag = pd.read_csv(
            self.paths["diag"],
            parse_dates=["SVCDATE"],
            dtype={"ENROLID": str, "DIAG_CD": str, "DIAG_NUM": "Int64"},
        )
        self.encounter = pd.read_csv(
            self.paths["encounter"],
            parse_dates=["SVCDATE"],
            dtype={"ENROLID": str, "IS_MEDICARE": "Int64"},
        )
        self.redbook = pd.read_csv(
            self.paths["redbook"],
            dtype={"NDCNUM": str, "MASTFRM": str, "PRODNME": str, "ROADS": str},
        )
        self.procedure = pd.read_csv(
            self.paths["procedure"], parse_dates=["SVCDATE"], dtype={"ENROLID": str}
        )
        self.charlson = pd.read_csv(
            self.paths["charlson"], parse_dates=["SVCDATE"], dtype={"ENROLID": str}
        )

        self._clean_blank_strings(
            [
                self.cohort,
                self.enrolid_detail,
                self.rx,
                self.diag,
                self.encounter,
                self.redbook,
                self.procedure,
                self.charlson,
            ]
        )

        self.log.info("Loaded cohort with %s patients", f"{len(self.cohort):,}")

    @staticmethod
    def _clean_blank_strings(dfs: Iterable[pd.DataFrame]) -> None:
        for df in dfs:
            object_cols = df.select_dtypes(include=["object"]).columns
            if len(object_cols):
                df[object_cols] = df[object_cols].replace(r"^\s*$", np.nan, regex=True)

    # Pipeline Orchestration
    def run_full_pipeline(self) -> pd.DataFrame:
        self.load_all_datasets()
        self.step1_filter_by_age_and_study_period()
        self.step2_filter_by_followup_feasibility()
        self.step3_link_index_oud_diagnosis()
        self.extract_bup_prescriptions()
        self.step4_restrict_to_oud_formulations()
        self.step5_filter_by_minimum_prescription_count()
        self.step6_verify_continuous_insurance_enrollment()
        self.step7_exclude_medicare_encounters()
        self.step8_exclude_invalid_supply_records()
        self.step9_classify_index_daily_dose()
        self.calculate_index_fill_duration()
        self.step10_identify_MOUD_switch_events()
        self.step11_determine_discontinuation_status()
        self.step12_calculate_pdc_adherence()
        return self.cohort

    def step1_filter_by_age_and_study_period(self) -> None:
        self.log.info("=" * 80 + "\n")
        cfg = self.config
        self.log.info(
            f"[STEP 1] Applying base timeframe ({cfg.min_bup_start.date()} - {cfg.max_bup_start.date()}) + age >= {cfg.min_age}..."
        )

        initial_count = len(self.cohort)

        enrolid_age = (
            self.enrolid_detail.groupby("ENROLID")["DOBYR"].first().reset_index()
        )
        self.cohort = self.cohort.merge(enrolid_age, on="ENROLID", how="left")

        self.cohort["AGE_AT_BUP"] = (
            self.cohort["BUP_START_DATE"].dt.year - self.cohort["DOBYR"]
        )

        date_mask = self.cohort["BUP_START_DATE"].between(
            cfg.min_bup_start, cfg.max_bup_start
        )
        excluded_date = (~date_mask).sum()

        age_mask = self.cohort["AGE_AT_BUP"] >= cfg.min_age
        excluded_age = len(self.cohort[date_mask & ~age_mask])

        final_mask = date_mask & age_mask
        self.cohort = self.cohort.loc[final_mask].copy()

        self.log.info(f"   - Initial Cohort: {initial_count:,}")
        self.log.info(f"   - Excluded due to Date Range: {excluded_date:,}")
        self.log.info(f"   - Excluded due to Age < {cfg.min_age}: {excluded_age:,}")
        self.log.info(f"→ Remaining: {len(self.cohort):,}\n")

    def step2_filter_by_followup_feasibility(self) -> None:
        self.log.info("=" * 80 + "\n")
        cfg = self.config
        self.log.info(
            f"[STEP 2] Checking Follow-up Feasibility (BUP Start Date <= {cfg.followup_cutoff.date()}) to allow ≥ 12-month follow-up..."
        )

        mask = self.cohort["BUP_START_DATE"] <= cfg.followup_cutoff
        excluded_cnt = (~mask).sum()

        self.cohort = self.cohort.loc[mask].copy()

        self.log.info(f"   - Excluded (Insufficient follow-up time): {excluded_cnt:,}")
        self.log.info(f"→ Remaining: {len(self.cohort):,}\n")

    def step3_link_index_oud_diagnosis(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 3] Identifying most recent OUD diagnosis relative to BUP Start.\n"
            f"   - Window: SVCDATE ≤ BUP_START + 3 days (post-start diagnoses treated as 0-day offset)\n"
            f"   - Definition: ICD-9 (304.0x, 305.5x) or ICD-10 (F11.xx)\n"
            f"   - Classification: Using DIAG_NUM to flag Primary (Pos 1) vs. Secondary (Pos >1) diagnosis"
        )

        is_oud = (
            self.diag["DIAG_CD"]
            .str.strip()
            .str.upper()
            .str.startswith(("3040", "3055", "F11"), na=False)
        )

        oud = self.diag.loc[is_oud, ["ENROLID", "SVCDATE", "DIAG_NUM"]].copy()
        cohort_dates = self.cohort[["ENROLID", "BUP_START_DATE"]]

        d = oud.merge(cohort_dates, on="ENROLID", how="inner")
        d = d.loc[d["SVCDATE"] <= d["BUP_START_DATE"] + pd.Timedelta(days=3)].copy()

        d["OUD_OFFSET_DAYS"] = (d["BUP_START_DATE"] - d["SVCDATE"]).dt.days
        d.loc[d["OUD_OFFSET_DAYS"] < 0, "OUD_OFFSET_DAYS"] = 0

        most_recent = d.sort_values(
            ["ENROLID", "SVCDATE"], ascending=[True, False]
        ).drop_duplicates("ENROLID")

        primary_flag = (
            d.groupby("ENROLID")["DIAG_NUM"]
            .apply(lambda x: int((x == 1).any()))
            .reset_index(name="OUD_PRIMARY")
        )
        secondary_flag = (
            d.groupby("ENROLID")["DIAG_NUM"]
            .apply(lambda x: int((x > 1).any()))
            .reset_index(name="OUD_SECONDARY")
        )

        mr = most_recent.merge(primary_flag, on="ENROLID", how="left").merge(
            secondary_flag, on="ENROLID", how="left"
        )

        mr["OUD_OFFSET_DAYS"] = mr["OUD_OFFSET_DAYS"].astype("Int64")
        mr["OUD_PRIMARY"] = mr["OUD_PRIMARY"].astype("Int64")
        mr["OUD_SECONDARY"] = mr["OUD_SECONDARY"].astype("Int64")

        self.cohort = self.cohort.merge(
            mr[["ENROLID", "OUD_OFFSET_DAYS", "OUD_PRIMARY", "OUD_SECONDARY"]],
            on="ENROLID",
            how="left",
        )

        total_pts = len(self.cohort)
        with_oud = self.cohort["OUD_OFFSET_DAYS"].notna().sum()
        missing_oud = total_pts - with_oud

        prim_count = self.cohort["OUD_PRIMARY"].sum()
        sec_count = self.cohort["OUD_SECONDARY"].sum()
        avg_offset = self.cohort["OUD_OFFSET_DAYS"].mean()

        self.log.info(
            f"   - Patients with identified OUD Diagnosis: {with_oud:,} ({with_oud/total_pts:.1%})"
        )
        self.log.info(
            f"   - Patients without recent OUD Dx (historical & handled in SQL): {missing_oud:,}"
        )
        self.log.info(
            f"   - Diagnosis Position: {prim_count:,} had Primary, {sec_count:,} had Secondary"
        )
        self.log.info(f"   - Avg Days from OUD Dx to BUP Start: {avg_offset:.1f} days")
        self.log.info(f"→ Remaining (No filter applied): {len(self.cohort):,}\n")

    def extract_bup_prescriptions(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 4] Extracting BUP prescriptions during follow-up ({self.config.followup_days} days).\n"
            f"   - Window: BUP_START_DATE to BUP_START_DATE + {self.config.followup_days} days\n"
            f"   - Criteria: TCGPI_ID starts with '6520001' AND TCGPI_NAME contains 'buprenorphine'"
        )

        rx = self.rx.copy()

        rx["TCGPI_ID_STR"] = rx["TCGPI_ID"].fillna("").astype(str)
        rx["TCGPI_NAME_L"] = rx["TCGPI_NAME"].fillna("").str.lower()

        is_bup = rx["TCGPI_ID_STR"].str.startswith("6520001", na=False) & rx[
            "TCGPI_NAME_L"
        ].str.contains("buprenorphine", na=False)

        bup = rx.loc[is_bup].copy()
        bup = bup.merge(
            self.cohort[["ENROLID", "BUP_START_DATE"]], on="ENROLID", how="inner"
        )

        start = bup["BUP_START_DATE"]
        end = start + pd.Timedelta(days=self.config.followup_days)
        in_window = bup["FILLDATE"].between(start, end)

        self.bup_rx_all = bup.loc[in_window].copy()

        self.log.info(
            f"→ Extracted {len(self.bup_rx_all):,} valid BUP fills in follow-up window.\n"
        )

    def step4_restrict_to_oud_formulations(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            "[STEP 4] Filtering for OUD-indicated formulations only (Excluding Pain products).\n"
            "   - Logic: Retain only Buccal (Bunavail), Subcutaneous (Sublocade/Brixadi), and Sublingual (Suboxone/Zubsolv/Subutex/Buprenorphine)\n"
            "   - Source: Using Rx columns 'ROA_DESCR' (as Route of Administration) and 'DRUG_NAME' (as Product)\n"
            f"   - Window: BUP_START_DATE to BUP_START_DATE + {self.config.followup_days} days"
        )

        r = self.bup_rx_all.copy()

        r["ROADS"] = r["ROA_DESCR"]
        r["PRODNME"] = r["DRUG_NAME"]
        r["ROADS_L"] = r["ROADS"].fillna("").str.strip().str.lower()
        r["PRODNME_U"] = r["PRODNME"].fillna("").str.strip().str.upper()

        valid = (
            (
                (r["ROADS_L"] == "buccal")
                & r["PRODNME_U"].str.contains("BUNAVAIL", na=False)
            )
            | (
                (r["ROADS_L"] == "subcutaneous")
                & r["PRODNME_U"].str.contains("SUBLOCADE|BRIXADI", na=False)
            )
            | (
                (r["ROADS_L"] == "sublingual")
                & r["PRODNME_U"].str.contains(
                    "SUBOXONE|ZUBSOLV|SUBUTEX|BUPRENORPHINE", na=False
                )
            )
        )

        r["IS_VALID_OUD_RX"] = valid
        non_oud_patients = set(r.loc[~valid, "ENROLID"].unique())

        self.cohort = self.cohort[~self.cohort["ENROLID"].isin(non_oud_patients)].copy()
        self.bup_rx_all = r.loc[
            r["IS_VALID_OUD_RX"] & r["ENROLID"].isin(self.cohort["ENROLID"])
        ].copy()

        self.log.info(
            f"   - Patients Excluded (Prescribed non-OUD formulations): {len(non_oud_patients):,}"
        )

        self.log.info(
            "   --- Included Formulations Analysis (Grouped by Route & Drug) ---"
        )
        if not self.bup_rx_all.empty:
            included_stats = (
                self.bup_rx_all.groupby(["ROADS", "PRODNME"])
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )

            for _, row in included_stats.iterrows():
                self.log.info(
                    f"   - Route: {row['ROADS']:<15} | Drug: {row['PRODNME']:<30} (n={row['count']:,} fills)"
                )
        else:
            self.log.info("   - No valid OUD fills remaining.")

        self.log.info(f"→ Remaining: {len(self.cohort):,}\n")

    def step5_filter_by_minimum_prescription_count(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            "[STEP 5] Validating Treatment Persistence (Minimum 2 distinct fills).\n"
            "   - Logic: Excl. 'one-and-done' initiations; requires ≥2 fills on different dates\n"
            "   - Purpose: Ensuring cohort captures engaged patients vs. failed inductions\n"
            f"   - Window: BUP_START_DATE to BUP_START_DATE + {self.config.followup_days} days"
        )

        initial_count = len(self.cohort)

        fill_counts_df = (
            self.bup_rx_all.drop_duplicates(["ENROLID", "FILLDATE"])
            .groupby("ENROLID")
            .size()
            .reset_index(name="fill_count")
        )

        all_patients = set(self.cohort["ENROLID"])
        patients_with_fills = set(fill_counts_df["ENROLID"])

        zero_fills = all_patients - patients_with_fills

        one_fill = set(fill_counts_df.loc[fill_counts_df["fill_count"] == 1, "ENROLID"])
        one_fill = one_fill.intersection(all_patients)

        eligible = set(fill_counts_df.loc[fill_counts_df["fill_count"] >= 2, "ENROLID"])
        eligible = eligible.intersection(all_patients)

        self.cohort = self.cohort[self.cohort["ENROLID"].isin(eligible)].copy()
        self.bup_rx_all = self.bup_rx_all[
            self.bup_rx_all["ENROLID"].isin(self.cohort["ENROLID"])
        ].copy()

        self.log.info(f"   - Initial Cohort: {initial_count:,}")
        self.log.info(
            f"   - Excluded (0 fills found): {len(zero_fills):,} "
            "(No Rx records found in follow-up window)"
        )
        self.log.info(
            f"   - Excluded (1 fill only): {len(one_fill):,} "
            "(Early discontinuation / 'One-and-Done')"
        )
        self.log.info(
            f"→ Remaining (Retained ≥2 distinct fills): {len(self.cohort):,}\n"
        )

    def step6_verify_continuous_insurance_enrollment(self) -> None:
        self.log.info("=" * 80 + "\n")
        cfg = self.config
        self.log.info("[STEP 6] Continuous Enrollment Check...")
        self.log.info(
            f"   [NOTE] Logic Skipped (Pre-calculated in SQL). "
            f"Required Definition: {cfg.enrollment_days_pre} days pre-index "
            f"through {cfg.enrollment_days_post} days post-index."
        )
        self.log.info(f"→ Remaining: {len(self.cohort):,}\n")

    def step7_exclude_medicare_encounters(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 7] Excluding patients with Medicare Advantage encounters.\n"
            f"   - Logic: Identify patients with any encounter where IS_MEDICARE = 1\n"
            f"   - Window: BUP_START_DATE to BUP_START_DATE + {self.config.followup_days} days"
        )

        m = self.cohort[["ENROLID", "BUP_START_DATE"]].merge(
            self.encounter[["ENROLID", "SVCDATE", "IS_MEDICARE"]],
            on="ENROLID",
            how="left",
        )

        follow_end = m["BUP_START_DATE"] + pd.Timedelta(days=self.config.followup_days)
        in_window = m["SVCDATE"].between(m["BUP_START_DATE"], follow_end)

        flag = (
            m.loc[in_window]
            .groupby("ENROLID")["IS_MEDICARE"]
            .max()
            .reset_index()
            .query("IS_MEDICARE == 1")
        )

        medicare_patients = set(flag["ENROLID"])
        count_medicare = len(medicare_patients)

        before = len(self.cohort)
        self.cohort = self.cohort[
            ~self.cohort["ENROLID"].isin(medicare_patients)
        ].copy()
        after = len(self.cohort)

        self.bup_rx_all = self.bup_rx_all[
            self.bup_rx_all["ENROLID"].isin(self.cohort["ENROLID"])
        ].copy()

        self.log.info(
            f"   - Patients identified with Medicare Advantage: {count_medicare:,}"
        )
        self.log.info(f"   - Excluded: {before - after:,}")
        self.log.info(f"→ Remaining: {after:,}\n")

    def step8_exclude_invalid_supply_records(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            "[STEP 8] Data Quality Check: Dropping patients with invalid DAYSUPP (≤ 0)."
        )

        r = self.bup_rx_all.copy()
        r["DAYSUPP"] = pd.to_numeric(r.get("DAYSUPP"), errors="coerce")

        invalid_patients = set(r.loc[r["DAYSUPP"] <= 0, "ENROLID"].unique())
        count_invalid = len(invalid_patients)

        before = len(self.cohort)
        self.cohort = self.cohort[~self.cohort["ENROLID"].isin(invalid_patients)].copy()
        after = len(self.cohort)

        self.bup_rx_all = r[r["ENROLID"].isin(self.cohort["ENROLID"])].copy()

        self.log.info(
            f"   - Patients with invalid Rx records (Days Supply <= 0): {count_invalid:,}"
        )
        self.log.info(f"   - Excluded: {before - after:,}")
        self.log.info(f"→ Remaining: {after:,}\n")

    def step9_classify_index_daily_dose(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            "[STEP 9] Classifying Initial Daily Dose (Index Date).\n"
            "   - Logic: SUM of daily dose contributions of all fills occurring exactly on BUP_START_DATE.\n"
            "   - Formula: (Strength * Quantity) / Days Supply\n"
            "   - Dose Bins: 0=Low(≤8), 1=Std(8-16), 2=High(16-24), 3=Very High(>24), 4=injectable, -1=Unknown\n"
            "   - Grouping Definitions:\n"
            "     * Groups 0-3 (Non-injectable): ['sublingual', 'buccal', 'film', 'tablet', 'transmucosal']\n"
            "     * Group 4 (Injectable): ['subcutaneous', 'intramuscular', 'injection', 'liquid']\n"
            "     * Unknown (-1): Any descriptions not matching above\n"
        )

        r = self.bup_rx_all.copy()
        r = r.loc[r["FILLDATE"] == r["BUP_START_DATE"]].copy()

        r["DAYSUPP"] = pd.to_numeric(r.get("DAYSUPP"), errors="coerce")
        r["METQTY"] = pd.to_numeric(r.get("METQTY"), errors="coerce")
        r["ROADS"] = r.get("ROADS", "").fillna("").str.strip().str.lower()

        if "STRENGTH_UOM" in r.columns:
            unique_units = r["STRENGTH_UOM"].dropna().unique()
            # self.log.debug("Unique STRENGTH_UOM values: %s", list(unique_units))

        def _parse_strength(v: object) -> Optional[float]:
            if pd.isna(v):
                return None
            s = str(v).strip()
            m = re.match(r"^\s*(\d+(?:\.\d+)?)", s)
            if m:
                try:
                    return float(m.group(1))
                except (ValueError, IndexError):
                    return None
            return None

        r["STRENGTH_PARSED"] = r.get("STRENGTH").apply(_parse_strength)

        r["DAILY_DOSE_CONTRIB"] = (r["STRENGTH_PARSED"] * r["METQTY"]) / r["DAYSUPP"]

        dose_agg = (
            r.groupby("ENROLID")["DAILY_DOSE_CONTRIB"]
            .sum()
            .reset_index()
            .rename(columns={"DAILY_DOSE_CONTRIB": "DAILY_DOSE_TOTAL"})
        )

        def _get_route_code(roads_series):
            valid = [x for x in roads_series.unique() if x and str(x).strip()]
            if not valid:
                return -1
            combined = " ".join(valid).lower()
            if any(
                x in combined
                for x in ["subcutaneous", "intramuscular", "injection", "liquid"]
            ):
                return 1
            if any(
                x in combined
                for x in ["sublingual", "buccal", "film", "tablet", "transmucosal"]
            ):
                return 0
            return -1

        route_agg = (
            r.groupby("ENROLID")["ROADS"]
            .apply(_get_route_code)
            .reset_index(name="ROUTE_GROUP")
        )

        first = dose_agg.merge(route_agg, on="ENROLID", how="left")

        def _dose_bin(row):
            dose = row["DAILY_DOSE_TOTAL"]
            route = row["ROUTE_GROUP"]
            if route == 1:
                return 4  # Injectable
            if route == -1 or pd.isna(dose):
                return -1
            if dose <= 8:
                return 0
            if dose <= 16:
                return 1
            if dose <= 24:
                return 2
            return 3

        first["INITIAL_DOSE_GROUP"] = first.apply(_dose_bin, axis=1)

        self.cohort = self.cohort.merge(
            first[["ENROLID", "DAILY_DOSE_TOTAL", "INITIAL_DOSE_GROUP", "ROUTE_GROUP"]],
            on="ENROLID",
            how="left",
        )

        self.cohort = self.cohort.rename(columns={"DAILY_DOSE_TOTAL": "DAILY_DOSE"})
        self.cohort["DAILY_DOSE"] = self.cohort["DAILY_DOSE"].round(2)

        group_labels = {
            -1: "Unknown/Missing",
            0: "Low (<= 8 mg)",
            1: "Standard (> 8-16 mg)",
            2: "High (> 16-24 mg)",
            3: "Very High (> 24 mg)",
            4: "Injectable",
        }

        counts = self.cohort["INITIAL_DOSE_GROUP"].value_counts().sort_index()
        total_classified = counts.sum()

        self.log.info(f"   - Total Patients Classified: {total_classified:,}")
        self.log.info("   - Distribution of Initial Doses:")
        for grp_id, count in counts.items():
            label = group_labels.get(grp_id, f"Group {grp_id}")
            pct = (count / total_classified) * 100
            self.log.info(f"     * {label:<22} : {count:,} ({pct:.1f}%)")

        non_injectable = self.cohort.loc[
            self.cohort["INITIAL_DOSE_GROUP"].isin([0, 1, 2, 3]), "DAILY_DOSE"
        ]
        if not non_injectable.empty:
            self.log.info(
                f"   - Non-injectable Dose Stats: Mean={non_injectable.mean():.1f}mg, Median={non_injectable.median():.1f}mg"
            )

        self.log.info(f"→ Remaining: {len(self.cohort):,}\n")

    def calculate_index_fill_duration(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            "[INFO] Adding 'INITIAL_DAYSUPP' (Days Supply on BUP Start Date).\n"
            "   - Logic: Filter to fills occurring exactly on BUP_START_DATE.\n"
            "   - Aggregation: SUM of DAYSUPP if multiple fills exist on this date.\n"
        )

        r = self.bup_rx_all.copy()
        r = r.loc[r["FILLDATE"] == r["BUP_START_DATE"]].copy()
        r["DAYSUPP"] = pd.to_numeric(r.get("DAYSUPP"), errors="coerce")

        first = (
            r.groupby("ENROLID")["DAYSUPP"]
            .sum()
            .reset_index()
            .rename(columns={"DAYSUPP": "INITIAL_DAYSUPP"})
        )
        self.cohort = self.cohort.merge(first, on="ENROLID", how="left")

    def step10_identify_MOUD_switch_events(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 10] Identifying MOUD Switchers (Switch to Methadone or Naltrexone).\n"
            f"   - Window: BUP_START_DATE to BUP_START_DATE + {self.config.followup_days} days\n"
            f"   - Criteria: \n"
            f"       1. Rx: Must match TCGPI_NAME contains 'naltrexone')\n"
            f"          AND fall into one of these specific Formulation categories:\n"
            f"          a. Route='intramuscular' & Name='VIVITROL'\n"
            f"          b. Route='oral' & Name in {{'EMBEDA', 'NALTREXONE HCL'}})\n"
            f"          c. TCGPI_NAME='naltrexone hcl (bulk) powder' & Name in {{'NALTREXONE HCL', 'NALTREXONE'}})\n"
            f"       2. Procedure: Methadone/Naltrexone (Codes: H0020, G2067, G2078, S0109, J2315)"
        )

        # --- 1. Naltrexone RX Logic ---
        r = self.rx.copy()
        r["ROADS"] = r["ROA_DESCR"]
        r["PRODNME"] = r["DRUG_NAME"]
        r["ROADS_L"] = r["ROADS"].fillna("").str.strip().str.lower()
        r["PRODNME_U"] = r["PRODNME"].fillna("").str.strip().str.upper()
        r["TCGPI_ID_STR"] = r["TCGPI_ID"].fillna("").astype(str)
        r["TCGPI_NAME_L"] = r["TCGPI_NAME"].fillna("").str.lower()

        is_naltrexone_base = r["TCGPI_NAME_L"].str.contains(
            "naltrexone", case=False, na=False
        )

        is_Intramuscular = (r["ROADS_L"] == "intramuscular") & (
            r["PRODNME_U"] == "VIVITROL"
        )
        is_oral_nal = (r["ROADS_L"] == "oral") & r["PRODNME_U"].isin(
            {"EMBEDA", "NALTREXONE HCL"}
        )

        is_bulk = (r["TCGPI_NAME_L"] == "naltrexone hcl (bulk) powder") & r[
            "PRODNME_U"
        ].isin({"NALTREXONE HCL", "NALTREXONE"})

        is_naltrexone_rx = is_naltrexone_base & (
            is_Intramuscular | is_oral_nal | is_bulk
        )
        switch_naltrexone_all_fills = r.loc[is_naltrexone_rx].copy()

        switch_naltrexone = switch_naltrexone_all_fills.merge(
            self.cohort[["ENROLID", "BUP_START_DATE"]], on="ENROLID", how="inner"
        )
        follow_end_nal = switch_naltrexone["BUP_START_DATE"] + pd.Timedelta(
            days=self.config.followup_days
        )
        in_window_nal = switch_naltrexone["FILLDATE"].between(
            switch_naltrexone["BUP_START_DATE"], follow_end_nal
        )

        switch_fills_in_window = switch_naltrexone.loc[in_window_nal]
        switchers_naltrexone_rx = set(switch_fills_in_window["ENROLID"].unique())

        # --- 2. Procedure Logic ---
        switcher_proc_codes = {"H0020", "G2067", "G2078", "S0109", "J2315"}
        proc = self.procedure.loc[
            self.procedure.get("PROCCD").isin(switcher_proc_codes)
        ].copy()
        proc = proc.merge(
            self.cohort[["ENROLID", "BUP_START_DATE"]], on="ENROLID", how="inner"
        )
        follow_end_p = proc["BUP_START_DATE"] + pd.Timedelta(
            days=self.config.followup_days
        )
        in_window_p = proc["SVCDATE"].between(proc["BUP_START_DATE"], follow_end_p)
        switchers_proc = set(proc.loc[in_window_p, "ENROLID"].unique())

        self.switchers = switchers_naltrexone_rx | switchers_proc
        self.cohort["MOUD_SWITCHER"] = (
            self.cohort["ENROLID"].isin(self.switchers)
        ).astype(int)

        self.log.info(f"   - Total Switchers Identified: {len(self.switchers):,}")
        self.log.info(f"     * Via Rx: {len(switchers_naltrexone_rx):,}")
        self.log.info(f"     * Via Procedure: {len(switchers_proc):,}")

        if not switch_fills_in_window.empty:
            self.log.info("   - Breakdown of Naltrexone Rx Formulations Found:")
            summary = (
                switch_fills_in_window.groupby(["ROADS", "PRODNME"])
                .size()
                .reset_index(name="count")
            )
            for _, row in summary.iterrows():
                self.log.info(
                    f"     * {row['ROADS']:<15} | {row['PRODNME']:<20} : {row['count']:,} fills"
                )
        else:
            self.log.info("   - No Naltrexone Rx switcher fills found.")

        self.log.info(f"→ Remaining: {len(self.cohort):,}\n")

    def consolidate_daily_prescriptions(self) -> pd.DataFrame:
        self.log.info("=" * 80 + "\n")
        r = self.bup_rx_all.copy()
        input_count = len(r)
        r["DAYSUPP"] = pd.to_numeric(r.get("DAYSUPP"), errors="coerce")

        r_agg = (
            r.groupby(["ENROLID", "FILLDATE"])
            .agg(
                DAYSUPP=("DAYSUPP", "sum"),
                BUP_START_DATE=("BUP_START_DATE", "first"),
            )
            .reset_index()
        )
        r_agg = r_agg.sort_values(["ENROLID", "FILLDATE"], ascending=True)
        r_agg["END_DATE"] = r_agg["FILLDATE"] + pd.to_timedelta(
            r_agg["DAYSUPP"], unit="D"
        )

        output_count = len(r_agg)
        collapsed_count = input_count - output_count
        return r_agg

    def step11_determine_discontinuation_status(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 11] Classifying Patient-Level Treatment Status (Continued vs Discontinued).\n"
            f"   - Gap Threshold: ≥ {self.config.gap_days} days between supply end and next fill implies discontinuation.\n"
            f"   - Hierarchy: If 'Switcher' flag is set → 'Discontinued – Switch'; otherwise check gaps.\n"
            f"   - Note: Same-day fills are aggregated (DAYSUPP summed).\n"
        )

        r = self.consolidate_daily_prescriptions()

        def _status(group: pd.DataFrame, enrolid: str) -> str:
            if enrolid in self.switchers:
                return "Discontinued – Switch"
            group = group.sort_values("FILLDATE")
            prev_end = group.iloc[0]["END_DATE"]
            for i in range(1, len(group)):
                fill_date = group.iloc[i]["FILLDATE"]
                if (fill_date - prev_end).days >= self.config.gap_days:
                    return "Discontinued – Gap"
                prev_end = max(prev_end, group.iloc[i]["END_DATE"])
            return "Continued"

        status_df = (
            r.groupby("ENROLID")
            .apply(lambda g: _status(g, g.name))
            .reset_index(name="BUP_STATUS")
        )
        self.cohort = self.cohort.merge(status_df, on="ENROLID", how="left")

        status_map = {
            "Continued": 0,
            "Discontinued – Gap": 1,
            "Discontinued – Switch": 2,
        }
        self.cohort["BUP_STATUS_NUM"] = (
            self.cohort["BUP_STATUS"].map(status_map).astype("Int64")
        )

        vc = self.cohort["BUP_STATUS"].value_counts(dropna=False)
        total = len(self.cohort)

        self.log.info("   - Status Distribution:")
        for status, count in vc.items():
            pct = (count / total) * 100
            self.log.info(f"     * {status:<25} : {count:,} ({pct:.1f}%)")

        self.log.info("\n   - Discontinuation by Initial Dose Group:")
        ct = pd.crosstab(self.cohort["INITIAL_DOSE_GROUP"], self.cohort["BUP_STATUS"])
        group_labels = {
            -1: "Unknown",
            0: "Low",
            1: "Std",
            2: "High",
            3: "V.High",
            4: "Inj/Imp",
        }
        ct.index = ct.index.map(lambda x: group_labels.get(x, x))
        self.log.info("\n" + ct.to_string())

        self.log.info(f"→ Remaining: {len(self.cohort):,}\n")

    def step12_calculate_pdc_adherence(self) -> None:
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 12] Calculating Proportion of Days Covered (PDC).\n"
            f"   - Window: First {self.config.pdc_days} days starting on the index date (index day included).\n"
            f"   - Definition: (Unique days with drug supply / {self.config.pdc_days}).\n"
            f"   - Threshold for High Adherence: ≥ 80% (PDC ≥ 0.8).\n"
            f"   - Note: Same-day fills are aggregated (DAYSUPP summed)."
        )

        r = self.consolidate_daily_prescriptions()
        pdc_rows: List[Dict[str, object]] = []

        for enrolid, g in r.groupby("ENROLID"):
            start_date = g["BUP_START_DATE"].iloc[0]
            days = np.zeros(self.config.pdc_days, dtype=bool)
            for _, row in g.iterrows():
                s = max((row["FILLDATE"] - start_date).days, 0)
                e = min((row["END_DATE"] - start_date).days, self.config.pdc_days - 1)
                if s <= e:
                    days[s : e + 1] = True
            pdc = round(days.mean(), 2)
            pdc_rows.append({"ENROLID": enrolid, "PDC_30": pdc})

        pdc_df = pd.DataFrame(pdc_rows)
        self.cohort = self.cohort.merge(pdc_df, on="ENROLID", how="left")

        self.cohort["PDC30_CAT"] = self.cohort["PDC_30"].apply(
            lambda x: 1 if (pd.notna(x) and x >= 0.8) else 0
        )

        desc = self.cohort["PDC_30"].describe(percentiles=[0.25, 0.5, 0.75]).round(2)
        self.log.info(f"   - PDC Stats (Mean/Median): {desc['mean']} / {desc['50%']}")

        cat_counts = self.cohort["PDC30_CAT"].value_counts().sort_index()
        self.log.info("   - Adherence Categories (Threshold 80%):")

        total = len(self.cohort)
        if 0 in cat_counts:
            c0 = cat_counts[0]
            self.log.info(f"     * Low Adherence (<80%)  : {c0:,} ({c0/total:.1%})")
        if 1 in cat_counts:
            c1 = cat_counts[1]
            self.log.info(f"     * High Adherence (≥80%) : {c1:,} ({c1/total:.1%})")

        self.log.info("=" * 80 + "\n")


# -----------------------------------------------------------------------------

"""Features Extraction"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureConfig:
    lookback_days_prebup: int = 180
    lookforward_days_postbup: int = 360

    age_bins: Tuple[Tuple[int, int, Optional[int]], ...] = (
        (0, 18, 24),
        (1, 25, 34),
        (2, 35, 44),
        (3, 45, 54),
        (4, 55, None),
    )

    diagnosis_code_sets: Dict[str, Tuple[str, ...]] = field(
        default_factory=lambda: {
            "CHRONIC_PAIN": (
                "E0842",
                "E0942",
                "E1042",
                "E1142",
                "E1342",
                "G43",
                "G44",
                "G501",
                "G560",
                "G564",
                "G57",
                "G589",
                "G60",
                "G61",
                "G62",
                "G63",
                "G64",
                "G65",
                "G890",
                "G892",
                "G894",
                "G900",
                "G990",
                "H46",
                "H47",
                "M00",
                "M01",
                "M02",
                "M05",
                "M06",
                "M07",
                "M08",
                "M11",
                "M12",
                "M13",
                "M14",
                "M15",
                "M16",
                "M17",
                "M18",
                "M19",
                "M20",
                "M21",
                "M22",
                "M23",
                "M24",
                "M25",
                "M30",
                "M31",
                "M32",
                "M33",
                "M34",
                "M35",
                "M36",
                "M37",
                "M38",
                "M39",
                "M40",
                "M41",
                "M42",
                "M43",
                "M44",
                "M45",
                "M46",
                "M47",
                "M48",
                "M49",
                "M50",
                "M51",
                "M52",
                "M53",
                "M54",
                "M55",
                "M56",
                "M57",
                "M58",
                "M59",
                "M60",
                "M61",
                "M62",
                "M63",
                "M64",
                "M65",
                "M66",
                "M67",
                "M68",
                "M69",
                "M70",
                "M71",
                "M72",
                "M73",
                "M74",
                "M75",
                "M76",
                "M77",
                "M78",
                "M79",
                "M80",
                "M81",
                "M82",
                "M83",
                "M84",
                "M85",
                "M86",
                "M87",
                "M88",
                "M89",
                "M90",
                "M91",
                "M92",
                "M93",
                "M94",
                "M95",
                "M96",
                "M97",
                "M98",
                "M99",
                "R262",
                "R294",
                "R29898",
                "R51",
            ),
            "HIV_AIDS": ("B20", "B21", "B22", "B23", "B24"),
            "HEPATITIS_C": ("B1710", "B1711", "B182", "B1920", "B1921"),
            "DEPRESSIVE_DISORDER": ("F32", "F33", "F341"),
            "ANXIETY": ("F40", "F41", "F42"),
            "PTSD": ("F431",),
            "BIPOLAR_DISORDER": ("F31", "F340"),
            "SCHIZOPHRENIA": ("F20", "F21", "F25"),
            "NON_OPIOID_SUD": ("F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"),
            "ALCOHOL_USE_DISORDER": ("F10",),
            "OUD_MILD": ("F1110",),
            "OUD_MODSEV": ("F1120",),
            "CUD_MILD": ("F1210",),
            "CUD_MODSEV": ("F1220",),
        }
    )

    medication_categories: Dict[str, Tuple[Tuple[str, str], ...]] = field(
        default_factory=lambda: {
            "ANTIDEPRESSANTS": (("58", "Antidepressants"),),
            "ANTIPSYCHOTICS": (("59", "Antipsychotics/antimanic Agents"),),
            "MOOD_STABILIZERS": (
                ("7937", "Lithium  (most common)"),
                ("5950", "Antimanic Agents"),
                ("7250", "Valproic Acid"),
            ),
            "BENZODIAZEPINES": (
                ("5710", "Benzodiazepines"),
                ("7210", "Anticonvulsants - Benzodiazepines"),
            ),
            "NONBENZODIAZEPINE": (("6020", "Non-barbiturate Hypnotics"),),
            "STIMULANTS": (
                ("6110", "Amphetamines"),
                ("61", "Adhd/anti-narcolepsy /anti-obesity/anorexiant Agents"),
            ),
            "OPIOID_ANALGESICS": (
                ("6599", "Opioid Combinations"),
                ("6510", "Opioid Agonists"),
            ),
        }
    )


# -----------------------------------------------------------------------------
# Processor
# -----------------------------------------------------------------------------


class FeatureProcessor:
    def __init__(
        self,
        cohort_path: str,
        enrolid_detail_path: str,
        rx_path: str,
        diag_path: str,
        encounter_path: str,
        redbook_path: str,
        procedure_path: str,
        charlson_path: str,
        *,
        config: Optional[FeatureConfig] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.paths = {
            "cohort": cohort_path,
            "enrolid_detail": enrolid_detail_path,
            "rx": rx_path,
            "diag": diag_path,
            "encounter": encounter_path,
            "redbook": redbook_path,
            "procedure": procedure_path,
            "charlson": charlson_path,
        }
        self.config = config or FeatureConfig()
        self.log = logger or logging.getLogger("FeatureProcessor")
        if not self.log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.log.addHandler(handler)
            self.log.setLevel(logging.INFO)

        self.cohort: pd.DataFrame = pd.DataFrame()
        self.enrolid_detail: pd.DataFrame = pd.DataFrame()
        self.rx: pd.DataFrame = pd.DataFrame()
        self.diag: pd.DataFrame = pd.DataFrame()
        self.encounter: pd.DataFrame = pd.DataFrame()
        self.redbook: pd.DataFrame = pd.DataFrame()
        self.procedure: pd.DataFrame = pd.DataFrame()
        self.charlson: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # IO & Cleaning
    # ------------------------------------------------------------------

    def load_all_datasets(self) -> None:
        cfg = self.config
        self.log.info("Loading phase1 cohort + dimension tables…")

        self.cohort = pd.read_csv(
            self.paths["cohort"],
            parse_dates=[
                "FIRST_RECORD_DATE",
                "OUD_DATE",
                "BUP_START_DATE",
            ],
            dtype={"ENROLID": str},
        )
        self.enrolid_detail = pd.read_csv(
            self.paths["enrolid_detail"],
            parse_dates=["ELIG_SPAN_BEGIN", "ELIG_SPAN_END"],
            dtype={
                "ENROLID": str,
                "DOBYR": "Int64",
                "SEX": "Int64",
                "REGION": "Int64",
                "EGEOLOC": "Int64",
                "PLANTYP": "Int64",
                "EMPREL": "Int64",
                "MSA": "Int64",
            },
        )
        self.rx = pd.read_csv(
            self.paths["rx"],
            parse_dates=["FILLDATE"],
            dtype={
                "ENROLID": str,
                "NDCNUM": str,
                "DRUG_NAME": str,
                "TCGPI_ID": str,
                "SECONDARY_CLASSIFICATION": str,
                "ROOT_CLASSIFICATION": str,
            },
        )
        self.diag = pd.read_csv(
            self.paths["diag"],
            parse_dates=["SVCDATE"],
            dtype={"ENROLID": str, "DIAG_CD": str, "DIAG_NUM": "Int64"},
        )
        self.encounter = pd.read_csv(
            self.paths["encounter"],
            parse_dates=["SVCDATE"],
            dtype={"ENROLID": str, "INOUT_DESCR": str},
        )
        self.redbook = pd.read_csv(
            self.paths["redbook"], dtype={"NDCNUM": str, "MASTFRM": str, "PRODNME": str}
        )
        self.procedure = pd.read_csv(
            self.paths["procedure"], parse_dates=["SVCDATE"], dtype={"ENROLID": str}
        )
        self.charlson = pd.read_csv(
            self.paths["charlson"],
            parse_dates=["SVCDATE"],
            dtype={"ENROLID": str, "CHARLSON_INDEX": float},
        )

        self._clean_blank_strings(
            [
                self.cohort,
                self.enrolid_detail,
                self.rx,
                self.diag,
                self.encounter,
                self.redbook,
                self.procedure,
                self.charlson,
            ]
        )

        self.log.info("Loaded cohort rows: %s", f"{len(self.cohort):,}")

    @staticmethod
    def _clean_blank_strings(dfs: Iterable[pd.DataFrame]) -> None:
        for df in dfs:
            object_cols = df.select_dtypes(include=["object"]).columns
            if len(object_cols):
                df[object_cols] = df[object_cols].replace(r"^\s*$", np.nan, regex=True)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def run_full_pipeline(self) -> pd.DataFrame:
        self.load_all_datasets()
        self.step1_enrich_demographics_and_eligibility()
        self.step2_enrich_comorbidities_diagnoses()
        self.step3_enrich_baseline_charlson_index()
        self.step4_enrich_prior_medication_history()
        self.step5_count_visit_types()

        return self.cohort

    def step1_enrich_demographics_and_eligibility(self) -> None:
        cfg = self.config
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 1] Merging Demographics.\n"
            f"   - Eligibility Check: Must span [{cfg.lookback_days_prebup} days pre-BUP, {cfg.lookforward_days_postbup} days post-BUP].\n"
            f"   - Logic: Retaining longest qualifying span per patient.\n"
            f"   - Classification: MSA=0 -> Rural, MSA>0 -> Urban, MSA=NaN -> Missing.\n"
        )

        self.cohort["ELIG_WINDOW_BEGIN"] = self.cohort["BUP_START_DATE"] - pd.Timedelta(
            days=cfg.lookback_days_prebup
        )
        self.cohort["ELIG_WINDOW_END"] = self.cohort["BUP_START_DATE"] + pd.Timedelta(
            days=cfg.lookforward_days_postbup
        )

        merged = self.enrolid_detail.merge(
            self.cohort[["ENROLID", "ELIG_WINDOW_BEGIN", "ELIG_WINDOW_END"]],
            on="ENROLID",
            how="inner",
        )

        ok = (merged["ELIG_SPAN_BEGIN"] <= merged["ELIG_WINDOW_BEGIN"]) & (
            merged["ELIG_SPAN_END"] >= merged["ELIG_WINDOW_END"]
        )
        detail_matched = merged.loc[ok].copy()

        dup_counts = (
            detail_matched.groupby("ENROLID")
            .size()
            .reset_index(name="QUALIFYING_SPANS")
        )
        multi_span = dup_counts.loc[dup_counts["QUALIFYING_SPANS"] > 1]

        n_multi = len(multi_span)
        if n_multi > 0:
            self.log.warning(
                "- Patients have >1 qualifying eligibility span covering the required window. "
                "Keeping the longest span per patient.",
                n_multi,
            )
        else:
            self.log.info(
                "- No patients have multiple qualifying spans — one span per patient."
            )

        # In case multiple spans match, keep the one with the longest coverage
        detail_matched["SPAN_DAYS"] = (
            detail_matched["ELIG_SPAN_END"] - detail_matched["ELIG_SPAN_BEGIN"]
        ).dt.days
        detail_matched = detail_matched.sort_values(
            ["ENROLID", "SPAN_DAYS"], ascending=[True, False]
        ).drop_duplicates("ENROLID")

        self.log.info(
            "→ %s unique patients retained after deduplication of eligibility spans.",
            f"{detail_matched['ENROLID'].nunique():,}",
        )

        def classify_urban_rural(msa: Optional[float]) -> int:
            if pd.isna(msa):
                return 2  # Missing
            return 0 if int(msa) == 0 else 1  # 0=rural, else urban

        detail_matched["URBAN_RURAL"] = (
            detail_matched["MSA"].apply(classify_urban_rural).astype("Int64")
        )

        keep = [
            "ENROLID",
            "SEX",
            "REGION",
            "EGEOLOC",
            "PLANTYP",
            "EMPREL",
            "URBAN_RURAL",
            "ELIG_SPAN_BEGIN",
            "ELIG_SPAN_END",
        ]
        demo = detail_matched[keep]

        self.cohort = self.cohort.merge(demo, on="ENROLID", how="left")

        def age_band(age: Optional[float]) -> int | np.nan:
            if pd.isna(age):
                return np.nan
            age = int(age)
            for code, lo, hi in cfg.age_bins:
                if hi is None and age >= lo:
                    return code
                if lo <= age <= hi:
                    return code
            return np.nan

        self.cohort["AGE_GROUP"] = (
            self.cohort["AGE_AT_BUP"].apply(age_band).astype("Int64")
        )

        total_pts = len(self.cohort)

        self.log.info("   --- Demographic Breakdown ---")

        ur_map = {0: "Rural", 1: "Urban", 2: "Unknown/Missing"}
        vc_ur = self.cohort["URBAN_RURAL"].value_counts(dropna=False).sort_index()
        for k, v in vc_ur.items():
            lbl = ur_map.get(k, k)
            self.log.info(f"    * {lbl:<15}: {v:,} ({v/total_pts:.1%})")

        # Age Groups
        self.log.info("   --- Age Groups ---")
        age_map = {0: "18-24", 1: "25-34", 2: "35-44", 3: "45-54", 4: "55+"}
        vc_age = self.cohort["AGE_GROUP"].value_counts(dropna=False).sort_index()
        for k, v in vc_age.items():
            lbl = age_map.get(k, f"Grp {k}")
            self.log.info(f"    * {lbl:<15}: {v:,} ({v/total_pts:.1%})")

        # Sex
        self.log.info("   --- Sex  ---")
        sex_map = {1: "Male", 2: "Female"}
        vc_sex = self.cohort["SEX"].value_counts(dropna=False).sort_index()
        for k, v in vc_sex.items():
            lbl = sex_map.get(k, k)
            self.log.info(f"    * {lbl:<10}: {v:,} ({v/total_pts:.1%})")

        self.log.info(f"→ Demographics Merged. Total Cohort: {total_pts:,}\n")

    def step2_enrich_comorbidities_diagnoses(self) -> None:
        cfg = self.config
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 2] Diagnosis Flagging.\n"
            f"   - Window: {cfg.lookback_days_prebup} days prior to BUP Start.\n"
            f"   - Logic: Binary flag if ICD code starts with defined prefixes.\n"
            f"   - Categories Checked: {len(cfg.diagnosis_code_sets)} disease groups.\n"
        )

        diag = self.diag.copy()
        diag["CD"] = diag["DIAG_CD"].astype(str).str.strip().str.upper()

        d = diag.merge(
            self.cohort[["ENROLID", "BUP_START_DATE"]], on="ENROLID", how="inner"
        )
        d["LB_START"] = d["BUP_START_DATE"] - pd.Timedelta(
            days=cfg.lookback_days_prebup
        )
        in_lb = (d["SVCDATE"] < d["BUP_START_DATE"]) & (d["SVCDATE"] >= d["LB_START"])
        d = d.loc[in_lb].copy()

        outputs: List[pd.DataFrame] = []
        for label, prefixes in cfg.diagnosis_code_sets.items():
            mask = d["CD"].str.startswith(prefixes, na=False)
            if not mask.any():
                continue

            sub = d.loc[mask, ["ENROLID", "DIAG_NUM"]].copy()

            any_flag = sub.drop_duplicates("ENROLID")[["ENROLID"]].assign(**{label: 1})
            pri = (
                sub.loc[sub["DIAG_NUM"].eq(1)]
                .drop_duplicates("ENROLID")[["ENROLID"]]
                .assign(**{f"{label}_PRIMARY": 1})
            )
            sec = (
                sub.loc[sub["DIAG_NUM"].gt(1)]
                .drop_duplicates("ENROLID")[["ENROLID"]]
                .assign(**{f"{label}_SECONDARY": 1})
            )
            outputs.extend([any_flag, pri, sec])

        for out in outputs:
            self.cohort = self.cohort.merge(out, on="ENROLID", how="left")

        created_cols = [
            c
            for c in self.cohort.columns
            if any(c.startswith(k) for k in cfg.diagnosis_code_sets.keys())
        ]
        self.cohort[created_cols] = self.cohort[created_cols].fillna(0).astype(int)

        self.log.info("   --- Comorbidity Prevalence (Pre-Index) ---")
        total = len(self.cohort)

        stats = {}
        for label in cfg.diagnosis_code_sets.keys():
            if label in self.cohort.columns:
                stats[label] = self.cohort[label].sum()
            else:
                stats[label] = 0

        sorted_stats = sorted(stats.items(), key=lambda item: item[1], reverse=True)

        for label, count in sorted_stats:
            self.log.info(f"    * {label:<25} : {count:,} ({count/total:.1%})")

        self.log.info(f"→ Diagnosis Processing Complete.\n")

    def step3_enrich_baseline_charlson_index(self) -> None:
        cfg = self.config
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 3] Charlson Comorbidity Index (CCI).\n"
            f"   - Window: {cfg.lookback_days_prebup} days prior to BUP Start.\n"
            f"   - Methodology: 'MAX' recorded index over the lookback period.\n"
        )

        c = self.charlson.merge(
            self.cohort[["ENROLID", "BUP_START_DATE"]], on="ENROLID", how="inner"
        )
        c["LB_START"] = c["BUP_START_DATE"] - pd.Timedelta(
            days=cfg.lookback_days_prebup
        )

        c = c.loc[
            (c["SVCDATE"] < c["BUP_START_DATE"]) & (c["SVCDATE"] >= c["LB_START"])
        ].copy()

        agg = c.groupby("ENROLID")["CHARLSON_INDEX"].max().reset_index()

        self.cohort = self.cohort.merge(agg, on="ENROLID", how="left")
        self.cohort["CHARLSON_INDEX"] = self.cohort["CHARLSON_INDEX"].fillna(0.0)

        desc = self.cohort["CHARLSON_INDEX"].describe(percentiles=[0.25, 0.5, 0.75])
        zero_score = (self.cohort["CHARLSON_INDEX"] == 0).sum()
        total = len(self.cohort)

        self.log.info("   --- CCI Distribution ---")
        self.log.info(
            f"    * Mean (SD)    : {desc['mean']:.2f} (+/- {desc['std']:.2f})"
        )
        self.log.info(
            f"    * Median (IQR) : {desc['50%']:.2f} ({desc['25%']:.2f}-{desc['75%']:.2f})"
        )
        self.log.info(f"    * Max Score    : {desc['max']:.2f}")
        self.log.info(
            f"    * Zero Score   : {zero_score:,} patients ({zero_score/total:.1%}) have CCI=0"
        )

        self.log.info(f"→ CCI Merged.\n")

    def step4_enrich_prior_medication_history(self) -> None:
        cfg = self.config
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 4] Medication History Flagging.\n"
            f"   - Window: {cfg.lookback_days_prebup} days prior to BUP Start.\n"
            f"   - Logic: Matches TCGPI codes (Medi-Span GPI) directly from Rx claims.\n"
        )

        rx = self.rx.merge(
            self.cohort[["ENROLID", "BUP_START_DATE"]], on="ENROLID", how="inner"
        )
        rx["LB_START"] = rx["BUP_START_DATE"] - pd.Timedelta(
            days=cfg.lookback_days_prebup
        )
        rx = rx.loc[
            (rx["FILLDATE"] >= rx["LB_START"]) & (rx["FILLDATE"] < rx["BUP_START_DATE"])
        ].copy()

        rx["TCGPI_PREFIX2"] = rx["TCGPI_ID"].astype(str).str[:2]
        rx["TCGPI_PREFIX4"] = rx["TCGPI_ID"].astype(str).str[:4]
        rx["TCGPI_PREFIX6"] = rx["TCGPI_ID"].astype(str).str[:6]

        rx["SECONDARY_CLASSIFICATION"] = (
            rx["SECONDARY_CLASSIFICATION"]
            .fillna("")
            .astype(str)
            .str.strip()
            .str.upper()
        )
        rx["ROOT_CLASSIFICATION"] = (
            rx["ROOT_CLASSIFICATION"].fillna("").astype(str).str.strip().str.upper()
        )

        def match_stimulants(
            df: pd.DataFrame, prefix: str, classification: str
        ) -> pd.DataFrame:
            if prefix == "6110":
                m = df["TCGPI_PREFIX4"].eq("6110") & (
                    (df["SECONDARY_CLASSIFICATION"] == classification)
                    | (df["ROOT_CLASSIFICATION"] == classification)
                )
            elif prefix == "61":
                m = (
                    df["TCGPI_PREFIX2"].eq("61")
                    & ~df["TCGPI_PREFIX4"].eq("6137")
                    & (
                        (df["SECONDARY_CLASSIFICATION"] == classification)
                        | (df["ROOT_CLASSIFICATION"] == classification)
                    )
                )
            else:
                return pd.DataFrame(columns=["ENROLID"])

            return df.loc[m, ["ENROLID"]].drop_duplicates()

        def match_antipsychotics(
            df: pd.DataFrame, prefix: str, classification: str
        ) -> pd.DataFrame:
            m = (
                df["TCGPI_PREFIX2"].eq(prefix)
                & ~df["TCGPI_PREFIX4"].eq("5950")
                & (
                    df["SECONDARY_CLASSIFICATION"].eq(classification)
                    | df["ROOT_CLASSIFICATION"].eq(classification)
                )
            )
            return df.loc[m, ["ENROLID"]].drop_duplicates()

        def match_generic(
            df: pd.DataFrame, prefix: str, classification: str
        ) -> pd.DataFrame:
            col = (
                "TCGPI_PREFIX2"
                if len(prefix) == 2
                else ("TCGPI_PREFIX4" if len(prefix) == 4 else "TCGPI_PREFIX6")
            )
            m = df[col].eq(prefix) & (
                df["SECONDARY_CLASSIFICATION"].eq(classification)
                | df["ROOT_CLASSIFICATION"].eq(classification)
            )
            return df.loc[m, ["ENROLID"]].drop_duplicates()

        created_cols: List[str] = []
        for label, pairs in cfg.medication_categories.items():
            frames: List[pd.DataFrame] = []
            for prefix, classification in pairs:
                classification_upper = classification.upper()
                if label == "STIMULANTS":
                    frames.append(match_stimulants(rx, prefix, classification_upper))
                elif label == "ANTIPSYCHOTICS":
                    frames.append(
                        match_antipsychotics(rx, prefix, classification_upper)
                    )
                else:
                    frames.append(match_generic(rx, prefix, classification_upper))

            if frames:
                flags = pd.concat(frames, ignore_index=True).drop_duplicates()
                if not flags.empty:
                    flags[label] = 1
                    self.cohort = self.cohort.merge(flags, on="ENROLID", how="left")
                    created_cols.append(label)

        if created_cols:
            self.cohort[created_cols] = self.cohort[created_cols].fillna(0).astype(int)

        self.log.info("   --- Medication Usage (Pre-Index) ---")
        total = len(self.cohort)

        med_stats = {col: self.cohort[col].sum() for col in created_cols}
        sorted_meds = sorted(med_stats.items(), key=lambda x: x[1], reverse=True)

        for label, count in sorted_meds:
            self.log.info(f"    * {label:<25} : {count:,} ({count/total:.1%})")

        self.log.info(f"→ Rx Flags Merged.\n")

    def step5_count_visit_types(self) -> None:
        cfg = self.config
        self.log.info("=" * 80 + "\n")
        self.log.info(
            f"[STEP 5a] Healthcare Utilization Counts.\n"
            f"   - Methodology: Count of UNIQUE service dates per setting (proxy for distinct visits).\n"
        )

        enc = self.encounter.merge(
            self.cohort[["ENROLID", "BUP_START_DATE"]], on="ENROLID", how="inner"
        )
        enc["LB_START"] = enc["BUP_START_DATE"] - pd.Timedelta(
            days=cfg.lookback_days_prebup
        )

        enc = enc.loc[
            (enc["SVCDATE"] >= enc["LB_START"])
            & (enc["SVCDATE"] < enc["BUP_START_DATE"])
        ].copy()
        enc["INOUT_U"] = enc["INOUT_DESCR"].astype(str).str.upper().str.strip()

        labels = {
            "INPATIENT": "INPATIENT_VISIT_COUNT",
            "OUTPATIENT": "OUTPATIENT_VISIT_COUNT",
            "FACILITY": "FACILITY_VISIT_COUNT",
            "EMERGENCY": "ED_VISIT_COUNT",
        }

        created_cols = []
        for key, out_col in labels.items():
            m = enc["INOUT_U"].str.contains(key, case=False, na=False)

            cnt = (
                enc.loc[m]
                .groupby("ENROLID")["SVCDATE"]
                .nunique()
                .reset_index(name=out_col)
            )

            self.cohort = self.cohort.merge(cnt, on="ENROLID", how="left")
            created_cols.append(out_col)

        self.cohort[created_cols] = self.cohort[created_cols].fillna(0).astype(int)

        self.log.info(f"→ Visit Counts Merged (Method: Unique Dates).\n")


# -----------------------------------------------------------------------------

"""SELECTED_FEATURES"""

import pandas as pd

SELECTED_FEATURES = [
    # --- Sociodemographics ---
    "AGE_GROUP",
    "SEX",
    "REGION",
    "URBAN_RURAL",
    "PLANTYP",
    "EMPREL",
    "EGEOLOC",
    # --- Charlson comorbidity ---
    "CHARLSON_INDEX",
    # --- Comorbidities ---
    "CHRONIC_PAIN",
    "CHRONIC_PAIN_PRIMARY",
    "CHRONIC_PAIN_SECONDARY",
    "HIV_AIDS",
    "HIV_AIDS_PRIMARY",
    "HIV_AIDS_SECONDARY",
    "HEPATITIS_C",
    "HEPATITIS_C_PRIMARY",
    "HEPATITIS_C_SECONDARY",
    "DEPRESSIVE_DISORDER",
    "DEPRESSIVE_DISORDER_PRIMARY",
    "DEPRESSIVE_DISORDER_SECONDARY",
    "ANXIETY",
    "ANXIETY_PRIMARY",
    "ANXIETY_SECONDARY",
    "PTSD",
    "PTSD_PRIMARY",
    "PTSD_SECONDARY",
    "BIPOLAR_DISORDER",
    "BIPOLAR_DISORDER_PRIMARY",
    "BIPOLAR_DISORDER_SECONDARY",
    "SCHIZOPHRENIA",
    "SCHIZOPHRENIA_PRIMARY",
    "SCHIZOPHRENIA_SECONDARY",
    # --- Substance use disorders ---
    "NON_OPIOID_SUD",
    "NON_OPIOID_SUD_PRIMARY",
    "NON_OPIOID_SUD_SECONDARY",
    "ALCOHOL_USE_DISORDER",
    "ALCOHOL_USE_DISORDER_PRIMARY",
    "ALCOHOL_USE_DISORDER_SECONDARY",
    "OUD_PRIMARY",
    "OUD_SECONDARY",
    "OUD_MILD",
    "OUD_MILD_PRIMARY",
    "OUD_MILD_SECONDARY",
    "OUD_MODSEV",
    "OUD_MODSEV_PRIMARY",
    "OUD_MODSEV_SECONDARY",
    "CUD_MILD",
    "CUD_MILD_PRIMARY",
    "CUD_MILD_SECONDARY",
    "CUD_MODSEV",
    "CUD_MODSEV_PRIMARY",
    "CUD_MODSEV_SECONDARY",
    # --- Health Services ---
    "INPATIENT_VISIT_COUNT",
    "OUTPATIENT_VISIT_COUNT",
    "FACILITY_VISIT_COUNT",
    # 'ED_VISIT_COUNT',
    # --- Pre-index medication use ---
    "ANTIDEPRESSANTS",
    "ANTIPSYCHOTICS",
    "MOOD_STABILIZERS",
    "BENZODIAZEPINES",
    "NONBENZODIAZEPINE",
    "STIMULANTS",
    "OPIOID_ANALGESICS",
    # --- Buprenorphine treatment features ---
    "OUD_OFFSET_DAYS",
    "INITIAL_DAYSUPP",
    "INITIAL_DOSE_GROUP",
    # --- Adherence & outcomes ---
    "PDC_30",
    "PDC30_CAT",
    "BUP_STATUS_NUM",
]

# --- Input/output paths ---
input_file = "bup_pipeline_phase2.csv"
output_file = "bup_features.xlsx"

REMOVE_PRIMARY_SECONDARY = False
filtered_features = [
    f
    for f in SELECTED_FEATURES
    if (not REMOVE_PRIMARY_SECONDARY)
    or not (f.endswith("_PRIMARY") or f.endswith("_SECONDARY"))
]

print(f"Original feature count: {len(SELECTED_FEATURES)}")
print(f"Filtered feature count: {len(filtered_features)}")

# --- Read the dataset ---
df = pd.read_csv(input_file)

# --- Validate feature list against dataset columns ---
existing_features = [col for col in filtered_features if col in df.columns]
missing_features = [col for col in filtered_features if col not in df.columns]

print(f"Columns found in dataset: {len(existing_features)}")
print(f"Columns missing from dataset: {len(missing_features)}")

# --- Subset the dataset ---
df_filtered = df[existing_features]

# --- Save to Excel ---
df_filtered.to_excel(output_file, index=False)

print(f"Filtered dataset saved to: {output_file}")
