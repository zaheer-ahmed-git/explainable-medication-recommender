- **MIMIC-IV v3.1** → `Dataset/mimiciv/3.1/hosp/`
- **eICU-CRD v2.0** → `Dataset/eicu-crd/2.0/`

**General rules**

- **MIMIC timestamps** (`admittime`, `charttime`, etc.) are **shifted/de-identified datetimes** — treat as relative time, not real calendar dates.
- **eICU `offset` fields** are **integers = minutes from ICU admission** for that `patientunitstayid` (can be negative if recorded before ICU admit).
- **MIMIC dose fields** are often stored as **text**, not pure numbers, because values can include ranges, units, or free text.

---

## MIMIC-IV `hosp`

### `admissions.csv.gz` (~546k rows)

| Column | Type | Detail |
| --- | --- | --- |
| `subject_id` | **INTEGER** | De-identified patient key. 8-digit integer. ~223k distinct patients. Never null. |
| `hadm_id` | **INTEGER** | De-identified hospital admission key. One row per admission. Never null. |
| `admittime` | **TIMESTAMP** | Hospital admission time (shifted). Format `YYYY-MM-DD HH:MM:SS`. Never null. |
| `dischtime` | **TIMESTAMP** | Discharge time (shifted). Usually after `admittime`. Never null. |
| `deathtime` | **TIMESTAMP** | In-hospital death time. **~98% null** (only set when patient died in hospital). |
| `admission_type` | **Categorical text** | Admission class. Values include `EW EMER.`, `EU OBSERVATION`, `ELECTIVE`, `URGENT`, etc. (9 categories). |
| `admit_provider_id` | **VARCHAR ID** | 6-char de-identified provider token. Links to `provider.provider_id`. |
| `admission_location` | **Categorical text** | Where patient came from: `EMERGENCY ROOM`, `PHYSICIAN REFERRAL`, `TRANSFER FROM HOSPITAL`, etc. (11 values). |
| `discharge_location` | **Categorical text** | Discharge destination: `HOME`, `SKILLED NURSING FACILITY`, `DIED`, `HOSPICE`, etc. **~27% null** (e.g. in-hospital deaths). |
| `insurance` | **Categorical text** | `Medicare`, `Private`, `Medicaid`, `Other`, `No charge`. **~2% null**. |
| `language` | **Categorical text** | Preferred language; `English` dominates. ~25 distinct values. |
| `marital_status` | **Categorical text** | `MARRIED`, `SINGLE`, `WIDOWED`, `DIVORCED`. **~2.5% null**. |
| `race` | **Categorical text** | Race/ethnicity category (33 values): `WHITE`, `BLACK/AFRICAN AMERICAN`, `HISPANIC OR LATINO`, etc. |
| `edregtime` | **TIMESTAMP** | ED registration time. **~30% null** (not all admissions via ED). |
| `edouttime` | **TIMESTAMP** | ED departure time. **~30% null**. |
| `hospital_expire_flag` | **INTEGER (0/1)** | `1` = died before discharge, `0` = survived. Binary outcome flag. |

---

### `diagnoses_icd.csv.gz` (~6.36M rows)

| Column | Type | Detail |
| --- | --- | --- |
| `subject_id` | **INTEGER** | Patient identifier. |
| `hadm_id` | **INTEGER** | Admission identifier. Multiple diagnosis rows per admission. |
| `seq_num` | **INTEGER** | Diagnosis priority/sequence. `1` = principal diagnosis; ranges 1–39. |
| `icd_code` | **VARCHAR code** | ICD diagnosis code (3–7 chars). ~28.5k distinct codes. No long titles here — join `d_icd_diagnoses`. |
| `icd_version` | **INTEGER** | `9` = ICD-9, `10` = ICD-10. Admissions after ~2015 are mostly ICD-10. |

---

### `prescriptions.csv.gz` (very large; ~20M+ rows)

| Column | Type | Detail |
| --- | --- | --- |
| `subject_id` | **INTEGER** | Patient identifier. |
| `hadm_id` | **INTEGER** | Admission identifier. |
| `pharmacy_id` | **VARCHAR/numeric ID** | Pharmacy system order ID (4–8 digit token). |
| `poe_id` | **VARCHAR ID** | Provider order entry ID (10–13 chars). Links to `poe` table. **~1% null**. |
| `poe_seq` | **INTEGER** | Sequence within POE order. **~1% null** (when `poe_id` missing). |
| `order_provider_id` | **VARCHAR ID** | Ordering clinician. 6-char provider token. **~0.4% null**. |
| `starttime` | **TIMESTAMP** | Prescription start (shifted). |
| `stoptime` | **TIMESTAMP** | Prescription end (shifted). |
| `drug_type` | **Categorical text** | `MAIN` (primary drug), `BASE` (carrier/base), `ADDITIVE` (IV additive). |
| `drug` | **Free text** | Medication name as ordered (2–67 chars). Hospital formulary wording. |
| `formulary_drug_cd` | **VARCHAR code** | Internal formulary code. |
| `gsn` | **VARCHAR code** | Generic Sequence Number (drug coding). **~12% null**. |
| `ndc` | **VARCHAR/INTEGER code** | National Drug Code (up to 11 digits). Product identifier. |
| `prod_strength` | **Free text** | Strength string, e.g. `500mg`, `10mg/5mL`. |
| `form_rx` | **Categorical text** | Rx form code. **~99% null** (rarely populated). |
| `dose_val_rx` | **VARCHAR (mixed)** | Prescribed dose value — often numeric but stored as text (`1`, `0.5`, `1-2`). |
| `dose_unit_rx` | **VARCHAR** | Dose unit: `mg`, `mL`, `units`, etc. |
| `form_val_disp` | **VARCHAR (mixed)** | Dispense quantity value. |
| `form_unit_disp` | **VARCHAR** | Dispense unit: `TAB`, `VIAL`, `SYR`, etc. |
| `doses_per_24_hrs` | **INTEGER** | Scheduled doses per day (0–24). **~39% null**. |
| `route` | **Categorical text** | `IV`, `PO`, `PO/NG`, `SC`, `IM`, `PR`, `IV DRIP`, etc. |

These are **orders/prescriptions**, not proof the drug was actually given.

---

### `emar.csv.gz` (very large; eMAR administration events)

| Column | Type | Detail |
| --- | --- | --- |
| `subject_id` | **INTEGER** | Patient identifier. |
| `hadm_id` | **INTEGER** | Admission ID. **~2% null** (some events not tied to a specific admission). |
| `emar_id` | **VARCHAR ID** | Unique eMAR event identifier (10–13 chars). |
| `emar_seq` | **INTEGER** | Line number within the eMAR event (one event can have multiple rows). |
| `poe_id` | **VARCHAR ID** | Link back to provider order. |
| `pharmacy_id` | **VARCHAR ID** | Link to pharmacy order. **~19% null**. |
| `enter_provider_id` | **VARCHAR ID** | Nurse/provider who charted. **~86% null**. |
| `charttime` | **TIMESTAMP** | When administration was charted (shifted). |
| `medication` | **Free text** | Medication name at administration time. **~5% null**. |
| `event_txt` | **Categorical text** | Event type: `Administered`, `Not Given`, `Flushed`, `Started`, `Stopped`, `Confirmed`, etc. |
| `scheduletime` | **TIMESTAMP** | Scheduled administration time. **~0.1% null**. |
| `storetime` | **TIMESTAMP** | When record was saved in the system. |

High-level **“was it given?”** events. Join to `emar_detail` for dose/route/infusion specifics.

---

### `emar_detail.csv.gz` (very large; sparse detail rows)

Most columns are **only filled for certain administration types** (IV infusions, insulin, patches, etc.), so null rates are high.

| Column | Type | Detail |
| --- | --- | --- |
| `subject_id` | **INTEGER** | Patient identifier. |
| `emar_id` | **VARCHAR ID** | Join key to `emar`. |
| `emar_seq` | **INTEGER** | Join key to `emar`. |
| `parent_field_ordinal` | **NUMERIC** | Field ordering within a structured eMAR form (e.g. `1.1`, `2.2`). **~49% null**. |
| `administration_type` | **Categorical text** | e.g. `IV Infusion`, `Standard Maintenance Medication`, `Insulin SC Sliding Scale`, `PCA`. **~51% null**. |
| `pharmacy_id` | **VARCHAR ID** | Pharmacy order link. **~53% null**. |
| `barcode_type` | **Categorical text** | Barcode scan type: `if`, `iv`, `bc`, `tpn`. **~51% null**. |
| `reason_for_no_barcode` | **Free text** | Why barcode wasn't scanned. **~98% null**. |
| `complete_dose_not_given` | **VARCHAR (Y/N-like)** | Partial dose indicator. **~87% null**. |
| `dose_due` / `dose_due_unit` | **VARCHAR (mixed)** | Scheduled dose and unit. **~52% null**. |
| `dose_given` / `dose_given_unit` | **VARCHAR (mixed)** | Actual dose given and unit. **~53% null**. |
| `will_remainder_of_dose_be_given` | **VARCHAR (Y/N-like)** | Remainder flag. **~83% null**. |
| `product_amount_given` | **NUMERIC** | Product amount administered. **~56% null**. |
| `product_unit` | **VARCHAR** | Unit for product amount (`mg`, `mL`, etc.). **~54% null**. |
| `product_code` | **VARCHAR code** | Scanned product code. **~51% null**. |
| `product_description` | **Free text** | Product label text. **~59% null**. |
| `product_description_other` | **Free text** | Alternate product description. **~97% null**. |
| `prior_infusion_rate` | **NUMERIC** | Previous IV rate. **~98% null** (infusions only). |
| `infusion_rate` | **NUMERIC** | Current IV rate (e.g. mL/hr). **~97% null**. |
| `infusion_rate_adjustment` | **Free text** | Rate change description. **~97% null**. |
| `infusion_rate_adjustment_amount` | **NUMERIC** | Rate change amount. **~99.9% null**. |
| `infusion_rate_unit` | **VARCHAR** | Infusion rate unit. **~96% null**. |
| `route` | **Categorical text** | `PO`, `NG`, `SC`, `IM`, `PR`. **~84% null**. |
| `infusion_complete` | **VARCHAR (0/1)** | Infusion completion flag. **~99.5% null**. |
| `completion_interval` | **VARCHAR** | Time to completion. **~99.8% null**. |
| `new_iv_bag_hung` | **VARCHAR (0/1)** | New bag flag. **~99.4% null**. |
| `continued_infusion_in_other_location` | **VARCHAR (0/1)** | Continued elsewhere. **~100% null** in sample. |
| `restart_interval` | **VARCHAR** | Restart timing. **~99.9% null**. |
| `side` | **Categorical text** | Body side (`Left`, `Right`). **~99.7% null**. |
| `site` | **VARCHAR** | Injection/administration site. **~99.7% null**. |
| `non_formulary_visual_verification` | **VARCHAR (0/1)** | Non-formulary verification. **~99.9% null**. |

---

### `provider.csv.gz` (~42k rows)

| Column | Type | Detail |
| --- | --- | --- |
| `provider_id` | **VARCHAR ID** | De-identified 6-character provider token. In MIMIC-IV v3.1 this table intentionally has **no names or specialties** — only the ID list. Referenced from `admit_provider_id`, `order_provider_id`, `enter_provider_id`, etc. |

---

## eICU-CRD v2.0

### `diagnosis.csv.gz`

| Column | Type | Detail |
| --- | --- | --- |
| `diagnosisid` | **INTEGER** | Unique row identifier. |
| `patientunitstayid` | **INTEGER** | ICU unit stay identifier (primary join key). |
| `activeupondischarge` | **BOOLEAN text** | `True` / `False` — diagnosis still active at discharge. |
| `diagnosisoffset` | **INTEGER (minutes)** | Minutes from ICU admission when diagnosis recorded. Can be negative (pre-ICU). Range in sample: about -1,526 to +153,813. |
| `diagnosisstring` | **Hierarchical text** | Pipe-delimited clinical path, e.g. `cardiovascular|shock|septic`. 24–146 chars. Primary semantic diagnosis field. |
| `icd9code` | **VARCHAR code** | ICD-9 code when mapped. Can be multiple codes separated by commas. **~25% null**. |
| `diagnosispriority` | **Categorical text** | `Primary`, `Major`, `Other`. |

---

### `treatment.csv.gz`

| Column | Type | Detail |
| --- | --- | --- |
| `treatmentid` | **INTEGER** | Unique row identifier. |
| `patientunitstayid` | **INTEGER** | ICU stay identifier. |
| `treatmentoffset` | **INTEGER (minutes)** | Minutes from ICU admission. Can be negative. |
| `treatmentstring` | **Hierarchical text** | Pipe-delimited treatment path (not only drugs): ventilation, dialysis, lines, etc. |
| `activeupondischarge` | **BOOLEAN text** | `True` / `False`. |

---

### `infusionDrug.csv.gz`

| Column | Type | Detail |
| --- | --- | --- |
| `infusiondrugid` | **INTEGER** | Unique row identifier. |
| `patientunitstayid` | **INTEGER** | ICU stay identifier. |
| `infusionoffset` | **INTEGER (minutes)** | Time of infusion record relative to ICU admission. |
| `drugname` | **Free text** | Infusion drug name (3–134 chars). Continuous IV meds (vasopressors, sedatives, etc.). |
| `drugrate` | **VARCHAR (mixed)** | Documented drug rate string. **~0.2% null**. |
| `infusionrate` | **NUMERIC** | Pump infusion rate (units vary by drug). **~14% null**. |
| `drugamount` | **NUMERIC** | Drug amount in bag/syringe (0.01–500,000 in sample). **~17% null**. |
| `volumeoffluid` | **NUMERIC** | Fluid volume (mL). **~17% null**. |
| `patientweight` | **NUMERIC** | Weight (kg) at time of record. **~79% null**. |

---

### `medication.csv.gz`

| Column | Type | Detail |
| --- | --- | --- |
| `medicationid` | **INTEGER** | Unique row identifier. |
| `patientunitstayid` | **INTEGER** | ICU stay identifier. |
| `drugorderoffset` | **INTEGER (minutes)** | When order was placed. |
| `drugstartoffset` | **INTEGER (minutes)** | When administration started (can differ from order time). |
| `drugivadmixture` | **BOOLEAN text** | `Yes` / `No` — IV admixture order. |
| `drugordercancelled` | **BOOLEAN text** | `Yes` / `No` — order cancelled before/at charting. |
| `drugname` | **Free text** | Medication name (12–74 chars). **~17% null**. |
| `drughiclseqno` | **INTEGER code** | HICL sequence number (hospital drug coding). **~37% null**. |
| `dosage` | **VARCHAR (mixed)** | Dosage string (not always parseable as a single number). **~13% null**. |
| `routeadmin` | **Categorical text** | Route: `Intravenous`, `Oral`, etc. |
| `frequency` | **Categorical text** | Dosing frequency: `Once`, `Q6H`, `Continuous`, etc. **~1.5% null**. |
| `loadingdose` | **BOOLEAN text** | Loading dose flag. **~100% null** in sample (rarely used). |
| `prn` | **BOOLEAN text** | `Yes` / `No` — as-needed medication. |
| `drugstopoffset` | **INTEGER (minutes)** | When medication stopped. |
| `gtc` | **INTEGER code** | Generic Therapeutic Class code (0–99). Drug classification. |

---

## Practical notes for your project

| Use case | MIMIC-IV | eICU-CRD |
| --- | --- | --- |
| Stay key | `hadm_id` (+ `stay_id` in ICU module) | `patientunitstayid` |
| Diagnoses | `diagnoses_icd` + `d_icd_diagnoses` | `diagnosis` (`diagnosisstring`, `icd9code`) |
| Med orders | `prescriptions`, `pharmacy`, `poe` | `medication` |
| Med given | `emar` + `emar_detail` | `medication` + `infusionDrug` |
| Treatments/procedures | `procedures_icd`, ICU events | `treatment` |

**Parsing cautions**

- Treat MIMIC `dose_val_rx`, `dose_given`, eICU `dosage`, `drugrate` as **semi-structured text**, not clean floats.
- `emar_detail` is **event-type sparse** — filter by `administration_type` before expecting dose columns to be populated.
- eICU `diagnosisstring` / `treatmentstring` need **path parsing** (split on `\` or `|` depending on export).
- Negative offsets in eICU are valid (events documented around ICU admit boundary).