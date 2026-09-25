# ML Challenge 2026 — Business Entity Resolution Dataset: Complete Analysis

> **Challenge:** Given business records from 3 independent data sources with noisy and inconsistent fields, determine which records across sources refer to the same real-world business entity.

---

## 1. Overview & Context

| Property | Details |
|---|---|
| **Challenge Name** | ML Challenge 2026 — Business Entity Resolution |
| **Problem Type** | Entity Resolution / Record Linkage |
| **Domain** | Business identity data from commercial platforms |
| **File Format** | Tab-Separated Values (`.tsv`) — **NOT** CSV |
| **Encoding** | UTF-8 |
| **Evaluation Metric** | F₀.₅ Score (precision-heavy, macro-averaged per S1 entity) |
| **Model Constraint** | MIT/Apache 2.0 licensed, up to 8B parameters |

### Core Objective

Source 1 is the **deduplicated reference source**. The task is to find all matching records from Source 2 and Source 3 for each Source 1 entity. An S1 entity may match **zero, one, or many** records from S2/S3.

---

## 2. File Inventory & Sizes

### 2.1 Directory Structure

```
student_resource/
├── README.md                         (14,130 bytes — main problem statement)
├── Documentation_template.md         (2,175 bytes — methodology write-up template)
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv         (210.1 MB)
│   │   ├── train_source2.tsv         (489.3 MB)
│   │   ├── train_source3.tsv         (503.7 MB)
│   │   └── train_ground_truth.tsv    (127.0 MB)
│   └── test/
│       ├── test_source1.tsv          (175.0 MB)
│       ├── test_source2.tsv          (509.5 MB)
│       └── test_source3.tsv          (506.0 MB)
└── utils/
    └── validate_submission.py        (13,687 bytes — submission validator)
```

**Total dataset size: ~2.5 GB**

### 2.2 Row Counts

| File | Rows (excl. header) | Unique Entity IDs |
|---|---|---|
| `train_source1.tsv` | ~2,206,822 | ~2,206,822 |
| `train_source2.tsv` | ~5,034,617 | ~5,034,617 |
| `train_source3.tsv` | ~5,285,604 | ~5,285,604 |
| `train_ground_truth.tsv` | ~2,206,822 | ~2,206,822 (1:1 with S1) |
| `test_source1.tsv` | ~1,732,545 | ~1,732,545 |
| `test_source2.tsv` | ~4,887,274 | ~4,887,274 |
| `test_source3.tsv` | ~5,082,317 | ~5,082,317 |

**Total records: ~26.4 million across all files**

> [!NOTE]
> Source 2 and Source 3 are each roughly 2.3-2.4x the size of Source 1. These are **not** deduplicated — the same real-world business can have multiple noisy records within S2 or S3.

---

## 3. Schema & Columns

All source files (train and test) share the **exact same 4 columns**:

| Column | Type | Description |
|---|---|---|
| `entity_id` | String | Unique ID with source prefix (`S1-`, `S2-`, `S3-`) + numeric suffix |
| `business_name` | String | Name of the business (noisy, multilingual) |
| `business_address` | String | Address of the business (highly variable format) |
| `country` | String | Country label: `US`, `India`, or `France` (test only) |

### Ground Truth Schema (`train_ground_truth.tsv`)

| Column | Type | Description |
|---|---|---|
| `source1_entity_id` | String | An `S1-` entity ID |
| `matched_entity_ids` | String | Comma-separated list of matching `S2-`/`S3-` IDs (empty = singleton) |

---

## 4. Entity ID Format

| Source | Prefix | Example IDs |
|---|---|---|
| Source 1 | `S1-` | `S1-925783039`, `S1-773889195`, `S1-377745466` |
| Source 2 | `S2-` | `S2-166376419`, `S2-764573417`, `S2-639257739` |
| Source 3 | `S3-` | `S3-202863386`, `S3-859268022`, `S3-22467283` |

- IDs are **unique within each source**
- The numeric suffix appears to be a randomly assigned integer (up to 9 digits)
- There is **no shared identifier** across sources — that's the whole challenge

---

## 5. Countries & Geographic Distribution

### 5.1 Training Data Countries

| Country | Present In |
|---|---|
| `US` | Source 1, Source 2, Source 3 |
| `India` | Source 1, Source 2, Source 3 |

### 5.2 Test Data Countries

| Country | Present In |
|---|---|
| `US` | Source 1, Source 2, Source 3 |
| `India` | Source 1, Source 2, Source 3 |
| **`France`** | **Source 1, Source 2, Source 3** *(NEW — not in training data!)* |

> [!WARNING]
> **France is a zero-shot country** — it appears ONLY in the test set. Your model must generalize to French business names, French addresses, and French legal entity types without any training examples. Do NOT hard-code your pipeline to `{US, India}`.

### 5.3 Approximate Country Split (from data samples)

The data is roughly split between US and India in training, with France added for the test set. Each country appears in **all three sources** (S1, S2, S3).

---

## 6. Languages & Writing Scripts

This dataset is **heavily multilingual**. The following languages and writing systems are present:

### 6.1 Languages Detected

| Language | Script | Where Found | Examples |
|---|---|---|---|
| **English** | Latin | All sources, US/India/France | Most business names and US addresses |
| **Hindi** | Devanagari | S2, S3 (India records) | `राम मार्केटिंग प्राइवेट लिमिटेड` |
| **Marathi** | Devanagari | S2, S3 (Maharashtra) | `महाराष्ट्र` in addresses |
| **Tamil** | Tamil script | S2, S3 (Tamil Nadu) | `குளோபல் பிசினஸ் பிரைவேட் லிமிடெட்`, `தமிழ்நாடு` |
| **Telugu** | Telugu script | S2, S3 (Telangana/AP) | `కృష్ణా ఇంపెక్స్ లిమిటెడ్` |
| **Kannada** | Kannada script | S2, S3 (Karnataka) | `ಡಿಜಿಟಲ್ ಬಿಲ್ಡರ್ಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`, `ಕರ್ನಾಟಕ` |
| **Malayalam** | Malayalam script | S2, S3 (Kerala) | `സിൽവർ കൺസൾട്ടൻസി പ്രൈവറ്റ് ലിമിറ്റഡ്` |
| **Gujarati** | Gujarati script | S2, S3 (Gujarat) | `ક્રિએટિવ ટેક્નોલોજી પ્રા. લિ.`, `ગુજરાત` |
| **Punjabi** | Gurmukhi script | S2, S3 (Punjab) | `ਸਕਾਈ ਅਰਿਹੰਤ ਗਲੋਬਲ ਪ੍ਰਾ. ਲਿ.`, `ਪੰਜਾਬ` |
| **Bengali/Bangla** | Bengali script | S2, S3 (West Bengal) | `ইনোভেটিভ প্রোডাক্টস রেস্টুরেন্ট লিমিটেড`, `পশ্চিমবঙ্গ` |
| **Odia** | Odia script | S2, S3 (Odisha) | `ଶ୍ୟାମ ଫୁଡ୍ସ୍ ଲିମିଟେଡ୍` |
| **French** | Latin Extended | S1, S2, S3 (France, test only) | `Developpement Ecole`, `Securite Darts Sport SARL` |

### 6.2 Script Mixing Patterns

A critical noise pattern is **mixed-script business names**, where part of the name is in a native Indian script and part is in English/Latin. Examples:

- Full native script: `అర్బన్ ఎనర్జీ ప్రైవేట్ లిమిటెడ్` (Telugu)
- English + Hindi: `Sun पावर Provision`
- Telugu + English + Telugu: `గుజరాత్ Logistics లిమిటెడ్`
- Tamil + English: `அரிஹந்த் Foundation Private Limited`

> [!IMPORTANT]
> Source 1 business names are **predominantly English** across all countries. Sources 2 and 3 introduce **native-script variants** (Devanagari, Tamil, Telugu, etc.) of the same businesses. This is a key matching challenge — the same company name may appear in English in S1 but in Hindi/Tamil/etc. in S2 or S3.

### 6.3 Address Script Mixing

State names in Indian addresses can appear in either English or native script:

| English | Native Script |
|---|---|
| Maharashtra | महाराष्ट्र |
| Tamil Nadu | தமிழ்நாடு |
| Karnataka | ಕರ್ನಾಟಕ |
| Delhi | दिल्ली |
| Uttar Pradesh | उत्तर प्रदेश |
| West Bengal | পশ্চিমবঙ্গ |
| Gujarat | ગુજરાત |
| Punjab | ਪੰਜਾਬ |
| Haryana | हरियाणा |
| Rajasthan | राजस्थान |
| Madhya Pradesh | मध्य प्रदेश |

---

## 7. Business Name Abbreviations & Legal Suffixes

### 7.1 US Legal Entity Types

| Abbreviation | Full Form | Examples |
|---|---|---|
| `LLC` | Limited Liability Company | `Custom Wealth Services LLC` |
| `L.L.C.` | Limited Liability Company (dotted) | `Edwards Cornerstone Flexible L.L.C.` |
| `Inc` / `Inc.` | Incorporated | `B+ Retail Inc`, `George Saul Inc` |
| `Corp` / `Corp.` | Corporation | `Vision Partners Corp` |
| `Corporation` | Corporation (full) | same entity may use either form |
| `PC` / `P.C.` | Professional Corporation | `Crystal Lending PC` |
| `PLLC` | Professional LLC | `Center Grand PLLC` |
| `LP` / `L.P.` | Limited Partnership | `Crestline Clean LP` |
| `Co` / `Co.` | Company | `Bay Bdc Co` |
| `DBA` / `dba` | Doing Business As | `Ectolumdrex dba X+ Madison Inc` |
| `Associates` | Associates | `Foot and Ankle Care Associates` |

### 7.2 Indian Legal Entity Types

| Abbreviation | Full Form | Variations Found |
|---|---|---|
| `Pvt` / `Pvt.` | Private | `Pvt Ltd`, `Pvt. Ltd.`, `Private Limited` |
| `Ltd` / `Ltd.` | Limited | `Ltd`, `Ltd.`, `Limited` |
| `Private Limited` | Private Limited | Full form |
| `LLP` / `L.L.P.` | Limited Liability Partnership | `Core Welfare Society LLP` |
| `Public Limited` | Public Limited Company | `Palar Projects Public Limited` |
| `M/s` | Messrs (partnership prefix) | `M/s Sandeep Software (India) Pvt. Ltd` |
| `प्राइवेट लिमिटेड` | Private Limited (Hindi) | Hindi transliteration |
| `பிரைவேட் லிமிடெட்` | Private Limited (Tamil) | Tamil transliteration |
| `ప్రైవేట్ లిమిటెడ్` | Private Limited (Telugu) | Telugu transliteration |
| `ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | Private Limited (Kannada) | Kannada transliteration |
| `പ്രൈവറ്റ് ലിമിറ്റഡ്` | Private Limited (Malayalam) | Malayalam transliteration |
| `প্রাইভেট লিমিটেড` | Private Limited (Bengali) | Bengali transliteration |
| `પ્રાઇવેટ લિમિટેડ` | Private Limited (Gujarati) | Gujarati transliteration |
| `ਪ੍ਰਾ. ਲਿ.` | Private Limited (Punjabi abbrev.) | Gurmukhi transliteration |
| `प्रा. लि.` | Private Limited (Hindi abbrev.) | Hindi abbreviated form |

### 7.3 French Legal Entity Types (Test Only)

| Abbreviation | Full Form | Examples |
|---|---|---|
| `SARL` / `S.A.R.L.` | Societe a Responsabilite Limitee | `ZNB Club SARL` |
| `SAS` / `S.A.S.` | Societe par Actions Simplifiee | `Fractales Amis Groupe S.A.S` |
| `SASU` | SAS Unipersonnelle | `Thermal and Fils SASU` |
| `SA` / `S.A.` | Societe Anonyme | `Europ and Freres Distribution S.A.` |
| `EURL` | Entreprise Unipersonnelle a Responsabilite Limitee | `Developpement Ecole (France) EURL` |
| `SCI` | Societe Civile Immobiliere | `SCI Ptit Amicale`, `Engages Art Pharmacie SCI` |
| `SNC` | Societe en Nom Collectif | -- |
| `EI` | Entreprise Individuelle | `Vegan Groupe EI` |
| `Societe` | Company (generic) | `Saint-Herblain Societe SARL` |
| `Association` | Association (non-profit) | `Association de Pena` |
| `Fils` | Sons (and Sons) | `Grain and Fils`, `Thermal and Fils SASU` |
| `Freres` | Brothers | `Europ and Freres Distribution S.A.` |
| `Maison` | House / Establishment | `Maison de Sante Generation` |
| `Comite` | Committee | `Memorial Comite`, `Artisanal Comite SARL` |
| `Groupe` | Group | `Fractales Amis Groupe S.A.S` |
| `Clinique` | Clinic | `Clinique Saint Francois` |
| `College` | College/School | `College Saint (The)` |
| `Ecole` / `Ecole` | School | `Team Ecole`, `Developpement Ecole` |
| `Ligue` | League | `sci ligue ici parents` |
| `Amicale` | Friendly society | `Amicale des bonheur` |
| `Sportive` | Sports (adj.) | `IFW Sportive SARL` |

### 7.4 Common Business Name Patterns

| Pattern | Examples |
|---|---|
| Ampersand / "and" interchangeable | `Foot & Ankle` vs `Foot and Ankle` |
| Comma-separated partners | `Schaefer, Michael and Silver Associates` |
| Prefixed with `#` or `--` | `#centraleducation`, `-- Holloway Peak Inc Seafood` |
| Wrapped in brackets | `[INCORPORATED]`, `[LLC]`, `[TECHNOLOGIES]` |
| Contains URLs/domains | `www.shivshakti.com`, `heassociates.com`, `cuttackintlprivate.com` |
| Pipe separator | `SHIVSHAKTI VIDYALAYA VIDYALAYA OVERSEAS CORPORATION \| www.shivshakti.com` |
| Contains registration numbers | `MW Management Private Limited - 2067865001` |
| `formerly` / DBA aliases | `Vantagebrixdelta formerly Johnson Royal Metal Works` |
| Gibberish/garbled names | `XYL0TAVOQUO`, `Dovaxylo`, `Ectolumdrex`, `Xebravorix`, `Vantageavigild` |

---

## 8. Address Format Variations

### 8.1 US Address Patterns

| Pattern | Example |
|---|---|
| **Standard format** | `1795 Westchester Drive, High Point, NC` |
| **With unit/apt** | `2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD` |
| **State-first / reordered** | `OH, Columbus, 5559 Orville Avenue` |
| **PO Box** | `1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio` |
| **PMB** | `8517 DEWBERRY WAY, PMB 7952, ELK GROVE, CA` |
| **State as abbreviation** | `NC`, `CA`, `TX`, `OH` |
| **State as full name** | `North Carolina`, `California`, `Texas` |
| **Fractional addresses** | `19 1/2 STARDUST TRAIL`, `259 1/2 CAVALIER WAY` |
| **NULL placeholder** | `067 PRODUCTION CT, NULL, INDEPENDENCE, KY` |
| **N/A placeholder** | `CEDARBROOK ROAD, N/A, NAPERVILLE CITY, IL` |
| **City suffixes** | `JOHNSON CITY`, `HALTOM CITY`, `SALT LAKE CITY` |
| **Township/CDP** | `Town Of Rubicon`, `PLEASANT TWP`, `MORRISON CDP` |
| **Double hash** | `##2326 341ND PLACE`, `##8 Willow Oak Lane` |
| **Floor notation** | `Fl 0`, `Fl. 1`, `Fl 1` |

### 8.2 Indian Address Patterns

| Pattern | Example |
|---|---|
| **H.No / House No** | `H.No.16-11-23/37/A`, `House No-777` |
| **Plot / Flat / Door** | `Plot No.53`, `Flat No.402`, `Door No 180-B` |
| **Floor references** | `2Nd Floor`, `3Rd Floor`, `Ground Floor`, `Ist Floor` |
| **Building/Complex** | `Tower 1, Oakwood`, `Sakar 5 B/H Natraj Cinema` |
| **Landmark-based** | `Near SBI ATM`, `Near Fortis Hospital`, `Opp.Rta Office` |
| **C/O (Care Of)** | `C/O Manoj Kumar`, `C/O Dwarkadhis Enterprise` |
| **Colony/Nagar** | `Indira Nagar`, `Satnam Nagar`, `Sant Nagar` |
| **Block/Sector** | `Block A`, `Sector-74`, `Block-1` |
| **Old city names** | `Bombay` (Mumbai), `Madras` (Chennai), `Calcutta` (Kolkata), `Poona` (Pune) |
| **PIN codes** | Sometimes present, sometimes missing |
| **State full / abbreviated** | `Maharashtra` / `MH`, `Tamil Nadu` / `TN`, `Delhi` / `DL` |
| **Missing components** | Address may have no PIN, no state, or no city |
| **Mandal/Taluk/Tehsil** | `Serilingampally Mandal`, `Haveli`, `Andole Mandal` |
| **Native script states** | `महाराष्ट्र`, `ಕರ್ನಾಟಕ`, `தமிழ்நாடு` |
| **NULL/null tags** | `null` appears literally in some addresses |

### 8.3 French Address Patterns (Test Only)

| Pattern | Example |
|---|---|
| **Rue (Street)** | `20 Rue Parmentier`, `5 Rue des Canaris` |
| **R. (Rue abbreviated)** | `63 R. DE DIEPPE`, `68 R. DE JEMMAPES` |
| **Avenue / Ave** | `329 Avenue de Dunkerque` |
| **Boulevard / BD** | `175 Boulevard du President Franklin Roosevelt`, `089 BD SAINT-AIGNAN` |
| **Chemin (Path)** | `21 Chemin de la Traverse`, `65 Ch Du Moulin` |
| **Impasse (Dead-end)** | `5 Impasse Jean Baptiste Clement` |
| **Allee (Alley)** | `18 Allee de Gascogne`, `40 Allee Des Avocettes` |
| **Cite (Block)** | `9 B CITE MOUNEYRA` |
| **Quai (Quay)** | `15 QUAI ERNEST RANAUD` |
| **Bis (suffix)** | `5 bis Rue Pierre Dignac`, `20 bis RUE jules lefebvre` |
| **Region notation** | `Hauts-de-France`, `Nouvelle-Aquitaine`, `Pays de la Loire` |
| **Departement** | `Nord`, `Loire-Atlantique`, `Gironde`, `Pas-de-Calais` |

### 8.4 French Cities Observed in Test Data

`Bordeaux`, `Lille`, `Nantes`, `Dunkerque`, `La Teste-de-Buch`, `Calais`, `Tourcoing`, `Roubaix`, `Saint-Nazaire`, `Saint-Herblain`, `Pornic`, `La Baule Escoublac`, `Lege-Cap-Ferret`

### 8.5 French Regions Observed

- **Hauts-de-France** (Lille, Dunkerque, Calais, Tourcoing, Roubaix)
- **Nouvelle-Aquitaine** (Bordeaux, La Teste-de-Buch)
- **Pays de la Loire** (Nantes, Saint-Nazaire, Saint-Herblain)

### 8.6 Indian States Observed (Full Names & Abbreviations)

| State | Abbreviation | Native Script |
|---|---|---|
| Andhra Pradesh | AP | -- |
| Bihar | BR | -- |
| Delhi | DL | दिल्ली |
| Gujarat | GJ | ગુજરાત |
| Haryana | HR | हरियाणा |
| Himachal Pradesh | HP | -- |
| Jharkhand | JH | -- |
| Karnataka | KA | ಕರ್ನಾಟಕ |
| Kerala | KL | Keralam |
| Madhya Pradesh | MP | मध्य प्रदेश |
| Maharashtra | MH | महाराष्ट्र |
| Odisha | OD/OR | Orissa |
| Punjab | PB | ਪੰਜਾਬ |
| Rajasthan | RJ | राजस्थान |
| Tamil Nadu | TN | தமிழ்நாடு / Tamilnadu |
| Telangana | TG/TS | -- |
| Uttar Pradesh | UP | उत्तर प्रदेश |
| Uttarakhand | UK | -- |
| West Bengal | WB | পশ্চিমবঙ্গ |

### 8.7 US State Representation

US addresses use **2-letter state abbreviations** (`NC`, `TX`, `CA`, etc.) in Sources 1 and 2, but Source 3 frequently uses **full state names** (`North Carolina`, `Texas`, `California`).

### 8.8 Common Address Abbreviations (All Countries)

| Abbreviation | Full Form |
|---|---|
| `St` / `ST` | Street |
| `Rd` / `RD` | Road |
| `Ave` / `AVE` | Avenue |
| `Dr` / `DR` | Drive |
| `Ln` / `LN` | Lane |
| `Ct` / `CT` | Court |
| `Blvd` / `BLVD` | Boulevard |
| `Pl` / `PL` | Place |
| `Hwy` / `HWY` | Highway |
| `Cir` / `CIR` | Circle |
| `Pkwy` | Parkway |
| `H.No` / `HN` | House Number |
| `C/O` | Care Of |
| `Opp` / `OPP` | Opposite |
| `Nr` | Near |
| `Fl` | Floor |
| `Gf` / `GF` | Ground Floor |
| `Sf` / `SF` | Second Floor |
| `R.` | Rue (French: Street) |
| `Ch` / `Ch.` | Chemin (French: Path) |
| `BD` | Boulevard (French) |
| `AV` | Avenue (French) |

---

## 9. Noise Patterns & Data Quality Issues

### 9.1 Name-Level Noise

| Noise Type | Description | Examples |
|---|---|---|
| **Typos / OCR errors** | Character substitutions, missing letters | `Dmaigesostcis` (Diagnostics), `Tetlecommunication` (Telecommunication), `Hospirlg` (Hospital), `Hurricanne` (Hurricane) |
| **Number-for-letter substitution** | `0` for `O`, `5` for `S`, `6` for `G` | `TRADIN6` (TRADING), `5uperior` (Superior), `0tg` (OTG), `N0se` (Nose), `ART5` (ARTS) |
| **Letter transpositions** | Swapped characters | `Hasvey` (Harvey), `lnterstate` (Interstate), `Comimttee` (Committee) |
| **Word reordering** | Business name word order shuffled | `Private Ambernath Solar Limited` vs `Ambernath Solar Private Limited` |
| **Duplicate words** | Words repeated in name | `Crestline Crestline Clean LP`, `Odyssey Odyssey LLC`, `FERRERO FERRERO DUKE` |
| **Legal suffix variation** | Different forms of the same legal type | `LLC` / `L.L.C.`, `Inc` / `Inc.` / `Incorporated`, `Pvt` / `Private` |
| **Bracket/symbol wrapping** | Entity types wrapped in symbols | `[INCORPORATED]`, `[LLC]`, `(Co)`, `(Limited)` |
| **Diacritics added/corrupted** | Accented characters on English names | `Rexford`, `Finvest`, `Solar`, `Private`, `Network`, `Learning` |
| **Prefix noise** | Leading symbols or dashes | `-- Holloway Peak Inc`, `*** Yamuica Venus Pvt Ltd`, `#centraleducation` |
| **Hashtag-style** | Name formatted as hashtag | `#centraleducation`, `#gsmp0lytechnic` |
| **Casing variation** | ALL CAPS, lowercase, mixed | `ASSET BUILDING COALITION LLC` vs `asset building coalition llc` |
| **Gibberish/synthetic names** | Machine-generated nonsense | `XYL0TAVOQUO`, `Dovaxylo`, `Ectolumdrex`, `Xebravorix`, `Vantageavigild` |
| **URL-as-name** | Website used as business name | `heassociates.com`, `wilfordhancock.com`, `cuttackintlprivate.com` |

### 9.2 Address-Level Noise

| Noise Type | Description | Examples |
|---|---|---|
| **Component reordering** | City-State-Street vs Street-City-State | `OH, Columbus, 5559 Orville Avenue` vs `5559 Orville Avenue, Columbus, OH` |
| **Abbreviation variation** | Street vs St, Road vs Rd | `Westchester Drive` vs `Westchester Dr` |
| **Missing components** | No PIN, no state, no city | `No. 5, Chennai, TN` (minimal address) |
| **NULL / null placeholders** | Placeholder values | `067 PRODUCTION CT, NULL, INDEPENDENCE, KY` |
| **Empty addresses** | Completely blank address field | Several records have empty `business_address` |
| **Landmark references** | Indian addresses using landmarks | `Near SBI ATM`, `Opp Hotel Lords`, `Behind Gaur Nursing Home` |
| **Old/alternate city names** | Historical names used | `Bombay` to `Mumbai`, `Madras` to `Chennai`, `Calcutta` to `Kolkata`, `Poona` to `Pune` |
| **Spelling errors in addresses** | Typos in city/street names | `HYTERABAD` (Hyderabad), `CNROE` (Conroe), `BREMERTTON` (Bremerton), `CEDA CITY` (Cedar City), `AMMARILLO` (Amarillo) |
| **Double hash `##`** | Formatting artifacts | `##2326 341ND PLACE`, `##8 Willow Oak Lane` |
| **Double spaces** | Extra whitespace | `Diamond  Short LLC`, `Rapid  Learning Alliance LLC` |
| **Fractional addresses** | US addresses with fractions | `1791 1/2 TANNER WAY`, `19 1/2 STARDUST TRAIL` |
| **Ordinal typos** | Wrong ordinal suffix | `17RD STREET` (should be 17th), `31TH ST` (should be 31st) |
| **Dashes in house numbers** | Extra dashes | `2260- Housecreek Trail`, `1148- Heritage Ct` |
| **State in native script** | Indian state in non-Latin script | `महाराष्ट्र` instead of `Maharashtra` |

### 9.3 Country-Level Notes

- The `country` field is **always populated** (no nulls observed)
- Country is a **string label**, not a code: `US`, `India`, `France`
- The field should be treated as an **open set** — France proves new countries can appear

---

## 10. Ground Truth Analysis

### 10.1 Matching Statistics

| Metric | Value |
|---|---|
| **Total S1 entities (train)** | ~2,206,822 |
| **Singletons (no matches)** | Present -- entities with empty `matched_entity_ids` |
| **Max matches for one S1 entity** | 7+ (some entities match many S2/S3 records) |

### 10.2 Match Count Distribution (from samples)

Most S1 entities match 2-5 records across S2 and S3 combined. Common patterns:

- **0 matches** — Singletons (correctly predicting these scores 1.0)
- **1 match** — Only S2 or only S3
- **2 matches** — Typically 1 S2 + 1 S3
- **3-5 matches** — Multiple duplicates in S2 and/or S3
- **6-7+ matches** — Highly duplicated entities

### 10.3 S2 vs S3 Match Proportions

From the ground truth samples:

- Many entities have **both S2 and S3** matches
- Some entities match **only S2** (no S3 counterpart)
- Some entities match **only S3** (no S2 counterpart)
- S3 matches are often slightly more numerous than S2 matches per entity

### 10.4 Singleton Examples

```
S1-302869473	            (empty — singleton, no matches anywhere)
S1-262997549	            (empty — singleton)
S1-508022910	            (empty — singleton)
```

> [!TIP]
> **Singletons matter for scoring!** A singleton correctly predicted as empty scores **1.0**. A false match on a singleton scores **0.0**. Given F0.5 is precision-heavy, avoid generating false matches.

---

## 11. Source-Specific Characteristics

### 11.1 Source 1 (Reference / Anchor)

- **Deduplicated** — each real-world business appears exactly once
- Business names are **predominantly English** with clean formatting
- Addresses follow reasonably standard formats
- For India: Uses English state names (e.g., `Maharashtra`, `Tamil Nadu`)
- For US: Typically standard format with 2-letter state codes
- **Smallest source** (~2.2M train, ~1.7M test)

### 11.2 Source 2

- **Not deduplicated** — same business may appear multiple times
- Business names often in **ALL CAPS** for US and India
- Indian addresses frequently **ALL CAPS** with native-script state names
- Contains **native-script business names** (Hindi, Tamil, Telugu, etc.)
- Contains **more typos and OCR-like errors**
- Addresses sometimes have `N/A` or missing components
- French entries: street addresses use `R.`, `BD`, `AV` abbreviations
- **~5M records** (2.3x Source 1)

### 11.3 Source 3

- **Not deduplicated** — same business may appear multiple times
- US addresses use **full state names** (`North Carolina` instead of `NC`)
- Contains **native-script business names** similar to Source 2
- More **mixed-script names** (e.g., Tamil/Telugu + English)
- City names sometimes use old names (`Bombay`, `Madras`, `Calcutta`)
- More **garbled/synthetic names** and heavy noise
- French entries use departement names (e.g., `Gironde`, `Nord`) alongside regions
- Some addresses contain `null` as a literal string
- **~5.3M records** (2.4x Source 1)

---

## 12. Evaluation Details

### 12.1 Metric: F0.5 (Precision-Heavy)

```
F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
```

- **Precision weight: 2x** over Recall
- **Macro-averaged** per S1 entity, then averaged over all S1 entities
- **Singletons included** in the average

### 12.2 Scoring Implications

| Scenario | Score Impact |
|---|---|
| Correct match prediction | Increases Precision and Recall |
| False positive (wrong merge) | Heavily penalizes Precision (weighted 2x) |
| False negative (missed match) | Reduces Recall (weighted 1x) |
| Singleton correctly predicted empty | **1.0** for that entity |
| Singleton with false match predicted | **0.0** for that entity |

### 12.3 Example Calculation

- Your model predicts S1-00001 matches [S2-00047, S2-00193, S3-00812]
- Ground truth says S1-00001 matches [S2-00047, S3-00812]
- Precision = 2/3, Recall = 2/2 = 1.0
- F_0.5 = (1.25 x 0.667 x 1.0) / (0.25 x 0.667 + 1.0) = **0.714**

### 12.4 Leaderboard

- **Public Leaderboard:** Based on a subset of test set (real-time feedback)
- **Private Leaderboard:** Based on remaining test set (revealed after challenge)
- **Final rankings** determined by private leaderboard

---

## 13. Output Format Requirements

### 13.1 `matching_results.tsv` (Scored on Leaderboard)

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

### 13.2 `candidate_pairs.tsv` (Not Scored, Used for Audit)

```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

### 13.3 Validation Rules

| Rule | Status |
|---|---|
| Every S1 test entity must have exactly one row | Required |
| `matched_entity_ids` can be empty (for singletons) | Allowed |
| No S1- IDs in matched list (no self-matches) | Enforced |
| No duplicate IDs within a single list | Enforced |
| No duplicate `source1_entity_id` rows | Enforced |
| Only S2-/S3- IDs that exist in the test set | Enforced |
| Matches should be a subset of candidates | Warning only |

---

## 14. Utility Scripts

### 14.1 `utils/validate_submission.py`

- **Python 3.8+**, stdlib only (no dependencies)
- Validates both `matching_results.tsv` and `candidate_pairs.tsv`
- Checks: header format, tab-separation, no duplicates, no S1 self-matches, coverage of all S1 entities
- Optional `--check-ids` flag loads S2/S3 IDs (memory-heavy: several GB)
- Run from `student_resource/` directory:

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

## 15. Submission Package Structure

```
team_name_submission.zip
|-- output/
|   |-- matching_results.tsv        # final matches (leaderboard file)
|   |-- candidate_pairs.tsv         # blocking candidate set
|-- code/
|   |-- business_entity_resolution/
|       |-- src/                    # all source code
|       |-- README.md               # reproduction instructions
|       |-- requirements.txt        # pinned dependencies
|-- Documentation_template.md       # filled-in methodology write-up
```

---

## 16. Key Constraints & Rules

| Constraint | Details |
|---|---|
| **No external data** | No APIs, databases, geocoding, or internet lookups |
| **No commercial ER services** | No entity resolution APIs |
| **Model license** | MIT or Apache 2.0 only |
| **Model size** | Up to 8 billion parameters |
| **Format** | Tab-separated `.tsv` files |
| **Encoding** | UTF-8 |

---

## 17. Documentation Template Structure

The provided `Documentation_template.md` requires teams to fill in:

1. **Executive Summary** — 2-3 sentence overview
2. **Methodology** — Problem analysis, solution strategy, approach type
3. **Candidate Generation (Blocking)** — Blocking keys, pair counts, recall guarantees
4. **Matching Model** — Features (name/address), model type, threshold selection
5. **Results and Error Analysis** — F0.5 score, common false positives/negatives
6. **Conclusion** — Key achievements and lessons learned
7. **Appendix** — Code artefacts, additional results

---

## 18. Tips & Strategies (from README)

1. **Blocking is critical** — it determines recall's upper bound
2. **String similarity features**: Jaccard, Levenshtein, TF-IDF cosine for name/address matching
3. **Country-specific patterns** — each country has unique address structures
4. **Precision over Recall** — F0.5 penalizes false merges more than missed links
5. **Don't ignore singletons** — correctly predicting "no match" = 1.0 per entity
6. **Validate output format** before every submission

---

## 19. Summary Statistics at a Glance

| Dimension | Value |
|---|---|
| Total files | 7 data TSVs + 1 ground truth TSV |
| Total raw size | ~2.5 GB |
| Total records | ~26.4 million |
| Countries (train) | 2 (US, India) |
| Countries (test) | 3 (US, India, **France** -- zero-shot) |
| Writing scripts | 11+ (Latin, Devanagari, Tamil, Telugu, Kannada, Malayalam, Gujarati, Gurmukhi, Bengali, Odia, French diacritics) |
| Languages | 10+ (English, Hindi, Marathi, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Bengali, Odia, French) |
| Business legal forms | 20+ across 3 countries |
| Noise categories | 15+ distinct types |
| Evaluation metric | F0.5 (macro, precision-heavy) |
| Model constraint | Up to 8B params, MIT/Apache 2.0 |

---

> [!CAUTION]
> The dataset is designed to be extremely challenging with multi-lingual noise, cross-script matching, zero-shot generalization (France), and massive scale (~26M records). A naive approach will not work — you need robust blocking, multilingual text normalization, and careful threshold tuning.