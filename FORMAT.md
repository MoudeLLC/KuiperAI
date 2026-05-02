# Model Naming and Identification Format Specification
**Copyright © 2024-2026 Moude AI LLC. All Rights Reserved.**

---

## FnDS/FDS (Format and Decimal System / Format & Decimal System)
**Standard:** `Moude40150`

- **Company Nick Name:** `Moude`
- **Company Identifier Code:** `40150` (Moude LLC. identifier)
- **Format:** `{companyNickName}{companyIdentifierCode}`

**Note:** This code format rule is created and maintained by Moude AI LLC.

---

## Model Naming Structure

### Full Model Name Format
```
{mainName}-{productionSerieLetter}{serieVersionNum.noneDec}-{contextSizeOnlyNum.biggestPlaceValue}{biteValue.firstLetter.lowerCase}
```

**Example:** `{ModelName}-{X}{1}-{N}{unit}`

**Breakdown:**
- `{mainName}` = Model family name
- `{productionSerieLetter}` = Series letter (uppercase)
- `{serieVersionNum.noneDec}` = Version number (no decimals)
- `{contextSizeOnlyNum.biggestPlaceValue}` = Context size biggest place value
- `{biteValue.firstLetter.lowerCase}` = Bite value unit (k/m/g, lowercase)

---

## Model Nickname Format
```
{productionSerieLetter}{serieVersionNum.noneDec}-{contextSizeOnlyNum.biggestPlaceValue}{biteValue.firstLetter.lowerCase}
```

**Example:** `{X}{1}-{N}{unit}`

**Note:** No `{mainName}-` prefix

---

## Series Identifier Format
```
{productionSerieLetter}{serieVersionNum.noneDec}
```

**Example:** `{X}{1}`

**Breakdown:**
- `{productionSerieLetter}` = Series letter (uppercase)
- `{serieVersionNum.noneDec}` = Version number (no decimals)

---

## Context Format

### Full Context
```
{contextLength}
```
**Example:** `{NNNN}` (full number)

### Context Substring
```
{contextSizeOnlyNum.biggestPlaceValue}{biteValue.firstLetter.lowerCase}
```
**Example:** `{N}{unit}`

**Breakdown:**
- `{contextSizeOnlyNum.biggestPlaceValue}` = Biggest place value from context length
- `{biteValue.firstLetter.lowerCase}` = Bite value unit (k/m/g, lowercase)

---

## MMI (MoudeAI Models Identifier)

### Format
```
{GMIC}{modelName.firstLetter}{GAID}A{MDD.A.uppercase}I{MDD.I.uppercase}
```

**Example:** `{NNNN}{X}{NNNN}A{NN}I{NN}`

### Breakdown

| Component | Format | Description |
|-----------|--------|-------------|
| `{GMIC}` | `{NNNN}` | Generative Models Identifier Code (4 digits) |
| `{modelName.firstLetter}` | `{X}` | First letter of model name (uppercase) |
| `{GAID}` | `{NNNN}` | Generative Access I.D. (4 digits, leading zeros removed in MMI) |
| `A` | `A` | Artificial (required for all models) |
| `{MDD.A.uppercase}` | `{NN}` | Moude Development-Usaged Decimal Code for 'A' uppercase |
| `I` | `I` | Intelligence/Intelligent (required for all models) |
| `{MDD.I.uppercase}` | `{NN}` | Moude Development-Usaged Decimal Code for 'I' uppercase |

### GMIC Codes (Generative Models Identifier Code)
- **Format:** 4-digit code
- **Purpose:** Identifies model type/category
- **Example codes:**
  - Text-Generative Models: `{NNNN}`
  - Image-Generative Models: `{NNNN}`
  - Audio-Generative Models: `{NNNN}`

### GAID (Generative Access I.D.)
- **Format:** 4-digit number (with leading zeros)
- **Storage Format:** `{NNNN}` (always 4 digits)
- **MMI Format:** `{NNN}` or `{NNNN}` (leading zeros removed in MMI only)
- **Purpose:** Unique identifier for providers & clients to find and access the model
- **Created By:** Moude AI LLC
- **Usage:** Currently used only for Moude models
- **Function:** Allows providers and clients to locate and retrieve specific models from the model registry

### MDD (Moude Development-Usaged Decimal Code)
Character to decimal code mapping for uppercase letters:

| Character | MDD Code Format |
|-----------|-----------------|
| A-Z | `{NN}` (2-digit code) |

**Note:** Specific mappings are proprietary to Moude AI LLC

---

## Complete Model Metadata Structure

```json
{
  "company_name": "{CompanyName}",
  "model_base_name": "{ModelBaseName}",
  "model_name": "{ModelName}-{X}{N}-{N}{unit}",
  "model_nickname": "{X}{N}-{N}{unit}",
  "series_identifier": "{X}{N}",
  "mmi": "{NNNN}{X}{NNNN}A{NN}I{NN}",
  "gaid": "{NNNN}",
  "version": "{major}.{minor}.{patch}",
  "context": {contextLength},
  "context_substr": "{N}{unit}",
  "total_parameters": {parameterCount},
  "architecture": "{architectureType}",
  "copyright": "{copyrightInfo}",
  "license": "{licenseType}"
}
```

---

## Field Definitions

| Field | Format | Description |
|-------|--------|-------------|
| `company_name` | `{CompanyName}` | Company/brand name |
| `model_base_name` | `{ModelBaseName}` | Base model family name |
| `model_name` | `{mainName}-{series}-{context}` | Full official model name |
| `model_nickname` | `{series}-{context}` | Short/casual model name |
| `series_identifier` | `{letter}{number}` | Series and version |
| `mmi` | See MMI format above | MoudeAI Models Identifier |
| `gaid` | `{NNNN}` | Generative Access I.D. (4 digits) |
| `context` | `{number}` | Full context length in tokens |
| `context_substr` | `{num}{unit}` | Short context representation |

---

## Naming Examples

### Generic Model Examples

#### Standard Series
- `{Model}-{X}1-{N}k` (Series X1, N thousand context)
- `{Model}-{X}2-{N}k` (Series X2, N thousand context)
- `{Model}-{X}3-{N}k` (Series X3, N thousand context)

#### Large Series
- `{Model}-{Y}1-{N}k` (Series Y1, N thousand context)
- `{Model}-{Y}2-{N}k` (Series Y2, N thousand context)

#### Mini Series
- `{Model}-{Z}1-{N}k` (Series Z1, N thousand context)
- `{Model}-{Z}2-{N}k` (Series Z2, N thousand context)

---

## Context Size Conversion Table

| Full Context | Biggest Place Value | Bite Value | Context Substr |
|--------------|---------------------|------------|----------------|
| 1,024 | 1 | k | `1k` |
| 2,048 | 2 | k | `2k` |
| 4,096 | 4 | k | `4k` |
| 8,192 | 8 | k | `8k` |
| 16,384 | 16 | k | `16k` |
| 32,768 | 32 | k | `32k` |
| 65,536 | 65 | k | `65k` |
| 131,072 | 131 | k | `131k` |
| 1,048,576 | 1 | m | `1m` |

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2026-05-02 | Initial format specification |

---

## Notes

1. **Naming Conventions:**
   - All series letters are uppercase
   - All bite value letters are lowercase
   - No spaces in model names
   - Hyphens separate major components

2. **Format Rules:**
   - GAID is always stored as 4 digits with leading zeros
   - GAID in MMI has leading zeros removed
   - MDD codes are 2-digit decimal representations
   - MMI always includes 'A' and 'I' components

3. **Reserved Components:**
   - Series letters identify model categories
   - GMIC codes identify generative model types
   - MDD codes map characters to decimal values

---

**Document Maintained By:** Moude AI LLC  
**Last Updated:** May 2, 2026  
**Format Standard:** FnDS/FDS Moude40150
