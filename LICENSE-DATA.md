# Licence for results, charts and the plan

The code in this repository is under the MIT licence (`LICENSE`). The files below are the study's own outputs and are licensed under
[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). The full legal text is at
<https://creativecommons.org/licenses/by/4.0/legalcode>.

| Path | What it is |
| --- | --- |
| `results/` | the merge manifest, the results table (one row per run) and the rescore summary |
| `docs/` | charts and summary tables |
| `plan/` | the test plan, the question identifiers and the checked claims |

To use them, credit the repository and keep a link to the licence.

## Intended use

These files are for evaluation and analysis. They are not for training models. The results table holds no model-generated text, but it describes the behaviour of models whose vendors (OpenAI, Anthropic and Google) restrict training competing models on their outputs. This is a notice about intended use. It does not change the CC BY 4.0 terms above.

## What these files do not contain

They hold no BIRD question text, SQL, answer keys or database content, and no NYC taxi trip data. Questions are referred to by identifier
only. This is why they are not a derivative of BIRD and are not covered by its share-alike terms. If you add any BIRD question text,
SQL or schema text to a file, license that file under CC BY-SA 4.0 and mark it clearly.

## Third-party material

- BIRD (a BIg benchmark for laRge-scale Database grounded in text-to-SQL tasks) is licensed CC BY-SA 4.0 by its authors. It is not
  redistributed here. See the attribution section of `README.md`.
- NYC TLC Trip Record Data is published by the New York City Taxi and Limousine Commission and is not redistributed here.
- Model names and the names of vendors are used to identify the models that were tested. This study is independent and is not
  affiliated with or endorsed by any of them.
