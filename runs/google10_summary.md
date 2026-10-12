# Google_Chem_10_tasks: verifier results

Prompt `baseline_v0`, model `google/gemini-3.1-pro-preview`, thinking `low`, 5 run(s) per pair. **No gold answers exist for these 10, so a verdict here is not scored right or wrong.**

| Task | Verdict | Reason (run 0) | Cost | Time |
|---|---|---|---|---|
| molecule_213 | 0/0/0/1/1 | Image B replaces the explicit alternating double bonds in the benzene rings with aromatic circles, violating the guideline to preserve chemical notation exactly. | $0.014 | 11.65s |
| molecule_470 | 0/0/0/0/0 | Image B incorrectly expands the implicit methyl group to 'H3C', violating the guideline to preserve chemical notation exactly. | $0.006 | 4.79s |
| molecule_5 | 0/0/0/0/0 | The reproduction incorrectly expands the "Me" abbreviations to "CH3", violating the guideline against changing abbreviations. | $0.010 | 7.35s |
| molecule_252 | 0/0/0/0/0 | The annotator expanded abbreviations such as Me to CH3, which violates the guideline against changing abbreviations. | $0.007 | 5.46s |
| molecule_25 | 1/0/1/1/1 | Image B is the same compound as Image A and follows every guideline. | $0.006 | 4.11s |
| molecule_132 | 0/0/0/0/0 | The annotator replaced the explicit alternating double bonds of the benzene rings with circles, violating the guideline to preserve chemical notation exactly as represented. | $0.016 | 12.78s |
| molecule_353 | 0/0/0/0/0 | Image B uses a circle to represent the aromatic ring, whereas Image A uses alternating double bonds. | $0.014 | 10.1s |
| molecule_118 | 0/0/0/0/0 | The abbreviations "Me" and "PhOOC" were incorrectly expanded into "CH3" and a fully drawn phenyl ester structure, violating the guideline against abbreviation changes. | $0.014 | 20.75s |
| molecule_390 | 0/0/0/0/0 | The annotator expanded the abbreviations NHMe to NHCH3 and CF3 to a carbon bonded to three fluorine atoms, violating the guidelines. | $0.010 | 6.88s |
| molecule_455 | 0/0/0/0/0 | Image B expands the OEt abbreviation to OC2H5 and explicitly writes the CH2 group, violating the guideline against changing abbreviations and preserving chemical notation. | $0.009 | 7.74s |

Verdict counts (calls): {'0': 44, '1': 6}. Total spend on verdicts: $0.622.

## Perception cross-check (reference vs annotated, transcribed independently)

Each image was transcribed on its own by the same model. Differences in the critical fields are places where the two drawings differ or where the model misread one of them; both explanations need a look at the images. This is a cross-check on the verdict, not gold.

| Task | Annotated ink | Annotated legibility | Critical fields agree | Differences |
|---|---|---|---|---|
| molecule_213 | black | clear | yes |  |
| molecule_470 | black | clear | 6/7 | **atom_labels**: annotated only H3C |
| molecule_5 | black | clear | 5/7 | **atom_labels**: ref only HO2C, Me, Me; annotated only CH3, CH3, CO2H; **abbreviations**: ref only Me, Me |
| molecule_252 | black | clear | 4/7 | **atom_labels**: ref only COOMe, Me, MeO; annotated only CH3, CH3O, COOCH3; **abbreviations**: ref only COOMe, Me, MeO; **ring_sizes**: ref only 5 |
| molecule_25 | black | clear | yes |  |
| molecule_132 | black | clear | yes |  |
| molecule_353 | black | clear | yes |  |
| molecule_118 | black | clear | 4/7 | **atom_labels**: ref only Me, PhOOC; annotated only C, CH3, O, O; **abbreviations**: ref only Me, PhOOC; **ring_sizes**: annotated only 6 |
| molecule_390 | black | clear | 5/7 | **atom_labels**: ref only CF3, NHMe; annotated only C, F, F, F, NHCH3; **abbreviations**: ref only Me |
| molecule_455 | black | clear | 5/7 | **atom_labels**: ref only OEt; annotated only CH2, OC2H5; **abbreviations**: ref only OEt |

