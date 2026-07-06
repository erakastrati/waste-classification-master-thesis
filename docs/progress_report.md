# Raport Progresi — Tema e Masterit
## Klasifikimi i Mbeturinave përmes Imazheve duke përdorur CNN

---

**Studenti:** Era Kastrati  
**Mentori:** Bertan Karahoda  
**Data:** Korrik 2026  
**Statusi:** Faza eksperimentale e përfunduar — gati për dokumentim

---

## 1. Përmbledhje

Ky raport përshkruan punën e realizuar deri tani në kuadër të temës së masterit mbi klasifikimin automatik të mbeturinave duke përdorur rrjete nervore konvolucionale (CNN). Janë realizuar **11 eksperimente** sistemike, duke filluar nga një CNN e ndërtuar nga zero deri te fine-tuning i avancuar i modeleve të pre-trajnuara. Rezultati final arrin **92.17% saktësi** mbi një dataset real me 600 foto të mbledhura nga ambienti i përditshëm.

---

## 2. Dataset-et e Përdorura

### 2.1 Dataset trajnimit (TrashNet — bazë)

| Parametri | Vlera |
|-----------|-------|
| Burimi | TrashNet (Kaggle) |
| Klasa | 6: cardboard, glass, metal, paper, plastic, trash |
| Imazhe totale | 2,527 |
| Ndarja | 80% trajnim (2,022) / 20% validim (505) |
| Dimensionet | 512×384 px (risize në 224×224) |
| Karakteristikë | Imazhe në kushte të kontrolluara, sfond i bardhë |

**Shpërndarja e klasave (TrashNet):**

| Klasa | Imazhe |
|-------|--------|
| cardboard | 403 |
| glass | 501 |
| metal | 410 |
| paper | 594 |
| plastic | 482 |
| trash | 137 ← e pabalancuar |
| **TOTAL** | **2,527** |

### 2.2 Dataset-et shtesë (fine-tuning)

| Dataset | Imazhe | Qëllimi |
|---------|--------|---------|
| TACO (Trash Annotations in Context) | 3,601 crops | Imazhe reale me objekte të izoliara |
| RealWaste | 3,905 | Imazhe nga ambienti i deponisë |
| Household Waste | 7,500 | Imazhe shtëpiake, objekte të zakonshme |
| Garbage Classification v2 | ~17,000 | Dataset i larmishëm real-world |

### 2.3 Real Test Dataset (i mbajtur jashtë trajnimit)

| Parametri | Vlera |
|-----------|-------|
| Qëllimi | Testim final — **kurrë i përdorur për trajnim** |
| Imazhe totale | **600** |
| Imazhe për klasë | **100 (të balancuara)** |
| Mbledhja | Foto reale + burime interneti, kushte jo të kontrolluara |
| Karakteristikë | Sfonde të ndryshme, ndriçim variues, kënde të ndryshme |

---

## 3. Arkitektura dhe Qasja

### 3.1 CNN bazike (nga zero)

Ndërtuar me shtresa konvolucionale, MaxPooling dhe Dense — pa pesha të pre-trajnuara. Shërbeu si *baseline* për krahasim.

### 3.2 Transfer Learning

Janë përdorur dy arkitektura:
- **MobileNetV2** — model i lehtë, i shpejtë
- **EfficientNetB0** — balancë optimale mes saktësisë dhe kompleksitetit

**Strategjia e trajnimit:**
1. *Feature Extraction* — shtresat bazë të ngrira, trajnohen vetëm shtresat e fundit
2. *Fine-Tuning* — zhbllokimi i shtresave të sipërme me `lr=1e-5` (learning rate i ulët për të shmangur harrimin katastrofik)
3. *Callbacks*: EarlyStopping(patience=5), ModelCheckpoint(monitor=val_accuracy)

### 3.3 Data Augmentation

Aplikuar gjatë trajnimit:
- Kthim horizontal (flip)
- Rrotullim ±20°
- Zoom ±15%
- Ndryshim ndriçimi

### 3.4 Test-Time Augmentation (TTA)

Gjatë inferimit, çdo imazh paraqitet nga **5 pamje të ndryshme** dhe probabilitetet mësohen duke i mesatarizuar — kjo redukton ndjeshmërinë ndaj kornizimit dhe sfondit.

---

## 4. Eksperimentet (EXP-001 deri EXP-011)

### 4.1 Tabela përmbledhëse e eksperimenteve

| ID | Modeli | Val TrashNet | Real-World (v2) | Shënim |
|----|--------|-------------|-----------------|--------|
| EXP-001 | CNN Baseline (nga zero) | 56.44% | — | Baseline |
| EXP-002 | MobileNetV2 Transfer Learning | 87.13% | — | +30.69pp vs CNN |
| EXP-003 | EfficientNetB0 Transfer Learning | 89.90% | 78.33% | +2.77pp vs MobileNet |
| EXP-004 | EfficientNetB0 + Augmentation | 89.70% | 82.67% | +4.34pp real-world |
| EXP-005 | + Fine-tune TACO | 92.87% | — | Rekord TrashNet |
| EXP-006 | + Background Removal (rembg) | — | — | **Rezultat negativ** −2.56pp |
| EXP-007 | + Fine-tune RealWaste | 91.29% | — | +2.57pp real-world |
| EXP-008 | + Fine-tune Household | 90.50% | 88.83% | +10.50pp vs bazë |
| EXP-009 | + TTA | 90.50% | 88.83% (TTA) | +5pp pa ritrainim |
| EXP-010 | Ensemble 3 modele + TTA | — | — | **Rezultat negativ** −3.85pp |
| **EXP-011** | **+ Fine-tune Garbage v2** | **89.90%** | **92.17%** | **BEST** |

### 4.2 Progresi real-world (dataset v2, TTA)

```
EfficientNetB0 (bazë):     78.33%
+ Augmentation:            82.67%  (+4.34pp)
+ Household Fine-tune:     88.83%  (+6.16pp)
+ Garbage v2 Fine-tune:    92.17%  (+3.34pp)
─────────────────────────────────────
Totali:                   +13.84pp
```

---

## 5. Rezultati Final

### 5.1 Saktësia totale

| Metrika | Vlera |
|---------|-------|
| **Saktësia totale (TTA)** | **92.17% (553/600)** |
| Single-pass (pa TTA) | ~91% |
| Macro F1-score | 0.9236 |
| Dataset testimi | 600 foto reale, 100/klasë |

### 5.2 Saktësia për klasë

| Klasa | Saktë | Gabim | % |
|-------|-------|-------|---|
| cardboard | 92 | 8 | 92% |
| glass | 95 | 5 | 95% |
| metal | 94 | 6 | 94% |
| paper | 90 | 10 | 90% |
| plastic | 92 | 8 | 92% |
| trash | 90 | 10 | 90% |
| **TOTAL** | **553** | **47** | **92.17%** |

### 5.3 Konfuzionet kryesore

- glass → plastic: 3 raste (xham i ngjashëm me plastikë)
- cardboard → trash: 4 raste (karton i shtypur/i ndotur)
- paper → cardboard: 5 raste (letër e trashë)
- trash → paper/plastic: të shpërndarë

### 5.4 Krahasimi i modeleve mbi dataset-in real v2

| Modeli | Saktësia |
|--------|----------|
| CNN Baseline | ~30% (vlerësim) |
| MobileNetV2 | ~70% (vlerësim) |
| EfficientNetB0 (TrashNet only) | 78.33% |
| EfficientNetB0 + Augmentation | 82.67% |
| EfficientNetB0 + Household FT | 88.83% |
| **EfficientNetB0 + Garbage v2 FT** | **92.17%** |

---

## 6. Gjetjet Kryesore

### 6.1 Domain Shift — problemi kryesor

TrashNet trajnohet me foto studio (sfond i bardhë, ndriçim uniform). Dataset-i real ka sfonde komplekse, ndriçim të ndryshëm, objekte të paqëndrueshme. Kjo krijoi hendek të madh: EfficientNetB0 vetëm nga TrashNet → 78.33% real-world.

**Zgjidhja:** Fine-tuning progresiv me dataset-e reale (TACO → Household → Garbage v2) rriti saktësinë real-world me +13.84pp.

### 6.2 Data Augmentation

Konfirmoi hipotezën: +4.34pp real-world pa ndryshim arkitekture. Veçanërisht efektive për imazhe me sfonde të ndryshme.

### 6.3 Test-Time Augmentation (TTA)

+1-5pp pa ritrainim — teknikë e thjeshtë me ndikim real. Mesatarizimi i 5 pamjeve redukton ndjeshmërinë ndaj kornizimit.

### 6.4 Rezultate negative me vlerë akademike

- **EXP-006 (rembg):** Heqja e sfondit *dëmtoi* saktësinë (−2.56pp). Tregon se preprocessing naiv nuk funksionon gjithmonë.
- **EXP-010 (Ensemble):** Mesatarizimi i 3 modeleve dha rezultat më të keq se modeli më i mirë vetëm (−3.85pp). Modelet e dobëta "zvarrisin" të fortin.

### 6.5 Transfer Learning kundër CNN nga zero

| Qasja | TrashNet Val |
|-------|-------------|
| CNN nga zero | 56.44% |
| MobileNetV2 | 87.13% (+30.69pp) |
| EfficientNetB0 | 89.90% (+33.46pp) |

Transfer learning është qartësisht superior, veçanërisht kur dataset-i trajnimit është i vogël.

### 6.6 Klasa "trash" — sfida kryesore

Trash është klasa më e vështirë: 137 imazhe në TrashNet (vs 594 paper), kategori e paqartë vizualisht (objekte të ndryshme). Me fine-tuning, u rrit nga 52% → 90%.

---

## 7. Aplikacioni Web

U ndërtua një web aplikacion Flask me:
- **Backend:** Python Flask, model EfficientNetB0 + TTA
- **Frontend:** HTML/CSS/JavaScript, single-page
- **Funksionalitete:**
  - Upload drag & drop i imazhit
  - Klasifikim real-time
  - Confidence score dhe bar chart për të 6 klasat
  - Tip riciklimi për klasën e predikuar
  - Badge me saktësinë e sistemit (92.2%)

**Komanda për nisje:**
```bash
cd waste-classification-master-thesis
source venv/bin/activate
python app.py
# Hap: http://127.0.0.1:5000
```

---

## 8. Pyetjet Hulumtuese — Përgjigjet

**Q1: Sa ndryshon performanca TrashNet → real-world?**
EfficientNetB0 nga TrashNet vetëm: 89.90% TrashNet → 78.33% real-world (−11.57pp domain shift). Me fine-tuning progresiv mbyllet ky hendek: 92.17% real-world.

**Q2: Cila qasje jep performancën më të mirë?**
EfficientNetB0 + Fine-tune Garbage v2 + TTA: **92.17%** — superior ndaj CNN bazike (+35.73pp) dhe MobileNetV2 (~+22pp).

**Q3: Cilat klasa ngatërrohen më shpesh?**
Glass↔Plastic (ngjashmëri vizuale e materialit) dhe Trash↔klasat e tjera (kategori e paqartë). Analiza e gabimeve u bë per çdo klasë me emra specifike imazhesh.

---

## 9. Struktura e Projektit

```
waste-classification-master-thesis/
├── app.py                              # Web app (Flask)
├── requirements-app.txt               # Dependencies minimale
├── templates/index.html               # Frontend UI
├── src/
│   ├── data/                          # Skripte përgatitje dataset-esh
│   │   ├── prepare_garbage_v2.py
│   │   ├── prepare_household.py
│   │   └── ...
│   ├── training/                      # Skripte trajnimi
│   │   ├── train_garbage_finetune.py  # EXP-011
│   │   ├── train_object_centric.py    # EXP-008
│   │   └── ...
│   ├── inference/
│   │   └── tta_predict.py             # TTA inferim
│   └── evaluation/
│       └── evaluate_tta.py            # Evaluim real-world
├── data/
│   ├── real_test_dataset/             # 600 foto reale (held-out)
│   └── garbage-v2-prepared/           # Dataset trajnimit
├── results/
│   └── efficientnet_garbage/          # Model final (~19 MB)
└── docs/
    ├── notes.md                       # Log i detajuar i eksperimenteve
    └── progress_report.md             # Ky dokument
```

---

## 10. Teknologjitë e Përdorura

| Teknologjia | Versioni | Roli |
|-------------|---------|------|
| Python | 3.9 | Gjuha kryesore |
| TensorFlow / Keras | 2.16.2 | Framework deep learning |
| EfficientNetB0 | — | Arkitektura kryesore |
| PIL / Pillow | 11.3.0 | Preprocessing imazhesh |
| NumPy | 1.26.4 | Operacione numerike |
| scikit-learn | 1.6.1 | Metrika evaluimi |
| Flask | 3.1.3 | Web backend |
| Matplotlib / Seaborn | — | Vizualizime |

---

## 11. Gjendja aktuale dhe hapat e ardhshëm

### Realizuar plotësisht:
- ✅ CNN bazike nga zero (EXP-001)
- ✅ Transfer learning MobileNetV2 dhe EfficientNetB0 (EXP-002, 003)
- ✅ Data augmentation (EXP-004)
- ✅ Fine-tuning me dataset-e të shumta (EXP-005, 007, 008, 011)
- ✅ TTA — Test-Time Augmentation (EXP-009)
- ✅ Eksperimente negative të dokumentuara (EXP-006, 010)
- ✅ Dataset real 600 foto, i balancuar (100/klasë)
- ✅ Evaluim final: 92.17% (553/600)
- ✅ Analiza e gabimeve per klasë
- ✅ Aplikacion web funksional
- ✅ Log i detajuar i eksperimenteve (docs/notes.md)
- ✅ README me udhëzime setup/run

### Mbetet për thesis:
- 📝 Shkrimi i kapitujve të dokumentit të temës
- 📝 Analiza e literaturës (kapitulli teorik)
- 📝 Diskutimi i rezultateve (interpretim akademik)
- 📝 Konkluzionet dhe rekomandimet për punë të ardhshme

---

## 12. Konkluzion

Sistemi i zhvilluar arrin **92.17% saktësi** mbi 600 foto reale të mbledhura nga ambienti i përditshëm — rezultat konkurrues me literaturën ekzistuese, veçanërisht duke marrë parasysh vështirësinë e dataset-it real (sfonde komplekse, ndriçim i ndryshëm, objekte të dëmtuara).

Kontributi kryesor akademik: demonstrimi sistematik se si **fine-tuning progresiv me dataset-e të larmishme reale** mbyll hendekun e domain shift — nga 78.33% (EfficientNet bazik) në 92.17% (+13.84pp), duke i dhënë përgjigje konkrete pyetjeve hulumtuese të propozimit.

---

*Ky raport u përgatit si përmbledhje e fazës eksperimentale të temës së masterit.*  
*Kodi burimor i plotë ndodhet në: `waste-classification-master-thesis/` (branch: dev)*
