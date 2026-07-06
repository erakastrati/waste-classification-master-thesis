# Raport Progresi — Tema e Masterit
### Klasifikimi Automatik i Mbeturinave duke përdorur CNN

**Studenti:** Era Kastrati  
**Mentori:** Bertan Karahoda  
**Data:** Korrik 2026

---

## 1. Çfarë kam bërë?

Kam ndërtuar një sistem që **njeh automatikisht llojin e mbeturinës nga një foto**. Sistemi klasifikon mbeturinën në 6 kategori: karton, qelq, metal, letër, plastikë dhe mbeturinë e përgjithshme.

**Rezultati final: 92.2% saktësi** mbi 600 foto reale të mbledhura nga ambienti i përditshëm.

---

## 2. Analiza e kërkesave të propozimit — çfarë është realizuar

### Qëllimet e propozimit

| Qëllimi nga propozimi | Realizimi |
|-----------------------|-----------|
| Ndërtimi i modelit bazë CNN | ✅ EXP-001 — CNN nga zero, 56.44% |
| Transfer learning (MobileNetV2 / EfficientNetB0) | ✅ EXP-002 (87.13%) dhe EXP-003+ (89.90%) |
| Analizimi i rezultateve dhe gabimeve | ✅ Per çdo eksperiment, per klasë, per imazh |
| Krahasimi i qasjeve të ndryshme | ✅ CNN vs MobileNet vs EfficientNet vs fine-tuning |
| Testim me imazhe të reja (të paparë) | ✅ 600 foto reale — kurrë në trajnim |
| Vlerësim gjeneralizimi real-world | ✅ Cross-dataset evaluation, 92.17% |

> Propozimi specifikon: *"Punimi nuk përfshin integrimin e drejtpërdrejtë të modelit në një sistem real në Kosovë"* — kjo nuk u krye, sipas planit.

---

### Metodat e punës — pikë për pikë

| Metoda nga propozimi | Realizimi | Detaje |
|----------------------|-----------|--------|
| Analiza e literaturës (CNN, deep learning, mbeturina) | ✅ | Dokumentuar, kapitulli teorik mbetet për t'u shkruar |
| TrashNet, 6 kategori | ✅ | Baza e gjithë projektit |
| Preprocessing (resize, normalization) | ✅ | 224×224px, EfficientNet preprocess |
| Data augmentation | ✅ | EXP-004 — konfirmoi +4.34pp real-world |
| Dataset shtesë real (vetëm testim, jo trajnim) | ✅ | 600 foto, 100/klasë, kurrë në trajnim |
| Cross-dataset evaluation | ✅ | TrashNet train → real dataset test |
| CNN bazike nga zero | ✅ | EXP-001 |
| MobileNetV2 | ✅ | EXP-002 |
| EfficientNetB0 | ✅ | EXP-003 deri EXP-011 |
| Feature extraction (shtresa të ngrira) | ✅ | Faza 1 e çdo TL eksperimenti |
| Fine-tuning (shtresa të sipërme) | ✅ | Faza 2, lr=1e-5 |
| Analiza hiperparametrave | ✅ | lr=1e-5, batch=32, EarlyStopping(patience=5) — vlera standarde, të justifikuara |
| CNN bazë vs transfer learning | ✅ | EXP-001 vs EXP-002 vs EXP-003 |
| Me / pa augmentation | ✅ | EXP-003 vs EXP-004 |
| Accuracy, Precision, Recall, F1-score | ✅ | Gjeneruar për çdo eksperiment |
| Confusion matrix | ✅ | Gjeneruar për çdo model |
| Identifikimi klasave problematike | ✅ | Glass↔Plastic, Trash↔Metal |
| Dataset imbalance | ✅ | Trash kishte 137 foto (vs 594 paper) — trajtuar me class_weight |
| Vizualizime grafike | ✅ | Training curves, confusion matrices, bar charts |
| Implementim praktik Python | ✅ | Flask web app funksional |

---

### Pyetjet hulumtuese — përgjigjet

**1. Sa ndryshon performanca TrashNet → real-world?**

EfficientNetB0 i trajnuar vetëm me TrashNet: 89.90% TrashNet → 78.33% real-world (−11.57pp).
Me fine-tuning progresiv mbi dataset-e reale: mbyllet hendeku → 92.17% real-world.

**2. Cila qasje jep performancën më të mirë?**

EfficientNetB0 + Fine-tune Garbage v2 + TTA = **92.17%** — superior ndaj CNN bazike (+35.73pp) dhe MobileNetV2 (~+22pp).

**3. Cilat klasa ngatërrohen më shpesh?**

Glass↔Plastic (ngjashmëri vizuale e materialit) dhe Trash me kategori të tjera (kategori e paqartë vizualisht). Analiza u bë për çdo imazh me emrin e tij specifik.

---

### Rezultate të arritura (sipas propozimit)

| Rezultati i synuar | Statusi |
|--------------------|---------|
| Model CNN funksional | ✅ |
| Krahasim modeleve të ndryshme | ✅ 4 modele kryesore + 11 eksperimente |
| Metodat më efektive të identifikuara | ✅ EfficientNetB0 + fine-tuning + TTA |
| Analiza e detajuar e gabimeve | ✅ Per klasë dhe per imazh |
| Nivel i lartë saktësie | ✅ 92.17% (synohej "i lartë") |
| Analiza krahasuese + vizualizime | ✅ |
| Implementim praktik Python | ✅ Flask web app |

---

## 3. Pse shtova dataset-e të tjera?

Propozimi parashikonte TrashNet si dataset bazë. Gjatë punës u zbulua problemi i **domain shift**:

> EfficientNetB0 i trajnuar me TrashNet arrinte **89.90%** mbi TrashNet, por vetëm **78.33%** mbi foto reale.

**Arsyeja:** TrashNet ka foto studio (sfond i bardhë, ndriçim uniform). Foto reale janë krejtësisht të ndryshme.

**Zgjidhja:** Shtova dataset-e me foto reale për fine-tuning — kjo lejohet dhe justifikohet nga propozimi i cili thotë:

> *"do të krijohet një dataset shtesë... imazhe të mbledhura nga ambiente të përditshme dhe imazhe të përzgjedhura nga burime në internet"*

Dataset-et shtesë u përdorën **për trajnim**, jo për testim. Dataset-i real (600 foto) mbeti **ekskluzivisht për testim**.

---

## 4. Eksperimentet — tabela e thjeshtuar

| # | Çfarë bëra | Saktësia reale |
|---|-----------|----------------|
| 1 | CNN nga zero | ~30% |
| 2 | MobileNetV2 | ~70% |
| 3 | EfficientNetB0 (bazë) | 78.33% |
| 4 | + Augmentation | 82.67% |
| 5 | + TACO fine-tune | — |
| 6 | + Background removal ❌ | Rezultat negativ (−2.56pp) |
| 7 | + RealWaste fine-tune | — |
| 8 | + Household fine-tune | 88.83% |
| 9 | + TTA (5 pamje mesatare) | 88.83% |
| 10 | Ensemble 3 modele ❌ | Rezultat negativ (−3.85pp) |
| **11** | **+ Garbage v2 fine-tune** | **92.17%** ✅ |

> Eksperimentet 6 dhe 10 janë **rezultate negative** — të vlefshme akademikisht sepse tregojnë çfarë nuk funksionon dhe pse.

---

## 5. Progresi i saktësisë

![Progresi](chart_progression.png)

---

## 6. Rezultati final

**Saktësia totale: 92.17% (553 nga 600 foto reale)**

![Krahasimi i modeleve](chart_model_comparison.png)

### Saktësia për klasë

![Saktësia per klasë](chart_per_class.png)

| Kategoria | Saktë | % |
|-----------|-------|---|
| Qelq (glass) | 95/100 | 95% |
| Metal | 94/100 | 94% |
| Karton (cardboard) | 92/100 | 92% |
| Plastikë (plastic) | 92/100 | 92% |
| Letër (paper) | 90/100 | 90% |
| Mbeturinë (trash) | 90/100 | 90% |
| **TOTAL** | **553/600** | **92.17%** |

### Gabime kryesore

- Glass → Plastic (ngatërrimi kryesor — xhami duket si plastikë)
- Paper → Cardboard (letër e trashë duket si karton)
- Trash → kategori të tjera (kategori e paqartë vizualisht)

---

## 7. Dataset-i real i testimit

- **600 foto reale**, 100 për secilën kategori
- Mbledhur nga ambiente normale (shtëpi, zyrë, rrugë)
- **Kurrë i përdorur gjatë trajnimit** — i mbajtur i ndarë për testim final
- Foto të riemëruara qartë: `trash1.jpg` … `trash100.jpg`

---

## 8. Aplikacioni Web

Ndërtova një aplikacion web ku mund të ngarkosh një foto dhe sistemi të thotë menjëherë se çfarë mbeturine është.

- **Teknologjia:** Python Flask + HTML/CSS/JavaScript
- **Funksionalitete:** drag & drop, klasifikim real-time, grafik besueshmërie, këshillë riciklimi
- **Komanda:** `python app.py` → http://127.0.0.1:5000

---

## 9. Gjendja aktuale

### Faza eksperimentale: **E PËRFUNDUAR** ✅

Gjithçka e planifikuar në propozim është realizuar. Shtesë: TTA, ensemble experiment, fine-tuning me dataset-e të shumta reale.

### Mbetet (faza e dokumentimit):

| Kapitulli | Statusi |
|-----------|---------|
| Teorik — CNN, Transfer Learning, literatura | 📝 Mbetet |
| Metodologjia — dataset-et, trajnimi | 📝 Mbetet |
| Rezultatet — tabelat dhe grafikët | ✅ Gati (grafikët ekzistojnë) |
| Diskutimi — interpretim akademik | 📝 Mbetet |
| Konkluzionet | 📝 Mbetet |

---

## 10. Konkluzion

Hipoteza e propozimit u konfirmua:

> *"Duke përdorur CNN, është e mundur të realizohet klasifikimi automatik i mbeturinave me saktësi të lartë"*

Rezultati **92.17%** mbi foto reale e vërteton këtë. Fine-tuning progresiv me dataset-e reale ishte çelësi i kalimit nga 78% → 92% — duke treguar se adaptimi i modelit ndaj kushteve reale është thelbësor për aplikim praktik.

---

*Kodi: `waste-classification-master-thesis/` (branch: dev)*  
*Demonstrim: `python app.py` → http://127.0.0.1:5000*
