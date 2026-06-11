# Projekt: kiedy model powinien powiedzieć „nie wiem”?

Ten projekt realizuje wymagania z pliku `projekt_2_model_calib_and_uncert.pdf`: trenuje model klasyfikujący tweety jako katastroficzne lub niekatastroficzne, diagnozuje nadmierną pewność modelu, kalibruje predykcje i testuje prosty mechanizm odmowy odpowiedzi.

## Co jest zaimplementowane

- Model bazowy: własna implementacja `TF-IDF + Logistic Regression`.
- Dodatkowe modele porównawcze:
  - `TF-IDF + Multinomial Naive Bayes`,
  - `TF-IDF + Linear SVM`, którego margines jest celowo traktowany jako niekalibrowany score.
  - `Average learned token embeddings + MLP`.
- Metryki klasyfikacji: `accuracy`, `precision`, `recall`, `F1`, macierz pomyłek, `log-loss`, `Brier score`.
- Diagnoza kalibracji: reliability diagram oraz `Expected Calibration Error` (`ECE`).
- Przykłady błędnych predykcji z wysoką pewnością.
- Dostępne metody poprawy kalibracji / estymacji niepewności:
  - Temperature Scaling,
  - Label Smoothing,
  - Isotonic Regression,
  - bootstrap ensemble,
  - Platt Scaling jako opcja dodatkowa `--include-platt`,
  - proste split conformal prediction.
- Mechanizm „nie wiem”: jeśli confidence jest mniejsze lub równe progowi `tau`, model odrzuca predykcję.
- Tabele `coverage`, liczby odrzuconych przykładów i accuracy po odrzuceniu.

Projekt nie używa `scikit-learn` ani `matplotlib`, żeby dało się go uruchomić w lekkim środowisku. Wykresy są generowane jako pliki SVG.

Domyślnie uruchamiane są metody zgodne z listą przykładów w PDF-ie: Temperature Scaling, Label Smoothing, Isotonic Regression i Ensemble. Platt Scaling jest zostawiony jako opcja dodatkowa, bo jest popularną metodą kalibracji, ale nie było jej na liście w treści zadania.

## Struktura plików

```text
.
├── run_experiment.py
├── requirements.txt
├── requirements-transformers.txt
├── data/
│   └── README.md
├── reports/
├── src/
│   └── disaster_uncertainty/
│       ├── calibration.py
│       ├── cli.py
│       ├── data.py
│       ├── metrics.py
│       ├── modeling.py
│       └── plots.py
└── tests/
    └── test_project.py
```

## Uruchomienie

Najpierw zainstaluj zależności:

```bash
pip install -r requirements.txt
```

Właściwy eksperyment zakłada plik Kaggle `train.csv`. W tym projekcie dataset jest już w katalogu `nlp-getting-started/`, więc użyj:

```bash
python run_experiment.py --train-csv nlp-getting-started/train.csv --output-dir reports
```

Wersja z dodatkowym Platt Scaling:

```bash
python run_experiment.py --train-csv nlp-getting-started/train.csv --output-dir reports --include-platt
```

Wersja szybka, tylko z rodziną modelu bazowego:

```bash
python run_experiment.py --train-csv nlp-getting-started/train.csv --output-dir reports --skip-extra-models
```

## Opcjonalny eksperyment Transformer

Główny pipeline jest lekki i nie wymaga GPU. Transformery są osobnym eksperymentem, bo wymagają `torch`, `transformers`, pobrania wag modelu i znacznie więcej pamięci.

Instalacja zależności:

```bash
pip install -r requirements-transformers.txt
```

Sensowny mały model do tego projektu:

```bash
python run_transformer_experiment.py \
  --train-csv nlp-getting-started/train.csv \
  --model-name distilbert-base-uncased \
  --output-dir reports_transformers/distilbert \
  --epochs 3 \
  --batch-size 8
```

Szybki test techniczny na małym fragmencie danych:

```bash
python run_transformer_experiment.py \
  --train-csv nlp-getting-started/train.csv \
  --model-name distilbert-base-uncased \
  --output-dir reports_transformers/distilbert_smoke \
  --epochs 1 \
  --batch-size 4 \
  --limit-train 64 \
  --limit-calibration 64 \
  --limit-test 64
```

Llama 3 8B nie jest domyślnie uruchamiana. To duży model generatywny, a nie mały klasyfikator tekstu. Skrypt zatrzyma się bez flagi `--allow-large-model`, żeby przypadkiem nie pobrać i nie próbować trenować modelu, który może nie zmieścić się w pamięci. Jeśli masz dostęp do wag, token Hugging Face i odpowiedni sprzęt, techniczna próba wygląda tak:

```bash
python run_transformer_experiment.py \
  --train-csv nlp-getting-started/train.csv \
  --model-name meta-llama/Meta-Llama-3-8B \
  --output-dir reports_transformers/llama3_8b \
  --allow-large-model \
  --epochs 1 \
  --batch-size 1
```

Jeżeli nie masz jeszcze danych z Kaggle, możesz wykonać test demonstracyjny na małym syntetycznym zbiorze:

```bash
python run_experiment.py --demo --output-dir reports_demo
```

Testy jednostkowe:

```bash
python -m unittest discover -s tests
```

## Wyniki

Po uruchomieniu w katalogu `reports/` powstaną:

- `metrics_summary.csv` - porównanie modeli i kalibratorów,
- `reliability_bins.csv` - dane do reliability diagram,
- `high_confidence_errors.csv` - najbardziej pewne błędne predykcje,
- `rejection_curves.csv` - działanie mechanizmu „nie wiem” dla różnych progów,
- `conformal_prediction.csv` - wyniki prostego split conformal prediction,
- `error_analysis.csv` - wszystkie błędne predykcje z heurystyczną kategorią błędu,
- `error_category_summary.csv` - liczba błędów i błędów wysokiej pewności według kategorii,
- `error_examples.md` - gotowe przykłady błędnych tweetów do prezentacji,
- `plots/reliability_*.svg` - osobne wykresy kalibracji dla rodzin modeli,
- `plots/rejection_*.svg` - osobne wykresy coverage vs accuracy,
- `plots/threshold_tradeoff_*.svg` - prog `tau` vs accuracy, coverage i liczba odrzuconych przykładów,
- `plots/confusion_*.svg` - macierze `TN/FP/FN/TP`,
- `plots/conformal_prediction.svg` - coverage, singleton rate i średni rozmiar zbioru predykcji dla conformal prediction,
- `plots/error_categories_*.svg` - typy błędów wysokiej pewności, np. false alarm, fikcja/rozrywka, keyword trigger,
- `model_comparison.md` - krótki raport tekstowy do prezentacji.

## Jak działa eksperyment

1. Dane są dzielone stratyfikacyjnie na trzy części: treningową, kalibracyjną i testową. Zbiór kalibracyjny jest osobny, bo nie chcemy dopasowywać kalibratorów na danych testowych.
2. Model bazowy zamienia tekst na cechy TF-IDF. Wektoryzator liczy unigramy i bigramy, waży je przez IDF i normalizuje każdy dokument.
3. Regresja logistyczna uczy się wag przez stochastyczny spadek gradientu. Jej surowy wynik, czyli logit, jest zamieniany funkcją sigmoid na liczbę z przedziału 0-1.
4. Ta liczba wygląda jak prawdopodobieństwo klasy `1`, ale w praktyce może być źle skalibrowana. Dlatego mierzymy, czy confidence odpowiada rzeczywistej częstości poprawnych odpowiedzi.
5. `ECE` dzieli przykłady na koszyki według confidence. Dla każdego koszyka porównuje średnią pewność modelu ze średnią poprawnością, a potem liczy ważoną średnią lukę.
6. Temperature Scaling dzieli logity przez temperaturę `T`. Duże `T` łagodzi pewność modelu, małe `T` ją wzmacnia.
7. Label Smoothing trenuje wariant regresji logistycznej na łagodniejszych etykietach, np. `1` zmienia się w `0.95`, a `0` w `0.05`.
8. Isotonic Regression uczy monotonicznego, schodkowego przekształcenia prawdopodobieństw.
9. Bootstrap ensemble trenuje kilka modeli na losowanych próbkach danych i uśrednia ich predykcje.
10. MLP na embeddingach uczy wektor dla każdego tokenu, uśrednia tokeny w tweecie i klasyfikuje wynik jedną warstwą ukrytą.
11. Dodatkowo porównywane są Naive Bayes i Linear SVM, żeby zobaczyć, czy problem kalibracji wygląda podobnie dla innych rodzin modeli.
12. Mechanizm „nie wiem” używa confidence wybranej klasy: `max(p, 1-p)`. Jeśli confidence jest większe niż próg `tau`, model odpowiada. W przeciwnym razie odrzuca przykład.
13. Split conformal prediction używa zbioru kalibracyjnego do zbudowania zbioru możliwych klas, np. `{0}`, `{1}` albo `{0, 1}`, z kontrolowanym poziomem ryzyka `alpha`.
14. Analiza błędów przypisuje błędne predykcje do prostych, interpretowalnych kategorii. Kategorie są heurystyczne, ale pomagają pokazać, dlaczego model był nadmiernie pewny.

Najważniejsza interpretacja: po zwiększeniu progu `tau` zwykle rośnie accuracy na zaakceptowanych przykładach, ale maleje coverage, czyli model odpowiada na mniejszą część danych.
