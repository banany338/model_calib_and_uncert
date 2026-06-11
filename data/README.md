# Dane

Do właściwego eksperymentu pobierz plik `train.csv` z konkursu Kaggle:

Natural Language Processing with Disaster Tweets

Umieść go tutaj:

```text
data/train.csv
```

Wymagane kolumny to:

- `text` - treść tweeta,
- `target` - etykieta: `1` oznacza prawdziwą katastrofę, `0` oznacza brak prawdziwej katastrofy.

Jeżeli chcesz tylko sprawdzić, czy kod działa bez danych z Kaggle, uruchom:

```bash
python run_experiment.py --demo
```
