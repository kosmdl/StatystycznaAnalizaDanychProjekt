Skrypty do sprawozdania „Przestrzenne zróżnicowanie poziomu życia w województwie małopolskim (2011)”

Zawartość:
  raw.json     – dane pobrane z Banku Danych Lokalnych GUS (2011, 22 powiaty)
  analiza.py   – obliczenia: zmienne X1–X8, unitaryzacja, MS, MR, TOPSIS, grupy, τ Kendalla
  wykresy.py   – rysuje rys. 1–6 (biblioteka matplotlib)

Jak uruchomić:
  1. Zainstaluj Pythona 3 oraz biblioteki:  pip install numpy pandas scipy matplotlib adjustText
  2. W tym folderze uruchom:               python wykresy.py
  3. Obrazki rys1_eda.png ... rys6_bezrobocie.png pojawią się w tym folderze.

Przy pierwszym uruchomieniu skrypt pobierze z internetu plik z granicami powiatów
(powiaty-medium.geojson, ok. 7 MB) potrzebny do kartogramów (rys. 5).
