# BullsEye-CV

A BullsEye-CV (Bullseye Computer Vision) egy kameraalapú darts pontozó rendszer. Három USB-kamera képéből valós időben felismeri a táblába érkező nyilakat, kiszámítja a pontértéket, és tetszőleges böngészőből (akár mobiltelefonról) is elérhető webes felületen vezeti a játékot és a statisztikát.

A projekt egy BSc szakdolgozat gyakorlati része, melynek címe: *Kameraalapú objektumfelismerő és statisztikai rendszer megvalósítása beágyazott eszközökön sportanalitikai célokra* (Kovács Tamás, Gábor Dénes Egyetem, 2026).

## Főbb jellemzők

- három kamera, YOLOv8n nyílhegy-detektor, síkbeli homográfia (képpont → tábla-koordináta mm-ben → pontérték)
- többkamerás adategyesítés, nyílkövetés, bizonytalansági jelzés (szektorhatár, csak két kamera látta, alacsony bizonyosság)
- X01 játékok (301, 501, 701 stb.), 1–4 játékos, több leg, visszavonás, kiszálló-javaslat
- mobilbarát webes felület (Server-Sent Events), SQLite alapú statisztika
- az USB-kamerák automatikus azonosítása indításkor
- OpenVINO-gyorsítás az Intel integrált GPU-ján
- kamera és modell nélküli szimulációs mód a kipróbáláshoz

## Követelmények

**Kipróbáláshoz és teszteléshez** (szimulációs mód) nem szükséges célhardver, bármilyen számítógép megfelel.

**Valós használathoz:**

| | |
|---|---|
| Feldolgozóegység | Intel N100 processzoros mini PC (a fejlesztés és a mérések egy asztali számítógépen és egy Chuwi Larkbox X x86-os miniPC-n készültek) |
| Operációs rendszer | Windows 11. A szimulációs mód és az egységtesztek Linuxon is futnak, a telepítő szkriptek viszont Windowsra készültek. |
| Kamerák | 3 db USB-kamera, 1280 × 720 felbontás, MJPG; a tábla köré szerelve, egyenletes megvilágítással (LED). |
| Tábla | szabványos szizál, lehetőség szerint vékony drótos darts-tábla |
| Python | 3.12-n történt a fejlesztés, de 3.13-n is futtatható |
| Gyorsítás (opcionális) | Intel integrált GPU az OpenVINO-hoz; enélkül a modell CPU-n is fut, de lassabban |

A hardveres környezet (3D-nyomtatott tartógyűrű, LED-világítás, kamerák elrendezése) leírása a szakdolgozat 3. fejezetében található.

## Telepítés

```
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Linuxon és macOS-en: `python3 -m venv .venv`, majd `source .venv/bin/activate`.

A `requirements.txt` az `ultralytics` csomagot rögzített verzióval kéri (8.4.174), mert a mellékelt modell ezzel készült.

## Gyors kipróbálás (hardver nélkül)

```
python server.py --sim
```

Ezután a böngészőben: <http://localhost:8000>. A szimulációs mód kamera és modell nélkül, szimulált dobásokkal működik.

Egységtesztek (73 db, kb. 20 másodperc):

```
python -m unittest discover -s . -p "test_*.py"
```

## Futtatás valós kamerákkal

A betanított modell a tárolóban van: `runs/tips/weights/best.pt` (YOLOv8n, egyetlen osztály: nyílhegy, 960 px bemenet). A `calib_board/board_calib.json` a szerző hardveréhez tartozó kalibráció; más kamera- vagy táblaelrendezéshez újra kell kalibrálni (lásd lentebb).

**Indítás CPU-n:**

```
python server.py --weights runs\tips\weights\best.pt --cams 0 1 2 --auto-devices
```

**Indítás Intel integrált GPU-n (OpenVINO, gyorsabb).** Először exportáld a modellt, majd indítsd a szervert:

```
yolo export model=runs/tips/weights/best.pt format=openvino imgsz=960
python server.py --weights runs\tips\weights\best_openvino_model --device intel:gpu --fallback-weights runs\tips\weights\best.pt --cams 0 1 2 --auto-devices --fast-confirm 0.75
```

A `--fallback-weights` miatt hiba esetén a program a PyTorch-modellre, CPU-ra vált, és ezt figyelmeztetéssel jelzi.

**Kezelőfelület:** a böngészőben `http://<a gép IP-címe>:8000` (telefonról is, ugyanazon a helyi hálózaton). A szerver alapból az összes hálózati interfészen figyel (`--host 0.0.0.0`) és nem kér jelszót, ezért ne tedd elérhetővé az internetről.

Gyakoribb kapcsolók: `--cams` (kamerák sorszámai), `--devices` (USB-eszközindexek, ha az automatikus azonosítás nem használható), `--port`, `--data-dir` (játékok és statisztika helye, alapból `data/`), `--no-power` (a weboldalról ne lehessen leállítani a gépet). Az összes kapcsoló: `python server.py --help`.

## Állandó üzem (Windows)

- `start_darts.bat`: elindítja a szervert, és összeomlás esetén 5 másodperc múlva újraindítja; a napló a `logs\server.log` fájlba kerül. Ha a leállítást a weboldalról kérik, nem indul újra. A fájlban szereplő `--devices 1 0 2` a szerző gépére jellemző tartalék sorrend, szükség esetén írd át.
- `install_autostart_windows.ps1` (rendszergazdai PowerShellben): bejelentkezéskor induló feladatot hoz létre, tűzfalszabályt a 8000-es portra (csak a helyi alhálózatról), és kikapcsolja az altatást és a hibernálást. A gépnek automatikusan be kell jelentkeznie (`netplwiz`), különben a feladat nem indul.

## Kalibráció új hardverhez

Szükséges egy sakktábla-minta (9 × 6 belső sarok, 25 mm-es négyzetek). A `capture.py` ablakában az `1`, `2`, `3` billentyű vált módot (sakktábla, üres tábla, nyilak), az `s` ment, a `q` kilép.

```
python capture.py --session kalib1
python calibrate_intrinsics.py --session kalib1 --out calib
python calibrate_board.py --session kalib1 --intrinsics calib\intrinsics.json --out calib_board
```

Figyelem: a két kalibráló szkript felülírja a `calib\` és a `calib_board\` mappa tartalmát, előtte mentsd el az eredetit. A részletes eljárást a szakdolgozat 3.10. alfejezete írja le.

## Modell betanítása (opcionális)

Saját tanítóadatból (a `capture.py` 3-as módjával rögzített felvételek, kézi bejelöléssel):

```
python annotate_tips.py --session kalib1
python train_tips.py --session kalib1
```

A kész modell a `runs\tips\weights\best.pt` helyre kerül (a `--final` kapcsolóval a `runs\tips_final\weights\best.pt` mappába).

## Fájlok

| Fájl / mappa | Feladata |
|---|---|
| `server.py` | belépési pont: webszerver (Starlette), eseményküldés a böngészőnek, játék- és statisztikakezelés |
| `live_darts.py` | képfeldolgozó ciklus: kameraforrás, mozgásérzékelő kapu, YOLO-detektor; konzolos módban önállóan is indítható |
| `engine.py` | a szerver és a képfeldolgozás közti réteg: valós (`LiveEngine`) és szimulációs (`SimEngine`) működés |
| `tracker.py` | nyílkövetés, megerősítés, bizonytalansági jelzés |
| `tipcommon.py` | közös segédfüggvények: kalibráció betöltése, vetítés, többkamerás egyesítés, címkék |
| `dartboard.py` | tábla-geometria és pontérték-meghatározás |
| `game.py` | X01 játéklogika (eseményalapú állapotkezelés) |
| `stats.py` | SQLite statisztika és mérőmodul |
| `camid.py` | a kamerák automatikus azonosítása indításkor |
| `static/index.html` | a mobilbarát webes felület (egyetlen oldal) |
| `capture.py` | képrögzítés kalibrációhoz és tanításhoz |
| `calibrate_intrinsics.py` | belső (optikai) kalibráció sakktáblával |
| `calibrate_board.py` | külső kalibráció, tábla-homográfia |
| `annotate_tips.py` | nyílhegyek kézi bejelölése (tanítóadat készítése) |
| `train_tips.py` | a YOLOv8n modell betanítása |
| `evaluate_tips.py`, `evaluate_labels.py` | a detektálás és a pontszámítás kiértékelése rögzített felvételeken |
| `bench_backends.py` | PyTorch, OpenVINO és ONNX futásidő-összehasonlítás |
| `log_resources.py` | erőforrás-használat naplózása (mérésekhez) |
| `kiegyenesites_abra.py` | a torzításmentesítés szemléltetése (kamerakép előtte és utána) |
| `test_*.py` | egységtesztek |
| `calib/`, `calib_board/` | a szerző hardveréhez tartozó kalibrációs adatok |
| `runs/tips/weights/best.pt` | a betanított nyílhegy-detektor |
| `requirements.txt` | Python-függőségek |
| `start_darts.bat`, `install_autostart_windows.ps1` | Windows-os indító és automatikus indítást beállító szkript |
| `darts.service` | Linuxos (systemd) szolgáltatásminta |

Futás közben létrejövő mappák (nincsenek a tárolóban): `data/` (játékok, statisztika, eseménynapló), `logs/`, `captures/`, `tipdata/`, `measurements/`.

## Licenc és hivatkozások

- A projekt az [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) könyvtárat használja, amely AGPL-3.0 licenc alatt áll. A mellékelt modellfájl (`best.pt`) is ezt a licencet jelöli. A tároló licence: AGPL-3.0.
- A 3D-nyomtatott tartógyűrű az OMMO *Dartboard Ring for Autodarts* modelljén alapul ([MakerWorld](https://makerworld.com/hu/models/1124401-ommo-dartboard-ring-for-autodarts)). A tartógyűrű modellje nem része a tárolónak.
- A projekt önálló fejlesztés, nem áll kapcsolatban az Autodarts vagy a Winmau cégekkel.
