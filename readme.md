# 🐑 Latxa Trivia – Zientzia Astea 2025 Demoa

**Latxa Trivia** HiTZ taldeak sortutako **EusTrivia** datu-multzoan oinarritutako galdera-erantzun joko interaktiboa da. Proiektu hau **Zientzia Astea 2025**-rako egindako demo bat da, eta helburua da gizakiak **Latxa** adimen artifizialarekin noraino lehiatu daitezkeen erakustea.

![Latxa Trivia interfazearen pantaila-argazkia](https://i.ibb.co/mC1rbt6h/Screenshot-from-2025-11-10-18-40-50.png)

---

## 🎮 Jokoaren dinamika

- Partida bakoitzak **10 galdera** ditu, zailtasun ezberdinetakoak.  
- Galdera bakoitzean erabiltzaileak erantzuna aukeratzen du, eta **Latxa** ere bere erantzuna ematen du denbora errealean.  
- Gainera, **Latxa+web** bertsioa ere badago, hau da, Latxak interneteko bilaketak erabil ditzake **Tavily** bidez informazioa eskuratzeko.  
- Helburua: **Latxa baino galdera gehiago asmatzea** — erakutsi zuk gehiago dakizula!

---

## 🧠 Teknologia

- **EusTrivia** datu-multzoa: [HiTZ / EusTrivia](https://huggingface.co/datasets/HiTZ/EusTrivia)  
- **Latxaren erantzunak**: API deien bidez eskuratzen dira.  
- **Latxa+web agentea**: [LangChain](https://docs.langchain.com/oss/python/langchain/agents)-eko `create_React_agent` erabiliz sortzen da,  
  eta **Tavily** bilaketa-tresna erabiltzen du sarean erantzunak aurkitzeko.  
- **Interfazea**: [Gradio](https://www.gradio.app/) bidez eraikia.  

---

## 📊 Sailkapena

Aplikazioak erabiltzaileen **ranking** edo sailkapen orokorra erakusten du,  
non jokalariek ikus dezaketen nor dagoen Latxa baino jakintsuago.

---

## 📁 Edukia
```
📦 LatxaTrivia
├── latxa_trivia.py   # Aplikazioaren kode nagusia (Gradio interfazea)
├── style.css          # Itxura pertsonalizatzeko estilo-fitxategia
└── latxatrivia.png    # Aplikazioaren logoa
```

---

## 🚀 Exekuzioa

Erabiltzeko:

```bash
python latxa_trivia.py
```
Ondoren, zure nabigatzailean jokoaren interfazea irekiko da.




