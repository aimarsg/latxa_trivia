import os
import json
import time
import base64
import random
import threading
from typing import List, Dict, Tuple
from urllib.parse import unquote

from datasets import load_dataset, concatenate_datasets

import gradio as gr
from gradio_modal import Modal
import requests

from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

import pandas as pd


# set tavily api key env var
os.environ["TAVILY_API_KEY"] = "insert api key"


# =========================
# Configuración
# =========================

NUM_QUESTIONS = 10
LEADERBOARD_FILE = "leaderboard.jsonl"
LEADERBOARD_LOCK = threading.Lock()

OPENAI_MODEL = "Latxa-Llama-3.1-70B-Instruct-exp_2_101"

# =========================
# Utilidades
# =========================

def today():
    return time.strftime("%A %B %e, %Y", time.gmtime())

def b64_decode(s: str) -> str:
    try:
        return base64.b64decode(s).decode("utf-8")
    except Exception:
        # Fallback: intentar url-decode si viniera con otro encoding
        return unquote(s)

def shuffle_options(correct: str, incorrects: List[str]) -> Tuple[List[str], int]:
    options = incorrects + [correct]
    random.shuffle(options)
    correct_index = options.index(correct)
    return options, correct_index

def normalize_choice_letter(text: str) -> str:
    if not text:
        return ""
    t = text.strip().upper()
    # Aceptar 'A', 'B', 'C', 'D' o con texto como "A) ..."
    if t and t[0] in ["A", "B", "C", "D"]:
        return t[0]
    return ""

def choice_from_index(idx: int) -> str:
    return ["A", "B", "C", "D"][idx]

def index_from_choice_letter(letter: str) -> int:
    mapping = {"A": 0, "B": 1, "C": 2, "D": 3}
    return mapping.get(letter, -1)

def ensure_leaderboard_file():
    if not os.path.exists(LEADERBOARD_FILE):
        with open(LEADERBOARD_FILE, "w", encoding="utf-8") as f:
            pass

def read_leaderboard() -> List[Dict]:
    ensure_leaderboard_file()
    rows = []
    with LEADERBOARD_LOCK:
        with open(LEADERBOARD_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    return rows

def append_leaderboard(entry: Dict):
    ensure_leaderboard_file()
    with LEADERBOARD_LOCK:
        with open(LEADERBOARD_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

# =========================
# Preguntas: EusTrivia
# =========================


# descargar eustrivia desde huggingface
dataset_eustrivia = load_dataset("HiTZ/EusTrivia", split="test")
num_instances = dataset_eustrivia.num_rows

def fetch_questions(amount: int = NUM_QUESTIONS) -> List[Dict]:
    """
    Devuelve lista de dicts:
    {
      "question": str,
      "options": [str, str, str, str],
      "correct_index": int,
      "category": str,
      "difficulty": str
    }
    """
    try:
        # obtener 'amount' preguntas aleatorias del dataset, %60 erraza y %40 zaila
        amount_easy = int(amount * 0.6)
        amount_hard = amount - amount_easy
        easy_questions = dataset_eustrivia.filter(lambda x: x['difficulty'] == 'erraza').shuffle(seed=random.randint(0, num_instances)).select(range(amount_easy))
        hard_questions = dataset_eustrivia.filter(lambda x: x['difficulty'] == 'zaila').shuffle(seed=random.randint(0, num_instances)).select(range(amount_hard))
        questions_ = concatenate_datasets([easy_questions, hard_questions])

        questions = []
        for item in questions_:
            q = item['question']
            candidates = item['candidates']
            corr_idx = item['answer']
            corr = candidates[corr_idx]
            incs = [x for x in candidates if x != corr]
            options, correct_index = shuffle_options(corr, incs)
            questions.append({
                "question": q,
                "options": options,
                "correct_index": correct_index,
                "category": item.get("category", "General Knowledge"),
                "difficulty": item.get("difficulty", "easy"),
            })

        # Fallback si la API devolvió vacío
        if not questions:
            questions = _fallback_questions()
        return questions
    except Exception as e:
        print(f"[latxa_trivia] Error fetching questions: {e}")
        # Fallback offline
        return _fallback_questions()

def _fallback_questions() -> List[Dict]:
    print("[latxa_trivia] Using fallback questions.")
    seed = [
        {
            "question": "¿Cuál es la capital de Italia?",
            "correct": "Roma",
            "incorrects": ["Milán", "Venecia", "Florencia"],
            "category": "General Knowledge",
            "difficulty": "easy",
        },
        {
            "question": "¿Qué planeta es conocido como el planeta rojo?",
            "correct": "Marte",
            "incorrects": ["Venus", "Júpiter", "Mercurio"],
            "category": "Science",
            "difficulty": "easy",
        },
        {
            "question": "¿Cuál es el océano más grande del planeta?",
            "correct": "Océano Pacífico",
            "incorrects": ["Océano Atlántico", "Océano Índico", "Océano Ártico"],
            "category": "Geography",
            "difficulty": "easy",
        },
        {
            "question": "¿Quién pintó La noche estrellada?",
            "correct": "Vincent van Gogh",
            "incorrects": ["Pablo Picasso", "Claude Monet", "Salvador Dalí"],
            "category": "Art",
            "difficulty": "medium",
        },
        {
            "question": "¿En qué año comenzó la Segunda Guerra Mundial?",
            "correct": "1939",
            "incorrects": ["1941", "1936", "1945"],
            "category": "History",
            "difficulty": "medium",
        },
        {
            "question": "¿Qué gas es esencial para la respiración humana?",
            "correct": "Oxígeno",
            "incorrects": ["Nitrógeno", "Dióxido de carbono", "Hidrógeno"],
            "category": "Science",
            "difficulty": "easy",
        },
        {
            "question": "¿Cuál es el metal más ligero?",
            "correct": "Litio",
            "incorrects": ["Aluminio", "Magnesio", "Sodio"],
            "category": "Science",
            "difficulty": "medium",
        },
        {
            "question": "¿Cuál es la moneda de Japón?",
            "correct": "Yen",
            "incorrects": ["Won", "Dólar", "Yuan"],
            "category": "Economy",
            "difficulty": "easy",
        },
        {
            "question": "¿Qué idioma tiene más hablantes nativos?",
            "correct": "Chino mandarín",
            "incorrects": ["Inglés", "Español", "Hindustani"],
            "category": "Culture",
            "difficulty": "medium",
        },
        {
            "question": "¿Cuál es el elemento químico con símbolo Au?",
            "correct": "Oro",
            "incorrects": ["Plata", "Cobre", "Plomo"],
            "category": "Science",
            "difficulty": "easy",
        },
    ]
    out = []
    for item in seed[:NUM_QUESTIONS]:
        options, correct_index = shuffle_options(item["correct"], item["incorrects"])
        out.append({
            "question": item["question"],
            "options": options,
            "correct_index": correct_index,
            "category": item["category"],
            "difficulty": item["difficulty"],
        })
    return out

# =========================
# LLM
# =========================

_openai_client = None
def _get_openai_client():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    try:
        from openai import OpenAI 
        _openai_client = OpenAI(
            base_url="http://localhost:8002/v1", 
            api_key="EMPTY", 
        )
        return _openai_client
    except Exception:
        return None
        

class TavSearch(BaseModel):
    query: str = Field(description=("Search query to search in the internet."))


_latxa_agent = None
def _get_latxa_agent():
    global _latxa_agent
    if _latxa_agent is not None:
        return _latxa_agent
    try:
        model = ChatOpenAI(
            model=OPENAI_MODEL,
            temperature=0.8,
            top_p=0.9,
            timeout=20,
            max_retries=2,
            api_key="EMPTY", 
            base_url="http://localhost:8002/v1",
        )
        search = TavilySearch(
            max_results=7,  # Maximum number of results to return
            include_answer="advanced",  # Include the answer in the results (LMM Generated Answer)
            include_raw_content=False,  # Include raw content from each result ("text")
            search_depth="basic",    # 'advanced' for better results (2 credits per query), 'basic' for basic results (1 credit per query)
            include_favicon=False,
            time_range=None
        )
        search.name = "tavily_search"
        search.description = "General-purpose search engine that queries the open web. Use it whenever the user’s question requires information that is not part of your internal knowledge, or when broader, up-to-date context is needed. Input: A natural-language query in Basque, that best captures the information need. \n\n Example: User question: \"Zenbat biztanle ditu Bilbok 2025ean?\"\n Tool input: \"Bilboko biztanleria 2025 datu ofizialak\" \n Output: An automatically generated answer together with a list of web results about Bilbao's population, together with an auto-generated answer based on the retrieved results."
        search.args_schema = TavSearch
        _latxa_agent = create_react_agent(
            model=model,
            tools=[search],
            prompt="You are a Basque Trivia expert. Your task is to answer multiple-choice questions about Basque culture, history, geography, language, traditions, and people. Each question will provide four possible answers: A, B, C, and D. Your only output must be one of the letters: A, B, C, or D. Do not include any explanation, reasoning, or additional text. If you are not certain about the answer, you may use the tavily_search tool to look up accurate and recent information on the web. After using the tool, choose the most likely correct answer (A, B, C, or D). Never explain your reasoning. Never output anything other than A, B, C, or D. Example: Q: What is the capital of the Basque Country? A) Pamplona B) Vitoria-Gasteiz C) Bilbao D) San Sebastián Correct output: B"
        )
        return _latxa_agent
    except Exception as e:
        print(f"[latxa_trivia] Failed to create Latxa agent: {e}")
        return None


def call_agent_choice_spanish(question: str, options: List[str]) -> str:
    """
    Devuelve 'A'|'B'|'C'|'D' (o '' si falla).
    """
    letters = ["A", "B", "C", "D"]
    opts_text = "\n".join([f"{letters[i]}) {options[i]}" for i in range(len(options))])
    prompt = (
        "You are a Basque Trivia contestant. Answer ONLY with the letter of the correct option "
        "(A, B, C or D), without explanation.\n\n"
        f"Question: {question}\n\nOptions:\n{opts_text}\n\nAnswer:"
    )
    agent = _get_latxa_agent()
    if agent is not None:
        try: 
            resp = agent.invoke(
                {"messages": [{"role": "user", "content": prompt}]}
            )
            response_text = resp["messages"][-1].content.strip()
            print(response_text)
            try:
                # Find the ToolMessage
                tool_message = next(
                    (m for m in resp["messages"] if hasattr(m, "name") and m.name == "tavily_search"),
                    None
                )
                if tool_message:
                    tool_output = json.loads(tool_message.content)
                    print(tool_output["answer"])
            except Exception as e:
                print(f"[latxa_trivia] Failed to parse tool output: {e}")
                pass
            letter = normalize_choice_letter(response_text)
            print(letter)
            if letter in letters:
                return letter
        except Exception as e:
            print(f"[latxa_trivia] Agent call failed: {e}")
            pass
    
    print("[latxa_trivia] Agent call failed, using random choice.")
    return random.choice(letters)
        

def call_llm_choice_spanish(question: str, options: List[str]) -> str:
    """
    Devuelve 'A'|'B'|'C'|'D' (o '' si falla).
    """
    letters = ["A", "B", "C", "D"]
    opts_text = "\n".join([f"{letters[i]}) {options[i]}" for i in range(len(options))])
    prompt = (
        "You are a Basque Trivia contestant. Answer ONLY with the letter of the correct option "
        "(A, B, C or D), without explanation.\n\n"
        f"Question: {question}\n\nOptions:\n{opts_text}\n\nAnswer:"
    )

    # 1) OpenAI Responses API
    client = _get_openai_client()
    if client is not None:
        try:
            resp = client.chat.completions.create(
                model=OPENAI_MODEL,
                max_tokens=3,
                temperature=0,
                messages=[
                    {"role": "system", "content": "You are a helpful Artificial Intelligence assistant called Latxa, created and developed by HiTZ, the Basque Center for Language Technology research center. The user will engage in a multi-round conversation with you, asking initial questions and following up with additional related questions. Your goal is to provide thorough, relevant and insightful responses to help the user with their queries. Every conversation will be conducted in standard Basque, this is, the first question from the user will be in Basque, and you should respond in formal Basque as well. Conversations will cover a wide range of topics, including but not limited to general knowledge, science, technology, entertainment, coding, mathematics, and more. Today is {date}.".format(date=today())},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            text = resp.choices[0].message.content.strip()
            letter = normalize_choice_letter(text)
            if letter in letters:
                return letter
        except Exception as e:
            print(f"[latxa_trivia] LLM call failed: {e}")
            pass

    # 2) Fallback aleatorio
    print("[latxa_trivia] LLM call failed, using random choice.")
    return random.choice(letters)

# =========================
# Lógica de juego (estado)
# =========================

def init_game_state(username: str) -> Dict:
    return {
        "username": username,
        "questions": fetch_questions(NUM_QUESTIONS),
        "index": 0,
        "user_score": 0,
        "model_score": 0,
        "modelRAG_score": 0,
        "locked": False,  # para bloquear mientras se evalúa
        "revealed": False,  # si ya se mostró la respuesta
        "finished": False,
    }

def current_question(state: Dict) -> Dict:
    return state["questions"][state["index"]]

def evaluate_round(state: Dict, user_choice_letter: str) -> Dict:
    q = current_question(state)
    correct_idx = q["correct_index"]
    correct_letter = choice_from_index(correct_idx)

    # LLM responde
    model_letter = call_llm_choice_spanish(q["question"], q["options"])
    model_idx = index_from_choice_letter(model_letter)

    # RAG responde
    modelRAG_letter = call_agent_choice_spanish(q["question"], q["options"])
    modelRAG_idx = index_from_choice_letter(modelRAG_letter)

    # Actualizar marcadores
    user_correct = (user_choice_letter == correct_letter)
    model_correct = (model_idx == correct_idx)
    modelRAG_correct = (modelRAG_idx == correct_idx)

    if user_correct:
        state["user_score"] += 1
    if model_correct:
        state["model_score"] += 1
    if modelRAG_correct:
        state["modelRAG_score"] +=1

    state["revealed"] = True
    return {
        "correct_letter": correct_letter,
        "model_letter": model_letter,
        "modelRAG_letter": modelRAG_letter,
        "user_correct": user_correct,
        "model_correct": model_correct,
        "modelRAG_correct": modelRAG_correct,
    }

def advance_or_finish(state: Dict):
    state["index"] += 1
    state["revealed"] = False
    if state["index"] >= len(state["questions"]):
        state["finished"] = True

def record_result_to_leaderboard(state: Dict):
    entry = {
        "username": state["username"],
        "user_score": state["user_score"],
        "model_score": state["model_score"],
        "modelRAG_score": state["modelRAG_score"],
        "questions": len(state["questions"]),
        "timestamp": int(time.time()),
    }
    append_leaderboard(entry)

def compute_leaderboard_tables(rows: List[Dict]) -> Tuple[List[List], List[List]]:
    """
    Devuelve dos tablas en formato lista de listas:
    - Top puntuaciones recientes (username, user, model, preguntas, fecha)
    - Ranking acumulado por usuario (jugadas, victorias_usuario, victorias_modelo, empates, mejor_puntuación)
    """
    # Top recientes por user_score desc, timestamp desc
    all_sorted = sorted(rows, key=lambda x: (x.get("user_score", 0), x.get("timestamp", 0)), reverse=True)
    recent = []
    seen_users = set()
    for r in all_sorted:
        username = r.get("username", "")
        if username not in seen_users:
            recent.append(r)
            seen_users.add(username)
        if len(recent) >= 25:
            break
    top_table = [["Erabiltzailea", "Erabiltzaileak asmatutakoak", "Latxak asmatutakoak", "Galderak guztira", "Data"]]
    for r in recent:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(r.get("timestamp", 0)))
        top_table.append([
            r.get("username", ""),
            r.get("user_score", 0),
            r.get("model_score", 0),
            r.get("questions", 0),
            ts,
        ])

    # Acumulado por usuario
    agg = {}
    for r in rows:
        u = r.get("username", "")
        if not u:
            continue
        d = agg.setdefault(u, {"plays": 0, "user_wins": 0, "model_wins": 0, "draws": 0, "best": 0})
        d["plays"] += 1
        us = r.get("user_score", 0)
        ms = r.get("model_score", 0)
        if us > ms:
            d["user_wins"] += 1
        elif ms > us:
            d["model_wins"] += 1
        else:
            d["draws"] += 1
        d["best"] = max(d["best"], us)

    agg_rows = sorted(agg.items(), key=lambda kv: (kv[1]["user_wins"], kv[1]["best"]), reverse=True)
    rank_table = [["Erabiltzailea", "Jokatutako partidak", "Erabiltzailearen garaipenak", "Latxaren garaipenak", "Berdinketa", "Puntuazio hoberena"]]
    for user, d in agg_rows[:250]:
        rank_table.append([
            user, d["plays"], d["user_wins"], d["model_wins"], d["draws"], d["best"]
        ])
    return top_table, rank_table

# =========================
# Callbacks de UI
# =========================

def on_start(name, browser_name, state):
    # Preferir nombre de la caja si se ha escrito; si no, usar el del navegador
    username = (name or browser_name or "").strip()
    if not username:
        return (
            gr.Tabs(),  # no cambiar pestaña
            gr.Markdown("⚠️ Sartu izen bat hasi aurretik.", visible=True),
            state,
            gr.BrowserState([""]),  # no guardar
        )
    # Inicializar juego
    new_state = init_game_state(username)
    # Guardar en BrowserState
    bs = gr.BrowserState([username])
    # Intentar ir a la pestaña de juego (puede fallar según versión, el usuario puede pulsar manualmente)
    return (
        gr.Tabs(selected="juego"),
        gr.Markdown(f"✅ Ongi etorri, {username}. Joan 'Jokatu partida' atalera partida hasteko.", visible=True),
        new_state,
        bs,
    )

def render_round(state):
    if not state or not state.get("questions"):
        state = init_game_state("Invitado")
    idx = state["index"]
    total = len(state["questions"])
    finished = state.get("finished", False)

    header = ""
    q_text = ""
    opts = []
    meta = ""
    if finished:
        # Resultado final
        u = state["user_score"]
        m = state["model_score"]
        mr = state["modelRAG_score"]
        if u > m and u > mr:
            result = "🎉 Irabazi duzu!"
        elif m > u or mr > u:
            result = "🤖 Latxak irabazi du!"
        else:
            result = "🤝 Berdinketa."
        header = f"### Azken emaitza — Zu: {u} · Latxa: {m} · Latxa+Web: {mr} — {result}"
        q_text = "Partida amaiera."
        meta = ""
        opts = []
    else:
        q = current_question(state)
        header = f"Galdera {idx+1} / {total} — Kategoria: {q['category']} — Zailtasuna: {q['difficulty']}"
        q_text = q["question"]
        opts = [f"{choice_from_index(i)}) {c}" for i, c in enumerate(q["options"])]
        meta = ""

    # Reset de visibilidad/estado por render
    return (
        header,
        q_text,
        gr.Radio(choices=opts, value=None, interactive=(not state.get("revealed", False) and not finished)),
        f"### Puntuazioa -- Zu: {state['user_score']} | Latxa: {state['model_score']} | Latxa+Web: {state['modelRAG_score']}",
        gr.Markdown("", visible=False),  # respuesta correcta
        gr.Markdown("", visible=False),  # feedback
        gr.Markdown("", visible=False),  # modelo
        gr.Markdown("", visible=False),  # modelo RAG
        gr.Button(" 1️⃣ Bidali", interactive=(not state.get("revealed", False) and not finished), visible=(not finished)),
        gr.Button(" 2️⃣ Hurrengoa", interactive=False, visible=(not finished)),
        gr.Button("Berriz hasi", visible=finished),
    )

def on_submit(user_choice_text, state):
    if state.get("locked", False) or state.get("finished", False):
        return (gr.Markdown(), gr.Markdown(), gr.Button(), gr.Button(), gr.Radio())
    if user_choice_text is None:
        return (
            gr.Markdown("⚠️ Selecciona una opción antes de enviar.", visible=True),
            gr.Markdown("", visible=False),
            gr.Markdown("", visible=False),
            gr.Markdown("", visible=False),
            gr.Button(),
            gr.Button(),
            gr.Radio()
        )

    state["locked"] = True
    # quitar los primeros tres caracteres de user_choice_text
    user_choice_text = user_choice_text[3:]
    # Mapear elección del usuario a letra
    # Encontrar índice de la opción elegida
    q = current_question(state)
    try:
        user_idx = q["options"].index(user_choice_text)
    except ValueError:
        user_idx = -1
    user_letter = choice_from_index(user_idx) if user_idx >= 0 else ""

    outcome = evaluate_round(state, user_letter)

    # Preparar feedback
    correct_letter = outcome["correct_letter"]
    correct_idx = index_from_choice_letter(correct_letter)
    correct_text = q["options"][correct_idx]

    user_mark = "✅" if outcome["user_correct"] else "❌"
    model_mark = "✅" if outcome["model_correct"] else "❌"
    modelRAG_mark = "✅" if outcome["modelRAG_correct"] else "❌"

    feedback = (
        f"{user_letter}) {q['options'][user_idx]} {user_mark}"
    )
    model_reveal = f"{outcome['model_letter']}) {q['options'][index_from_choice_letter(outcome['model_letter'])]} {model_mark}"
    modelRAG_reveal = f"{outcome['modelRAG_letter']}) {q['options'][index_from_choice_letter(outcome['modelRAG_letter'])]} {modelRAG_mark}"

    correct_answer = f"Erantzun zuzena: {correct_letter}) {correct_text}"

    # Deshabilitar radio y botón Enviar; habilitar Siguiente
    state["locked"] = False
    return (
        gr.Markdown(correct_answer, visible=True),
        gr.Markdown(feedback, visible=True),
        gr.Markdown(model_reveal, visible=True),
        gr.Markdown(modelRAG_reveal, visible=True),
        gr.Button(interactive=False),
        gr.Button(interactive=True),
        gr.Radio(interactive=False),
    )

def on_next(state):
    if state.get("finished", False):
        return render_round(state)
    advance_or_finish(state)
    if state["finished"]:
        # guardar resultado final en leaderboard
        record_result_to_leaderboard(state)
    return render_round(state)

def on_restart(state):
    username = state.get("username", "Invitado")
    new_state = init_game_state(username)
    return (new_state, *render_round(new_state))

def refresh_leaderboard():
    rows = read_leaderboard()
    top_table, rank_table = compute_leaderboard_tables(rows)
    return pd.DataFrame(top_table[1:], columns=top_table[0]), pd.DataFrame(rank_table[1:], columns=rank_table[0])

    # Convertir a Markdown simple para buen render en móviles
    def to_md(table):
        if not table:
            return "Ez dago daturik."
        headers = table[0]
        lines = [" | ".join(headers), " | ".join(["---"] * len(headers))]
        for row in table[1:]:
            lines.append(" | ".join(str(c) for c in row))
        return "\n".join(lines)
    return gr.Markdown(to_md(top_table)), gr.Markdown(to_md(rank_table))


def check_and_show_modal(state):
    """Función separada que controla el modal"""
    if state.get("finished", False):
        u = state["user_score"]
        m = state["model_score"]
        mr = state["modelRAG_score"]
        
        if u > m and u > mr:
            content = f"## 🎉 Irabazi duzu!\n\n**Zu:** {u} | **Latxa:** {m} | **Latxa+Web:** {mr}"
        elif m > u or mr > u:
            content = f"## 🤖 Latxak irabazi du!\n\n**Zu:** {u} | **Latxa:** {m} | **Latxa+Web:** {mr}"
        else:
            content = f"## 🤝 Berdinketa\n\n**Zu:** {u} | **Latxa:** {m} | **Latxa+Web:** {mr}"

        return Modal(visible=True), content

    return Modal(visible=False), ""

# =========================
# UI
# =========================

with gr.Blocks(css_paths="./style.css", title="Trivia vs LLM") as demo:
    # Estado
    session_state = gr.State({})
    browser_state = gr.BrowserState([""])
    
    markdown_content = """
    <div class="responsive-header">
    <img src="https://i.ibb.co/hxVDy2CH/latxatrivia.png" alt="Latxa Trivia logo">
    <h1>🧠 Human vs 🤖 LLM: Latxak baino gehiago al dakizu?</h1>
    </div>

    <style>
    .responsive-header {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    flex-wrap: wrap;
    text-align: left;
    gap: 1rem;
    margin: 0;              /* elimina márgenes verticales */
    padding: 0;             /* elimina padding adicional */
    }

    /* Imagen */
    .responsive-header img {
    width: 120px;
    height: auto;
    flex-shrink: 0;
    margin: 0;
    }

    /* Texto */
    .responsive-header h1 {
    margin: 0;
    font-size: 1.8em;
    line-height: 1.2;
    }

    /* --- Vista móvil --- */
    @media (max-width: 700px) {
    .responsive-header {
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        gap: 0.5rem;
        margin: 0 !important;
        padding: 0 !important;
    }

    .responsive-header img {
        width: 90px !important;
    }

    .responsive-header h1 {
        font-size: 1.4em !important;
    }
    }
    </style>
    """
    gr.Markdown(markdown_content, elem_classes=["game-card"])


    with Modal(visible=False, allow_user_close=True) as result_modal:
        modal_text = gr.Markdown()
        close_btn = gr.Button("Itxi", variant="primary")
    with gr.Tabs() as tabs:
        with gr.Tab("Hasiera", id="inicio"):
            gr.Markdown("Sartu zure izena eta hasi jolasten!", elem_classes=["game-card"])
            with gr.Row():
                name_box = gr.Textbox(label="Izena", placeholder="Idatzi zure izena…", scale=2)
            start_btn = gr.Button("Hasi", elem_classes=["button-primary"])
            greet_md = gr.Markdown("", visible=False)

        with gr.Tab("Jokatu partida", id="juego"):
            with gr.Row():
                with gr.Column(scale=3):
                    header_md = gr.Markdown("", elem_classes=["score-badge"])
                with gr.Column(scale=1):
                    score_md = gr.Markdown("### Puntuazioa --\t\t Zu: 0 | Latxa: 0 | Latxa+Web: 0", elem_classes=["score-badge"])
            question_md = gr.Markdown("", elem_classes=["game-card"])
            with gr.Row():
                with gr.Column(scale=2):
                    options_radio = gr.Radio(choices=[], label="Aukerak", elem_classes=["option-radio", "game-card"])
                    correct_answer_md = gr.Markdown("", visible=False, elem_classes=["game-card"])
                with gr.Column(scale=1):
                    gr.Markdown("Latxaren erantzuna", elem_classes=["model-pill"], visible=True)
                    model_md = gr.Markdown("", visible=False, elem_classes=["game-card"])
                with gr.Column(scale=1):
                    gr.Markdown("Latxa+Web erantzuna", elem_classes=["model-pill"], visible=True)
                    modelRAG_md = gr.Markdown("", visible=False, elem_classes=["game-card"])
                with gr.Column(scale=1):
                    gr.Markdown("Zure erantzuna", elem_classes=["user-pill"], visible=True)
                    feedback_md = gr.Markdown("", visible=False, elem_classes=["game-card"])

            with gr.Row():
                submit_btn = gr.Button(" 1️⃣ Bidali", elem_classes=["button-primary"])
                next_btn = gr.Button(" 2️⃣ Hurrengoa", interactive=False, elem_classes=["button-secondary"])
                restart_btn = gr.Button("Partida berria", visible=False, elem_classes=["button-secondary"])
                gr.Markdown("", elem_classes=["spacer"], visible=restart_btn.visible)    

        with gr.Tab("Sailkapen orokorra", id="leaderboard"):
            gr.Markdown("## 🏆 Sailkapen orokorra", elem_classes=["game-card"])
            top_md = gr.DataFrame(value=None, interactive=False) #gr.Markdown("", elem_classes=["game-card"])
            rank_md = gr.DataFrame(value=None, interactive=False) #gr.Markdown("", elem_classes=["game-card"])
            refresh_btn = gr.Button("Eguneratu", elem_classes=["button-secondary"])

    # Eventos
    # Landing: cargar nombre desde BrowserState
    demo.load(lambda bs: gr.Textbox(value=(bs[0] if bs and bs[0] else "")),
            inputs=[browser_state], outputs=[name_box])
    # Botón empezar: inicializa estado y (si es posible) cambia a pestaña juego
    start_btn.click(
        on_start,
        inputs=[name_box, browser_state, session_state],
        outputs=[tabs, greet_md, session_state, browser_state],
    ).then(
        render_round, inputs=[session_state],
        outputs=[header_md, question_md, options_radio, score_md, correct_answer_md, feedback_md, model_md, modelRAG_md, submit_btn, next_btn, restart_btn]
    )

    # Enviar respuesta
    submit_btn.click(
        on_submit,
        inputs=[options_radio, session_state],
        outputs=[correct_answer_md, feedback_md, model_md, modelRAG_md, submit_btn, next_btn, options_radio],
    ).then(
        lambda state: f"### Puntuazioa -- Zu: {state['user_score']} | Latxa: {state['model_score'] }| Latxa+Web: {state['modelRAG_score']}",
        inputs=[session_state],
        outputs=[score_md]
    )

    close_btn.click(lambda: gr.update(visible=False), None, result_modal)

    # Siguiente
    next_btn.click(
        on_next,
        inputs=[session_state],
        outputs=[header_md, question_md, options_radio, score_md, correct_answer_md, feedback_md, model_md, modelRAG_md, submit_btn, next_btn, restart_btn]
    ).then(  # Añades .then() para controlar el modal después
        check_and_show_modal,
        inputs=[session_state],
        outputs=[result_modal, modal_text]
    )

    # Reiniciar
    restart_btn.click(
        on_restart,
        inputs=[session_state],
        outputs=[session_state, header_md, question_md, options_radio, score_md, correct_answer_md, feedback_md, model_md, modelRAG_md, submit_btn, next_btn, restart_btn]
    )

    # Leaderboard
    demo.load(refresh_leaderboard, None, [top_md, rank_md])
    refresh_btn.click(refresh_leaderboard, None, [top_md, rank_md])

if __name__ == "__main__":
    # Para concurrencia estable en llamadas al modelo
    demo.queue(max_size=64).launch(share=True)
