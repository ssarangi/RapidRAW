import json
from pathlib import Path

LOCALES_DIR = Path("./locales")

TRANSLATIONS = {
    "ca": {
        "editor": {
            "masks": {
                "aiTitle": "Seleccions d'IA",
                "basicTitle": "Eines bàsiques",
                "rangeTitle": "Rangs i global"
            }
        }
    },
    "de": {
        "editor": {
            "masks": {
                "aiTitle": "KI-Auswahl",
                "basicTitle": "Basis-Werkzeuge",
                "rangeTitle": "Bereiche & Global"
            }
        }
    },
    "en": {
        "editor": {
            "masks": {
                "aiTitle": "AI Selections",
                "basicTitle": "Basic Tools",
                "rangeTitle": "Ranges & Global"
            }
        }
    },
    "es": {
        "editor": {
            "masks": {
                "aiTitle": "Selecciones de IA",
                "basicTitle": "Herramientas básicas",
                "rangeTitle": "Rangos y global"
            }
        }
    },
    "fr": {
        "editor": {
            "masks": {
                "aiTitle": "Sélections IA",
                "basicTitle": "Outils de base",
                "rangeTitle": "Plages & Global"
            }
        }
    },
    "it": {
        "editor": {
            "masks": {
                "aiTitle": "Selezioni IA",
                "basicTitle": "Strumenti di base",
                "rangeTitle": "Intervalli e Globale"
            }
        }
    },
    "ja": {
        "editor": {
            "masks": {
                "aiTitle": "AI選択",
                "basicTitle": "基本ツール",
                "rangeTitle": "範囲とグローバル"
            }
        }
    },
    "ko": {
        "editor": {
            "masks": {
                "aiTitle": "AI 선택",
                "basicTitle": "기본 도구",
                "rangeTitle": "범위 및 글로벌"
            }
        }
    },
    "pl": {
        "editor": {
            "masks": {
                "aiTitle": "Zaznaczenia AI",
                "basicTitle": "Podstawowe narzędzia",
                "rangeTitle": "Zakresy i globalne"
            }
        }
    },
    "pt": {
        "editor": {
            "masks": {
                "aiTitle": "Seleções de IA",
                "basicTitle": "Ferramentas Básicas",
                "rangeTitle": "Intervalos e Global"
            }
        }
    },
    "ru": {
        "editor": {
            "masks": {
                "aiTitle": "Выделения ИИ",
                "basicTitle": "Базовые инструменты",
                "rangeTitle": "Диапазоны и глобальные"
            }
        }
    },
    "zh-CN": {
        "editor": {
            "masks": {
                "aiTitle": "AI 选择",
                "basicTitle": "基本工具",
                "rangeTitle": "范围与全局"
            }
        }
    },
    "zh-TW": {
        "editor": {
            "masks": {
                "aiTitle": "AI 選擇",
                "basicTitle": "基本工具",
                "rangeTitle": "範圍與全域"
            }
        }
    }
}

def deep_merge(target: dict, source: dict):
    """Recursively merges source dict into target dict."""
    for key, value in source.items():
        if isinstance(value, dict):
            node = target.setdefault(key, {})
            if isinstance(node, dict):
                deep_merge(node, value)
        else:
            target[key] = value

def sort_dict_recursively(item):
    if isinstance(item, dict):
        return {k: sort_dict_recursively(v) for k, v in sorted(item.items())}
    elif isinstance(item, list):
        return [sort_dict_recursively(x) for x in item]
    return item

def update_json_file(file_path: Path, trans: dict):
    if not file_path.exists():
        print(f"Skipping: {file_path.name} (File not found)")
        return

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        print(f"Error parsing JSON in {file_path.name}. Skipping.")
        return

    # 1. Merge new translations
    deep_merge(data, trans)

    # 2. Clean up removed keys from the diff
    if "editor" in data and "masks" in data["editor"]:
        data["editor"]["masks"].pop("createNewTitle", None)
        if "tooltips" in data["editor"]["masks"]:
            data["editor"]["masks"]["tooltips"].pop("showMore", None)

    # 3. Sort alphabetically
    sorted_data = sort_dict_recursively(data)

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sorted_data, f, ensure_ascii=False, indent=2)
        f.write("\n")

    print(f"Updated and Sorted: {file_path.name}")

def main():
    if not LOCALES_DIR.exists():
        print(f"Error: Locales directory '{LOCALES_DIR}' does not exist.")
        return

    print("Starting translation updates for Mask Panel sections...")
    for lang, trans in TRANSLATIONS.items():
        file_path = LOCALES_DIR / f"{lang}.json"
        update_json_file(file_path, trans)
    print("Done!")

if __name__ == "__main__":
    main()