from datasets import load_dataset
import json

# Загружаем датасет
print("Загрузка датасета...")
dataset = load_dataset("RicardoRei/wmt-mqm-human-evaluation", split="train")

# Фильтруем только пары английский -> русский
en_ru_data = [item for item in dataset if item["lp"] == "en-ru"]

print(f"Найдено {len(en_ru_data)} примеров для пары en-ru")

# Сортируем по MQM оценке (score) от минимальной (плохие) к максимальной (хорошие)
sorted_data = sorted(en_ru_data, key=lambda x: x["score"])

# Выбираем 20 примеров:
# - 7 самых плохих (низкие оценки)
# - 6 средних (оценки из середины)
# - 7 самых хороших (высокие оценки)

worst = sorted_data[:7]                    # худшие 7
medium_start = len(sorted_data) // 2 - 3   # середина
medium_end = len(sorted_data) // 2 + 3
medium = sorted_data[medium_start:medium_end]  # 6 средних
best = sorted_data[-7:]                    # лучшие 7

selected_examples = worst + medium + best

# Сохраняем в файл в читаемом формате
with open("test_examples_en_ru.txt", "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write("ТЕСТОВЫЕ ПРИМЕРЫ ДЛЯ ОЦЕНКИ КАЧЕСТВА ПЕРЕВОДА\n")
    f.write(f"Всего примеров: {len(selected_examples)}\n")
    f.write("Оценка MQM: чем МЕНЬШЕ число, тем ХУЖЕ перевод (0-100)\n")
    f.write("=" * 80 + "\n\n")
    
    for i, example in enumerate(selected_examples, 1):
        f.write(f"\n{'─' * 80}\n")
        f.write(f"ПРИМЕР #{i}\n")
        f.write(f"MQM Оценка: {example['score']:.2f} / 100\n")
        
        # Определяем категорию качества
        if example['score'] < 30:
            quality = "🔴 ПЛОХОЙ (много ошибок)"
        elif example['score'] < 70:
            quality = "🟡 СРЕДНИЙ (есть ошибки)"
        else:
            quality = "🟢 ХОРОШИЙ (мало ошибок)"
        f.write(f"Качество: {quality}\n")
        
        f.write(f"\n📝 ОРИГИНАЛ (англ):\n{example['src']}\n")
        f.write(f"\n🤖 МАШИННЫЙ ПЕРЕВОД:\n{example['mt']}\n")
        f.write(f"\n📖 ЭТАЛОННЫЙ ПЕРЕВОД (для справки):\n{example['ref']}\n")
        
        f.write("\n" + "─" * 80 + "\n")

# Также создаём JSON версию для удобного импорта
json_output = []
for example in selected_examples:
    json_output.append({
        "id": len(json_output) + 1,
        "original": example["src"],
        "machine_translation": example["mt"],
        "reference": example["ref"],
        "mqm_score": example["score"],
        "system": example["system"]
    })

with open("test_examples_en_ru.json", "w", encoding="utf-8") as f:
    json.dump(json_output, f, ensure_ascii=False, indent=2)

print("\n✅ Готово!")
print("📄 Файлы созданы:")
print("   - test_examples_en_ru.txt  (текстовый формат для чтения)")
print("   - test_examples_en_ru.json (JSON формат для импорта)")
print(f"\n📊 Статистика выбранных примеров:")
print(f"   - Плохие (оценки 0-30): {len([e for e in selected_examples if e['score'] < 30])}")
print(f"   - Средние (30-70): {len([e for e in selected_examples if 30 <= e['score'] < 70])}")
print(f"   - Хорошие (70-100): {len([e for e in selected_examples if e['score'] >= 70])}")
print(f"\n   - Минимальная оценка: {min(e['score'] for e in selected_examples):.2f}")
print(f"   - Максимальная оценка: {max(e['score'] for e in selected_examples):.2f}")
print(f"   - Средняя оценка: {sum(e['score'] for e in selected_examples)/len(selected_examples):.2f}")