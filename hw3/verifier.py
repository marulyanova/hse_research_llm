from typing import List
import re
import numpy as np


def extract_code_block(text: str) -> str:
    """
    Извлечение кода из markdown, если модель написала с markdown
    """

    pattern = r"```(?:python)?\s*\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1).strip()
    text = re.sub(r"</?answer>|</?think>", "", text)
    return text.strip()


def verify_solution(code: str, test_list: List[str], test_imports: str = "") -> bool:
    """
    Проверяет code, сгенерированный моделью на тестах. Возвращает True если все тесты прошли, False иначе

    code: сгенерированный код функции
    test_list: список assert-выражений (строки)
    test_imports: необходимые импорты для тестов
    """

    code = extract_code_block(code)
    full_code = test_imports + "\n" + code + "\n"

    try:
        namespace = {}
        exec(full_code, namespace)

        for test in test_list:
            if test.strip().startswith("assert "):
                test_expr = test.strip()[7:].strip()  # убираем "assert "
            else:
                test_expr = test.strip()

            result = eval(test_expr, namespace)
            if not result:
                return False
        return True

    except Exception as e:
        return False


def pass_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def calc_pass_k(num_samples: List[int], num_correct: List[int], k: int) -> float:
    """
    Подсчет метрики по всему набору задач
    """

    pass_values = [pass_k(n, c, k) for n, c in zip(num_samples, num_correct)]
    return np.mean(pass_values)
