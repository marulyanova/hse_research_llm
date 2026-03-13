import json
import random
from typing import Tuple, Dict, Any, Optional, List

from tool_env import ToolEnv
from data import Data

import re


class BookingEnv(ToolEnv):

    def __init__(self):
        super().__init__("appointment_booking")

        self.state = None
        self.done = False
        self.awaiting_confirmation = False
        self.pending_action = None

        # правило про галлюцинации, что модель не должна упоминать id и тд, которые не видела.
        # Поэтому запоминаем entities, которые появлялись в запросе пользователя
        self.known_entities = set()

        self.user_messages = []
        self.user_step = 0

    def reset(self, data: Data) -> str:
        """Установка состояния среды"""

        self.state = data["initial_state"].copy()
        self.done = False
        self.awaiting_confirmation = False  # правило про подтверждение от пользователя перед тем как вызвать тул на создание
        self.pending_action = None

        self.known_entities = {
            "class_types": set(),
            "dates": set(),
            "appointment_ids": set(),
            "client_ids": set(),
            "sex": set(),
        }

        self.user_messages = data["user_messages"]
        self.user_step = 0

        text = self.user_messages[self.user_step]

        # Извлечение сущностей из запроса пользователя
        if "yoga" in text:
            self.known_entities["class_types"].add("yoga")
        elif "dance" in text:
            self.known_entities["class_types"].add("dance")
        elif "stretching" in text:
            self.known_entities["class_types"].add("stretching")

        date_pattern = r"(\d{4}-\d{2}-\d{2})"
        client_pattern = r"client\s+(\d+)"
        sex_pattern = r"Sex:\s*(male|female)"

        date_match = re.search(date_pattern, text)
        client_match = re.search(client_pattern, text)
        sex_match = re.search(sex_pattern, text, re.IGNORECASE)

        if date_match:
            self.known_entities["dates"].add(date_match.group(1))

        if client_match:
            self.known_entities["client_ids"].add(int(client_match.group(1)))

        if sex_match:
            self.known_entities["sex"].add(sex_match.group(1).lower())

        return f"User request:\n{self.user_messages[self.user_step]}"

    def parse_tool_call(self, action: str):
        """Парсинг названия тула и аргументов"""

        if not action.startswith("TOOL_CALL"):
            return None

        try:
            payload = json.loads(action[len("TOOL_CALL ") :])
            return payload["name"], payload["args"]
        except:
            return None

    def check_availability(self, class_type, date, **kwargs):
        """Тул для проверки, можно ли записаться на занятие class_type в дату date"""

        # Агент зачастую передает лишние аргументы, сохраним их и проставим штраф пропорционально
        extra_args = 0
        if kwargs != {}:
            extra_args = len(kwargs)

        # правила, чтобы отвечать доступна запись/недоступна
        rules = {
            "dance": {"div": 5, "time": "18:00", "capacity": 15},
            "yoga": {"div": 3, "time": "19:00", "capacity": 10},
            "stretching": {"div": 4, "time": "18:30", "capacity": 8},
        }

        # Клиент назвал недопустимый вид тренировки
        if class_type not in rules:
            return {
                "valid": False,
                "reason": "unknown_class",
                "message": f"It is impossible to make an appointment for class type: {class_type}, date: {date}, because of unknown class type. Please choose one of yoga, dancing, streetching.",
                "extra_args": extra_args,
            }

        day = int(date.split("-")[-1])
        rule = rules[class_type]

        # Клиент назвал недопустимую дату для вида тренировки
        if day % rule["div"] != 0:
            return {
                "valid": False,
                "reason": "invalid_day",
                "message": f"It is impossible to make an appointment for class type: {class_type}, date: {date}, because of invalid date. Please choose another date.",
                "extra_args": extra_args,
            }

        current = self.state["schedule"].get(date, {}).get(class_type, 0)

        # Тренировка на дату уже полностью забита
        if current >= rule["capacity"]:
            return {
                "valid": False,
                "reason": "full",
                "message": f"It is impossible to make an appointment for class type: {class_type}, date: {date}, because of full class capacity. Please choose another date or class type.",
                "extra_args": extra_args,
            }

        # Если все ок
        return {
            "valid": True,
            "capacity_left": rule["capacity"] - current,
            "time": rule["time"],
            "message": f"It is possible to make an appointment for class type: {class_type}, date: {date}!",
            "extra_args": extra_args,
        }

    def create_appointment(self, class_type, date, client_id, sex):
        """Тул для создания записи на тренировку class_type в дату date клиента client_id"""

        appointment_id = f"{class_type}_{date}_{client_id}"  # формирование ID

        if date not in self.state["schedule"]:
            self.state["schedule"][date] = {}

        self.state["schedule"][date][class_type] = (
            self.state["schedule"][date].get(class_type, 0) + 1
        )

        # изменение состояния
        self.state["appointments"][appointment_id] = {
            "class_type": class_type,
            "date": date,
            "client_id": client_id,
        }

        return {
            "appointment_id": appointment_id,
            "message": f"Appointment for class type: {class_type}, date: {date}, client_id: {client_id} was created successfully!",
        }

    def cancel_appointment(self, appointment_id):
        """Тул для отмены записи по appointment_id"""

        if appointment_id not in self.state["appointments"]:
            return {
                "success": False,
                "message": f"Appointment {appointment_id} was not cancelled because appointment_id = {appointment_id} was not found.",
            }

        ap = self.state["appointments"].pop(appointment_id)

        self.state["schedule"][ap["date"]][ap["class_type"]] -= 1

        return {
            "success": True,
            "message": f"Appointment {appointment_id} was cancelled successfully!",
        }

    def step(self, action: str):
        """
        Реализация шага в среде
        Возвращает (message, reward, done/not done, json)
        """

        # если таргет действие уже выполнено
        if self.done:
            return "Episode finished.", 0, True, {}

        reward = -0.01
        # для хранения нарушений правил и вызова некорректных тулов
        info = {"policy_violation": False, "invalid_action": False}

        # TEXT MESSAGE
        if not action.startswith("TOOL_CALL"):

            text = action.lower()

            # если агент запросил подтверждение записи у пользователя
            if "confirm" in text:
                self.awaiting_confirmation = True
                reward += 0.1  # награда за то, что подтверждение запросил

            # если агент послал сообщение текстовое пользователю, то отвечаем след сообщением из сценария
            if self.user_step + 1 < len(self.user_messages):
                self.user_step += 1
                return (
                    f"User: {self.user_messages[self.user_step]}",
                    reward,
                    False,
                    info,
                )
            return "Message received.", reward, False, info

        # TOOL CALL
        tool = self.parse_tool_call(action)
        if tool is None:
            reward -= 0.1  # штраф за то, что написал TOOL_CALL, но вызов в неправильном формате/неполный
            info["invalid_action"] = True
            return "Invalid tool format, cannot be parsed as JSON", reward, False, info

        name, args = tool

        # hallucination rule
        hallucinations_percent, hallucinations = self.check_hallucination(name, args)
        if hallucinations_percent > 0.0:

            # штрафуем за галлюцинации (несуществующие сущности, вызов несуществующих тулов) в зависимости от кол-ва
            reward -= 0.2 * hallucinations_percent
            info["policy_violation"] = True

            print(
                "True Entities", self.known_entities, "Hallucinations", hallucinations
            )

            return "Hallucinated entity.", reward, False, info

        # confirmation rule
        if name in ["create_appointment", "cancel_appointment"]:
            # если агент вызвал создание, удаление, но ещё не спросил подтверждение
            if not self.awaiting_confirmation:
                reward -= 0.2
                info["policy_violation"] = True
                return (
                    "Before creation or cancelling an appointment, it is needed to request client for confirmation.",
                    reward,
                    False,
                    info,
                )

        # TOOL EXECUTION
        if name == "check_availability":
            result = self.check_availability(**args)
            # штраф за лишние переданные аргументы
            reward -= 0.05 * result["extra_args"]
            return str(result), reward, False, info

        if name == "create_appointment":
            result = self.create_appointment(**args)
            self.known_entities["appointment_ids"].add(result["appointment_id"])
            self.done = True
            reward += 1  # выполнено
            self.awaiting_confirmation = False  # сброс ожидания подтверждения
            return str(result), reward, True, {"success": True}

        if name == "cancel_appointment":
            result = self.cancel_appointment(**args)
            self.done = True
            if result["success"]:
                reward += 1
            else:
                reward -= 1
            return str(result), reward, True, {"success": result["success"]}

        # проверка на несуществующий тул уже есть в галлюцинациях
        # # если вообще несуществующий тул
        # reward -= 0.1
        # info["invalid_action"] = True
        # return "Unknown tool", reward, False, info

    def generate(self, num_of_questions=100):
        """
        Выбирает сложность и генерирует поведение клиента.

        Сложность 1: Сразу клиент говорит корректную дату для записи, записаться можно, подтверждение, запись
        Сложность 1: Сразу говорит некорректную дату для записи, записаться нельзя, агент должен попросить поменять дату, клиент меняет на корректную, подтверждение, запись
        """

        dataset = []

        for i in range(num_of_questions):

            difficulty = random.choices([1, 2], weights=[0.7, 0.3])[0]

            if difficulty == 1:
                msgs = self.generate_simple()
                state = {"schedule": {}, "appointments": {}}

            else:
                msgs = self.generate_invalid_date()
                state = {"schedule": {}, "appointments": {}}

            dataset.append(
                Data(
                    question_id=i,
                    user_messages=msgs,
                    initial_state=state,
                    difficulty=difficulty,
                )
            )

        return dataset

    def generate_simple(self):
        """
        Генерация примера сложности 1.
        Ожидаемое поведение на сложности 1:

        User: Book yoga on 2026-04-06 for client 12. Sex: male.
        Assistant: check_availability
        Assistant: ask confirmation
        User: Yes confirm.
        Assistant: create_appointment
        """

        class_type = random.choice(["yoga", "dance", "stretching"])

        rules = {"dance": 5, "yoga": 3, "stretching": 4}
        div = rules[class_type]

        day = random.randint(1, 10) * div
        date = f"2026-04-{day:02d}"
        client_id = random.randint(1, 100)
        sex = random.choice(["male", "female"])

        messages = [
            f"Book {class_type} on {date} for client {client_id}. Sex: {sex}.",
            "Yes, confirm the appointment.",
        ]

        return messages

    def generate_invalid_date(self):
        """
        Генерация примера сложности 2.
        Ожидаемое поведение на сложности 2:

        User: Book yoga on 2026-04-05
        Assistant: check_availability
        Tool: invalid_day
        Assistant: explain
        User: Ok try 2026-04-06
        Assistant: check_availability
        Assistant: ask confirmation
        User: Yes confirm.
        Assistant: create_appointment
        """

        class_type = random.choice(["yoga", "dance", "stretching"])

        rules = {"dance": 5, "yoga": 3, "stretching": 4}
        div = rules[class_type]
        invalid_day = random.randint(1, 30)

        while invalid_day % div == 0:
            invalid_day = random.randint(1, 30)

        valid_day = random.randint(1, 10) * div

        date1 = f"2026-04-{invalid_day:02d}"
        date2 = f"2026-04-{valid_day:02d}"

        client_id = random.randint(1, 100)
        sex = random.choice(["male", "female"])

        messages = [
            f"Book {class_type} on {date1} for client {client_id}. Sex: {sex}.",
            f"Ok, try {date2} instead.",
            "Yes confirm.",
        ]

        return messages

    def check_hallucination(self, tool_name, args):
        """Проверка, что в вызванный тул передали существующие аргументы и что название тула существует"""

        hallucinations = []

        if tool_name not in [
            "check_availability",
            "create_appointment",
            "cancel_appointment",
        ]:
            return 1.0, [tool_name]

        cnt = 0
        total = 0

        if tool_name == "check_availability":
            total = 2
            if args["class_type"] not in self.known_entities["class_types"]:
                cnt += 1
                hallucinations.append(args["class_type"])

            if args["date"] not in self.known_entities["dates"]:
                cnt += 1
                hallucinations.append(args["date"])

        if tool_name == "create_appointment":
            total = 4
            if args["class_type"] not in self.known_entities["class_types"]:
                cnt += 1
                hallucinations.append(args["class_type"])

            if args["date"] not in self.known_entities["dates"]:
                cnt += 1
                hallucinations.append(args["date"])

            if args["client_id"] not in self.known_entities["client_ids"]:
                cnt += 1
                hallucinations.append(args["client_id"])

            if args["sex"] not in self.known_entities["sex"]:
                cnt += 1
                hallucinations.append(args["sex"])

        if tool_name == "cancel_appointment":
            total = 1

            if args["appointment_id"] not in self.known_entities["appointment_ids"]:
                cnt += 1
                hallucinations.append(args["appointment_id"])

        percent = cnt / max(1, total)
        return percent, hallucinations
